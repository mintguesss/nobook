"""llama-server 子行程的生命週期與模型切換（規格 §7.2）。

llama-server 是獨立行程、非 Python in-process，所以「切模型」＝
kill 舊行程 → 等 port 釋放 → 啟新行程 → 輪詢 /health 直到就緒。
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
import time

import httpx

from . import config, winjob

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """llama-server 起不來或已死。摘要降級，逐字稿必須續行（規格 §7.4）。"""


class LLMPortBusy(LLMUnavailable):
    """port 被陌生行程佔用，或那個行程服務的是別的模型。

    這是環境問題不是資源問題：量測腳本必須大聲中止，
    不能當成「這個候選塞不下」而繼續往下試——否則整份 bench 都在
    量同一個陌生行程，結論看起來合理但完全是錯的。
    """


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) != 0


def _same_model(served: str, wanted_path: str) -> bool:
    """llama-server 回報的路徑格式各版本不一，比對檔名就夠。"""
    return os.path.basename(str(served).replace("\\", "/")) == \
        os.path.basename(wanted_path.replace("\\", "/"))


class LLMManager:
    """單一 llama-server 行程，依 model_key 在課中／課後模型間切換。

    model_key: "inclass" | "final"，對應 bench.json 的兩組選型。
    """

    def __init__(self, bench=None, bin_path=None, host=None, port=None):
        self.bench = bench if bench is not None else config.load_bench()
        self.bin = bin_path or config.LLAMA_SERVER_BIN
        self.host = host or config.LLAMA_HOST
        self.port = int(port or config.LLAMA_PORT)
        self.base_url = "http://%s:%d" % (self.host, self.port)
        self._proc = None
        self._current = None          # 目前服務中的 model_key
        self._lock = asyncio.Lock()
        self.degraded = False         # True = 摘要功能不可用
        self.last_switch_s = 0.0
        self.no_think_flag_ok = False
        self.current_ctx = 0          # 供 summarizer 算 max_tokens 用

    # 查詢 -------------------------------------------------------------
    @property
    def current(self):
        return self._current

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def choice_for(self, model_key: str):
        choice = (self.bench.inclass_model if model_key == "inclass"
                  else self.bench.final_model)
        if choice is None:
            raise LLMUnavailable(
                "bench.json 沒有 summary_model_%s，請先跑 scripts/bench_vram.py" % model_key)
        return choice

    # 生命週期 ---------------------------------------------------------
    async def ensure(self, model_key: str) -> None:
        """確保指定模型正在服務中。若當前是別的模型，先關閉再啟動。"""
        async with self._lock:
            if self._current == model_key and self.alive and await self._healthy():
                return
            t0 = time.monotonic()
            await self._stop_locked()
            choice = self.choice_for(model_key)
            await self._start_locked(model_key, choice)
            self.last_switch_s = time.monotonic() - t0
            log.info("llama-server 已切換至 %s（%s，%.1fs）",
                     model_key, choice.model, self.last_switch_s)

    async def shutdown(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _start_locked(self, model_key, choice) -> None:
        model_path = choice.resolve_model_path()
        if not os.path.exists(model_path):
            self.degraded = True
            raise LLMUnavailable("找不到模型檔：%s" % model_path)

        # 先確認 port 是空的。若有別人（例如上次沒關乾淨的 llama-server）
        # 佔著，/health 會回 200，我們就會把那個陌生行程當成自己的——
        # 量測全錯、正式服務還會靜靜地用到錯的模型。寧可大聲失敗。
        if not _port_free(self.host, self.port):
            self.degraded = True
            raise LLMPortBusy(
                "port %d 已被其他行程佔用（可能是上次沒關乾淨的 llama-server）。"
                "請先關掉它，或改設 LS_LLAMA_PORT。" % self.port)

        ngl = 0 if choice.device == "cpu" else choice.n_gpu_layers
        base = [
            self.bin,
            "-m", model_path,
            "-c", str(choice.ctx),
            "-ngl", str(ngl),
            "--host", self.host,
            "--port", str(self.port),
            "--no-webui",
            "-t", str(max(1, (os.cpu_count() or 4) // 2)),
        ]
        # Qwen3 hybrid 模型預設會輸出 <think>…</think>，摘要用不到，而且會吃掉
        # 延遲預算與 JSON 格式。旗標名稱在 llama.cpp 各版本間換過，起不來就退掉
        # 重試一次——summarizer 另外也會剝除 <think>，這裡失敗不致命。
        extras = ["--reasoning-budget", "0"] if choice.no_think else []

        for attempt, extra in enumerate(([extras] if extras else []) + [[]]):
            cmd = base + extra
            log.info("啟動 llama-server：%s", " ".join(cmd))
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=(subprocess.DEVNULL if not os.getenv("LS_LLAMA_VERBOSE")
                            else None),
                    stderr=(subprocess.STDOUT if not os.getenv("LS_LLAMA_VERBOSE")
                            else None),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                                   if sys.platform == "win32" else 0),
                )
            except (OSError, ValueError) as e:
                self.degraded = True
                raise LLMUnavailable("無法啟動 llama-server（%s）：%s" % (self.bin, e))

            # 綁進 Job Object：主行程被強制終止時，核心會一起收掉 llama-server，
            # 不會留下孤兒佔著 VRAM 與 port（README「量測時踩到的坑」第 1 點）
            winjob.assign(self._proc.pid)

            if await self._wait_healthy(config.LLAMA_STARTUP_TIMEOUT_S):
                served = await self._served_model()
                if served is not None and not _same_model(served, model_path):
                    await self._stop_locked()
                    self.degraded = True
                    raise LLMPortBusy(
                        "port %d 上的 llama-server 服務的是 %s，不是我們要的 %s"
                        % (self.port, served, os.path.basename(model_path)))
                self._current = model_key
                self.degraded = False
                self.no_think_flag_ok = bool(extra)
                self.current_ctx = choice.ctx
                return
            await self._stop_locked()
            if extra:
                log.warning("帶 %s 啟動失敗，改用不帶該旗標的參數重試"
                            "（改由 summarizer 剝除 <think>）", " ".join(extra))

        self.degraded = True
        raise LLMUnavailable("llama-server 在 %.0fs 內未就緒（模型 %s）"
                             % (config.LLAMA_STARTUP_TIMEOUT_S, model_path))

    async def _stop_locked(self) -> None:
        self._current = None
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        for _ in range(50):
            if proc.poll() is not None:
                break
            await asyncio.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # 等 port 真的釋放，否則新行程會 bind 失敗
        for _ in range(100):
            if _port_free(self.host, self.port):
                return
            await asyncio.sleep(0.1)
        log.warning("port %d 在 kill 後 10 秒仍未釋放", self.port)

    # 健康檢查 ---------------------------------------------------------
    async def _healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(self.base_url + "/health")
                return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def _served_model(self):
        """問 llama-server 現在到底載了哪個模型。拿不到就回 None（不阻擋啟動）。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(self.base_url + "/v1/models")
                if r.status_code != 200:
                    return None
                data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return None
        for key in ("data", "models"):
            items = data.get(key) or []
            if items:
                first = items[0]
                return first.get("id") or first.get("model") or first.get("name")
        return None

    async def _wait_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                log.error("llama-server 行程提前結束，exit code=%s", self._proc.returncode)
                return False
            if await self._healthy():
                return True
            await asyncio.sleep(0.5)
        return False

    async def health(self) -> bool:
        ok = self.alive and await self._healthy()
        if not ok and self._current is not None:
            self.degraded = True
        return ok

    # 測試用 -----------------------------------------------------------
    def _force_kill(self) -> None:
        """混沌測試注入點（規格 §14.6）：不清狀態直接砍行程。"""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            log.warning("llama-server 被 _force_kill() 砍掉（測試注入）")
        self.degraded = True


_manager = None


def get_manager():
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


def set_manager(m) -> None:
    global _manager
    _manager = m
