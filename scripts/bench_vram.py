"""決定摘要模型選型的量測腳本（規格 §7、§14.1）。

規格 §0.1：規格書上所有 VRAM/延遲數字都是估計值。能量的就量。
這支腳本的輸出 data/bench.json 才是服務真正使用的參數來源。

用法：
    python scripts/bench_vram.py [--asr-model PATH] [--audio PATH] [--skip-final]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from datetime import datetime

from _common import (ROOT, Gate, header, merge_bench, note_deviation,
                     percentile, find_fixture)

from server import config, gpu
from server.llm_manager import LLMManager, LLMPortBusy, LLMUnavailable
from server.summarizer import FINAL_SYSTEM, Summarizer

# 規格 §7.2 的候選順序。取第一個通過的。
# GGUF 檔請放在 models/ 下；找不到檔案的候選會標記 MISSING 並跳過
# （所以可以只先下載 4B 跑起來，之後補 8B 再重跑一次）。
#
# 關於檔名與規格書的差異：規格 §7.2 寫的是 "Qwen3-8B-Instruct"、
# "Qwen3-4B-Instruct"、"Qwen3-1.7B-Instruct"，但 Qwen 官方 GGUF 只有
# Qwen3-8B / Qwen3-4B / Qwen3-1.7B（hybrid thinking 模型，沒有 -Instruct 後綴）。
# 真正的 instruct-only 版本只出到 4B（Qwen3-4B-Instruct-2507），
# 所以 4B 用 2507 instruct 版，8B 與 1.7B 用 hybrid 版並在啟動時關掉 thinking
# —— thinking 會吃掉 8 秒的延遲預算，也會污染 JSON 輸出（見 no_think 欄位）。
CANDIDATES = [
    {"key": "qwen3-8b-q4km-c8192",  "model": "Qwen3-8B-Q4_K_M.gguf",
     "quant": "Q4_K_M", "ctx": 8192, "device": "cuda", "no_think": True,
     "repo": "unsloth/Qwen3-8B-GGUF"},
    {"key": "qwen3-8b-q4km-c4096",  "model": "Qwen3-8B-Q4_K_M.gguf",
     "quant": "Q4_K_M", "ctx": 4096, "device": "cuda", "no_think": True,
     "repo": "unsloth/Qwen3-8B-GGUF"},
    {"key": "qwen3-4b-q4km-c8192",  "model": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
     "quant": "Q4_K_M", "ctx": 8192, "device": "cuda", "no_think": False,
     "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF"},
    {"key": "qwen3-4b-q40-c4096",   "model": "Qwen3-4B-Instruct-2507-Q4_0.gguf",
     "quant": "Q4_0", "ctx": 4096, "device": "cuda", "no_think": False,
     "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF"},
    {"key": "qwen3-1.7b-q4km-c4096", "model": "Qwen3-1.7B-Q4_K_M.gguf",
     "quant": "Q4_K_M", "ctx": 4096, "device": "cuda", "no_think": True,
     "repo": "unsloth/Qwen3-1.7B-GGUF"},
    {"key": "qwen3-4b-q4km-cpu",    "model": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
     "quant": "Q4_K_M", "ctx": 8192, "device": "cpu", "no_think": False,
     "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF"},
]

# 規格書的估計值，僅用於偏離標註（§0.1），不作為判定依據
SPEC_ESTIMATES = {"summary_latency_p95_s": 8.0}

VRAM_BUDGET = config.VRAM_BUDGET_RATIO      # 0.85
# 規格 §7.2 的 8 秒是為「上課中按按鈕、盯著螢幕等摘要」訂的。
# 若實際用途是「低頭寫筆記、不看螢幕」，這個上限可以放寬換取品質——
# 實測 4B 比 1.7B 少自我重複、會考重點更具體，代價只是多 2 秒。
# 用 --inclass-latency 調整，預設維持規格值。
LATENCY_BUDGET_S = float(os.getenv("LS_INCLASS_LATENCY_S", "8.0"))
FINAL_SUMMARY_BUDGET_S = 90.0   # 規格 §11 G3：15 則 section 的總結上限
FINAL_SAFETY = 0.7              # 留餘裕給 prompt 處理與模型冷啟動
CPU_LATENCY_WARN_S = 60.0

def _final_summary_payload():
    """組一份和正式流程同形狀的總結輸入：15 則 section（規格 §11 G3）。

    課後總結的速度不能用「5 次 400-token 生成」外推——那是課中段落摘要的
    形狀。總結是一次性、輸入長（15 則 section）、輸出短（3-5 句概述加
    最多 3 則待釐清）的呼叫，直接量它才對得上 G3 的 90 秒門檻。
    """
    from server.courses import Course
    course = Course(id="bench", name="機器學習", instructor="施皇嘉",
                    asr_prompt="", glossary=[])
    sections = []
    for i in range(15):
        sections.append({
            "start_s": i * 600.0,
            "title": "第 %d 段：%s" % (i + 1, SECTION_TITLES[i % len(SECTION_TITLES)]),
            "bullets": ["這一段講了 %s 的定義與適用情境" % SECTION_TITLES[i % len(SECTION_TITLES)],
                        "老師舉了一個 sklearn 的實作例子說明",
                        "提醒這個概念與前一節有關，考試會一起考"],
            "user_note": "這題會考" if i in (3, 9) else None,
        })
    return course, sections


SECTION_TITLES = [
    "課程導論與評分方式", "監督式與非監督式學習", "線性迴歸的最小平方解",
    "梯度下降的收斂性", "learning rate scheduling", "momentum 與 Adam",
    "overfitting 的成因", "L1 與 L2 regularization", "cross validation",
    "混淆矩陣與召回率", "ROC 曲線與 AUC", "決策樹與資訊增益",
    "隨機森林與 bagging", "特徵工程實務", "下週小考範圍",
]

SAMPLE_PROMPT = (
    "老師剛剛講到梯度下降的收斂性，說在凸函數的情況下一定會收斂到全域最小值，"
    "但實際上深度學習的 loss surface 不是凸的，所以只能保證收斂到局部最小值或鞍點。"
    "他又提到 learning rate 太大會震盪、太小會收斂很慢，"
    "所以實務上會用 learning rate scheduling，例如 cosine annealing 或 warmup。"
    "然後講了 momentum 跟 Adam 的差別，Adam 會對每個參數有自己的 adaptive learning rate。"
) * 3


def _stray_llama_servers():
    """找出不是我們啟的 llama-server 行程。

    只認 tools/llama.cpp 底下那一支。Ollama 的推論行程**檔名也叫
    llama-server.exe**，照映像名稱抓會把它一起算進來，於是在有裝 Ollama
    的機器上這道閘門永遠過不了。
    """
    import subprocess as sp
    if sys.platform != "win32":
        return []
    try:
        out = sp.run(["powershell", "-NoProfile", "-Command",
                      "Get-Process llama-server -ErrorAction SilentlyContinue"
                      " | ForEach-Object { \"$($_.Id)`t$($_.Path)\" }"],
                     capture_output=True, text=True, timeout=20).stdout
    except (sp.SubprocessError, OSError):
        return []
    ours = str(config.LLAMA_SERVER_BIN).lower()
    pids = []
    for line in out.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) != 2 or parts[1].strip().lower() != ours:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            pass
    return pids


class VramSampler:
    """背景執行緒持續採樣 nvidia-smi，取峰值（規格 §7.1：峰值出現在推論當下）。"""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.peak = gpu.used_mb()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu.used_mb())
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)
        return False


def load_audio(path):
    import numpy as np
    if path is None:
        # 沒有 fixture 時用合成訊號：VRAM 與 RTF 仍可量，文字內容無意義。
        n = 16000 * 20
        t = np.arange(n, dtype=np.float32) / 16000.0
        sig = 0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t))
        return sig.astype(np.float32), None
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype("float32")
    return data[:16000 * 60], path


def resolve_asr_model(explicit=None):
    if explicit:
        return explicit
    env = os.getenv("LS_ASR_MODEL")
    if env:
        return env
    for name in ("breeze-asr-25-ct2", "Breeze-ASR-25-int8-CT2", "whisper-large-v3-turbo-ct2"):
        p = config.MODELS_DIR / name
        if p.exists():
            return str(p)
    return None


async def measure_candidate(cand, summarizer, llm, asr_engine, audio,
                            gate, keep_asr_busy, model_key, baseline_mb):
    """啟動候選模型、量測 combined peak 與延遲。回傳結果 dict。

    peak_mb 一律是「扣掉桌面基準之後、模型自己佔的量」，
    才能跟 available_mb（= total - baseline）直接比較。
    asr 的 idle_mb / peak_mb 也是同一個基準，兩邊要一致。
    """
    path = config.MODELS_DIR / cand["model"]
    if not path.exists():
        gate.info("跳過 %s：找不到 %s（可從 HuggingFace %s 下載）"
                  % (cand["key"], path.name, cand["repo"]))
        return {"model": cand["model"], "key": cand["key"], "quant": cand["quant"],
                "ctx": cand["ctx"], "device": cand["device"],
                "no_think": bool(cand.get("no_think")), "result": "MISSING"}

    llm.bench = _FakeBench(cand)
    latencies = []
    final_s = None
    try:
        with VramSampler() as sampler:
            try:
                await llm.ensure(model_key)
            except LLMPortBusy:
                raise            # 環境問題，不是「這個模型塞不下」
            except LLMUnavailable as e:
                gate.info("%s 啟動失敗：%s" % (cand["key"], e))
                return {"model": cand["model"], "key": cand["key"],
                        "quant": cand["quant"], "ctx": cand["ctx"],
                        "device": cand["device"],
                        "no_think": bool(cand.get("no_think")),
                        "result": "OOM", "error": str(e)}

            asr_task = None
            if keep_asr_busy and asr_engine is not None:
                asr_task = asyncio.create_task(_asr_load_loop(asr_engine, audio))
            try:
                # 生成 5 則約 400 token 的摘要（規格 §14.1 步驟 4）
                for _ in range(5):
                    t0 = time.monotonic()
                    try:
                        await summarizer._chat(
                            "你是課堂筆記助理，請用繁體中文條列重點。",
                            SAMPLE_PROMPT, 400, 0.3, None)
                    except LLMUnavailable as e:
                        gate.info("%s 生成失敗：%s" % (cand["key"], e))
                        return {"model": cand["model"], "key": cand["key"],
                                "quant": cand["quant"], "ctx": cand["ctx"],
                                "device": cand["device"],
                                "no_think": bool(cand.get("no_think")),
                                "result": "OOM", "error": str(e)}
                    latencies.append(time.monotonic() - t0)

                # 課後階段：再量一次真實形狀的總結呼叫（規格 §11 G3）
                if model_key == "final":
                    course, sections = _final_summary_payload()
                    t0 = time.monotonic()
                    try:
                        await summarizer._chat(
                            FINAL_SYSTEM,
                            Summarizer.build_final_user(course, sections),
                            config.FINAL_MAX_TOKENS, 0.3, None)
                        final_s = time.monotonic() - t0
                    except LLMUnavailable as e:
                        gate.info("%s 總結生成失敗：%s" % (cand["key"], e))
            finally:
                if asr_task is not None:
                    asr_task.cancel()
                    try:
                        await asr_task
                    except (asyncio.CancelledError, Exception):
                        pass
        combined_peak = sampler.peak
    finally:
        await llm.shutdown()

    p95 = percentile(latencies, 0.95)
    mean = sum(latencies) / len(latencies) if latencies else 0.0
    tps = (400.0 / mean) if mean > 0 else 0.0
    out = {"model": cand["model"], "key": cand["key"], "quant": cand["quant"],
           "ctx": cand["ctx"], "device": cand["device"],
           "no_think": bool(cand.get("no_think")),
           "peak_mb": round(max(0.0, combined_peak - baseline_mb), 1),
           "peak_abs_mb": round(combined_peak, 1),
           "tps": round(tps, 1),
           "latency_p95_s": round(p95, 2), "latency_mean_s": round(mean, 2)}
    if final_s is not None:
        out["final_summary_s"] = round(final_s, 2)
    return out


class _FakeBench:
    """讓 LLMManager 依候選設定啟動，不必先有定案的 bench.json。"""

    def __init__(self, cand):
        from server.config import ModelChoice
        self._c = ModelChoice(model=cand["model"], quant=cand["quant"],
                              ctx=cand["ctx"], device=cand["device"],
                              n_gpu_layers=0 if cand["device"] == "cpu" else -1,
                              no_think=bool(cand.get("no_think")))

    @property
    def inclass_model(self):
        return self._c

    @property
    def final_model(self):
        return self._c


async def _asr_load_loop(engine, audio):
    """在摘要生成期間持續跑 ASR，模擬課堂真實負載。"""
    while True:
        try:
            await asyncio.to_thread(engine.transcribe_joined, audio, "")
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1)
        await asyncio.sleep(0.1)


def _latency_ok(model_key, res, cand):
    """兩個階段的速度標準不同，套錯會白白犧牲品質。

    課中（規格 §7.2）：一則約 400 token 的段落摘要要在 8 秒內完成，
    因為使用者按了按鈕正在等，而且下一段課還在進行。

    課後總結（規格 §11 G3）：標準是「15 則 section 的總結 < 90 秒」。
    這是下課後一次性的動作，使用者看著進度條等，規格 §7.2 甚至明講
    8B 冷啟動 10–20 秒是可以接受的。拿課中的 8 秒去卡它，會把 8B
    誤判為太慢而退回小模型——正好違背 §7.3「總結模型可以比課中大一級，
    這是本架構刻意換來的品質，不要偷懶」。
    """
    p95 = res["latency_p95_s"]
    if cand["device"] == "cpu":
        return (p95 < CPU_LATENCY_WARN_S,
                "純 CPU P95 %.1fs 超過 %.0fs 上限" % (p95, CPU_LATENCY_WARN_S))
    if model_key == "inclass":
        return (p95 < LATENCY_BUDGET_S,
                "P95 延遲 %.1fs 超過課中的 %.0fs 上限" % (p95, LATENCY_BUDGET_S))

    measured = res.get("final_summary_s")
    if measured is None:
        return False, "量不到總結生成時間"
    limit = FINAL_SUMMARY_BUDGET_S * FINAL_SAFETY
    return (measured < limit,
            "15 則 section 的總結實測 %.1fs，超過 %.0fs"
            "（G3 門檻 90s，留 %.0f%% 餘裕給冷啟動與 prompt 處理）"
            % (measured, limit, (1 - FINAL_SAFETY) * 100))


async def run_phase(name, gate, summarizer, llm, asr_engine, audio,
                    available_mb, keep_asr_busy, model_key, baseline_mb):
    tried = []
    selected = None
    for cand in CANDIDATES:
        if selected is not None:
            break
        gate.info("測試候選：%s（%s, ctx=%d, %s）"
                  % (cand["key"], cand["quant"], cand["ctx"], cand["device"]))
        res = await measure_candidate(cand, summarizer, llm, asr_engine, audio,
                                      gate, keep_asr_busy, model_key, baseline_mb)
        if res.get("result") in ("MISSING", "OOM"):
            tried.append(res)
            continue

        peak_ok = (cand["device"] == "cpu"
                   or res["peak_mb"] <= available_mb * VRAM_BUDGET)
        lat_ok, lat_why = _latency_ok(model_key, res, cand)

        if not peak_ok:
            res["result"] = "OOM"
            gate.info("  → 模型峰值 %.0fMB（絕對 %.0fMB）超過預算 %.0fMB"
                      "（可用 %.0f × %.0f%%）"
                      % (res["peak_mb"], res.get("peak_abs_mb", 0),
                         available_mb * VRAM_BUDGET,
                         available_mb, VRAM_BUDGET * 100))
        elif not lat_ok:
            res["result"] = "SLOW"
            gate.info("  → %s" % lat_why)
        else:
            res["result"] = "SELECTED"
            selected = res
            extra_info = ("，總結 %.1fs" % res["final_summary_s"]
                          if res.get("final_summary_s") is not None else "")
            gate.info("  → SELECTED（峰值 %.0fMB，P95 %.1fs，%.0f tok/s%s）"
                      % (res["peak_mb"], res["latency_p95_s"], res["tps"],
                         extra_info))
        tried.append(res)

    if selected is not None and selected["device"] == "cpu":
        gate.info("警告：%s 只有純 CPU 方案可用，一則摘要約 %.0fs。"
                  "若無法接受，規格 §7.2 建議改用 API。"
                  % (name, selected["latency_p95_s"]))
    return selected, tried


async def main_async(args) -> int:
    header("bench_vram.py — 量測硬體實際能力，決定摘要模型選型（規格 §7、§14.1）")
    gate = Gate("BENCH")

    # 0. 前置檢查：確定量測環境是乾淨的 -----------------------------
    #    上次沒關乾淨的 llama-server 會同時汙染兩件事：吃掉 VRAM 讓
    #    baseline 虛高，以及佔著 port 讓每個候選的 /health 都打到它，
    #    結果就是整份 bench 都在量同一個陌生行程。
    stray = _stray_llama_servers()
    if stray:
        gate.check(False, "量測環境乾淨（沒有殘留的 llama-server）",
                   "偵測到 %d 個殘留行程 PID=%s。請先關掉再重跑，"
                   "否則量出來的數字全部無效。" % (len(stray), stray))
        return gate.finish()
    gate.check(True, "量測環境乾淨（沒有殘留的 llama-server）", "無殘留行程")

    # 1. GPU 基準值 -------------------------------------------------
    info = gpu.query()
    if info is None:
        gate.check(False, "讀取 GPU 資訊", "找不到 nvidia-smi 或無 NVIDIA GPU")
        return gate.finish()
    available = info.total_mb - info.used_mb
    gate.check(True, "讀取 GPU 資訊",
               "%s：total %.0fMB，桌面等既有佔用 %.0fMB，可用 %.0fMB"
               % (info.name, info.total_mb, info.used_mb, available))
    merge_bench({"measured_at": datetime.now().astimezone().isoformat(),
                 "gpu": {"name": info.name, "total_mb": info.total_mb,
                         "baseline_used_mb": info.used_mb,
                         "available_mb": round(available, 1)}})

    # 2./3. ASR 穩態與峰值 -----------------------------------------
    asr_path = resolve_asr_model(args.asr_model)
    if asr_path is None:
        gate.check(False, "找到 ASR 模型",
                   "請先取得 Breeze-ASR-25 的 CTranslate2 版本放到 models/"
                   "（規格 §3.3），或用 --asr-model 指定路徑")
        return gate.finish()

    audio_path = args.audio or find_fixture("bench.wav", "m0_sample.wav", "lecture_3h.wav")
    audio, used_path = load_audio(audio_path)
    if used_path is None:
        gate.info("找不到音檔 fixture，改用合成訊號（VRAM/RTF 仍有效，文字內容無意義）")

    from server.asr import ASREngine
    engine = ASREngine(asr_path, device="cuda", compute_type=args.compute_type)
    try:
        with VramSampler() as s_idle:
            await asyncio.to_thread(engine.load)
            await asyncio.sleep(1.5)
        asr_idle = s_idle.peak - info.used_mb
    except Exception as e:
        gate.check(False, "載入 ASR 模型", "%s：%s" % (asr_path, e))
        return gate.finish()
    gate.check(True, "載入 ASR 模型", "%s，穩態佔用約 %.0fMB" % (asr_path, asr_idle))

    rtfs = []
    with VramSampler() as s_peak:
        for _ in range(10):
            t0 = time.monotonic()
            await asyncio.to_thread(engine.transcribe_joined, audio, "")
            dur = len(audio) / 16000.0
            rtfs.append((time.monotonic() - t0) / dur)
    asr_peak = s_peak.peak - info.used_mb
    rtf = sum(rtfs) / len(rtfs)
    gate.check(True, "ASR 推論 10 次",
               "峰值佔用約 %.0fMB（比穩態高 %.0fMB），平均 RTF %.3f"
               % (asr_peak, asr_peak - asr_idle, rtf))
    merge_bench({"asr": {"model": asr_path, "device": "cuda",
                         "compute_type": args.compute_type,
                         "idle_mb": round(asr_idle, 1),
                         "peak_mb": round(asr_peak, 1),
                         "rtf": round(rtf, 4)}})

    llm = LLMManager(bench=_FakeBench(CANDIDATES[0]))
    summarizer = Summarizer(llm)

    # 4. 課中模型：ASR 已載入的前提下測試 --------------------------
    inclass, tried_in = None, []
    if args.phase in ("both", "inclass"):
        header("課中摘要模型（ASR 常駐）")
        inclass, tried_in = await run_phase("課中模型", gate, summarizer, llm, engine,
                                            audio, available, True, "inclass",
                                            info.used_mb)
        gate.check(inclass is not None, "課中摘要模型選型",
                   ("選定 %s（%s, ctx=%d, %s）"
                    % (inclass["model"], inclass["quant"],
                       inclass["ctx"], inclass["device"]))
                   if inclass else "所有候選都不通過，請確認 models/ 下有 GGUF 檔")
    else:
        gate.info("--phase %s：沿用 bench.json 既有的課中選型" % args.phase)
    if inclass:
        note_deviation("summary_latency_p95_s",
                       SPEC_ESTIMATES["summary_latency_p95_s"],
                       inclass["latency_p95_s"])
        merge_bench({"summary_model_inclass": {
            "model": inclass["model"], "quant": inclass["quant"],
            "ctx": inclass["ctx"], "device": inclass["device"],
            "n_gpu_layers": 0 if inclass["device"] == "cpu" else -1,
            "no_think": inclass.get("no_think", False),
            "peak_mb": inclass["peak_mb"], "tps": inclass["tps"],
            "latency_p95_s": inclass["latency_p95_s"]}})

    # 5. 卸載 ASR，決定課後總結模型（規格 §7.3）--------------------
    tried_fin = []
    final = None
    if not args.skip_final and args.phase in ("both", "final"):
        header("課後總結模型（ASR 已卸載，可用 VRAM 大增）")
        await asyncio.to_thread(engine.unload)
        await asyncio.sleep(3)
        after = gpu.query()
        freed = (info.used_mb + asr_idle) - (after.used_mb if after else 0)
        gate.info("卸載 ASR 後 VRAM 使用 %.0fMB（釋放約 %.0fMB）"
                  % (after.used_mb if after else 0, max(0.0, freed)))
        available_final = (after.total_mb - after.used_mb) if after else available
        final, tried_fin = await run_phase("總結模型", gate, summarizer, llm, None,
                                           audio, available_final, False, "final",
                                           after.used_mb if after else info.used_mb)
        gate.check(final is not None, "課後總結模型選型",
                   ("選定 %s（%s, ctx=%d）" % (final["model"], final["quant"],
                                             final["ctx"]))
                   if final else "所有候選都不通過")
        if final:
            merge_bench({"summary_model_final": {
                "model": final["model"], "quant": final["quant"],
                "ctx": final["ctx"], "device": final["device"],
                "n_gpu_layers": 0 if final["device"] == "cpu" else -1,
                "no_think": final.get("no_think", False),
                "peak_mb": final["peak_mb"], "tps": final["tps"],
                "latency_p95_s": final["latency_p95_s"]}})
    elif args.phase == "inclass":
        gate.info("--phase inclass：保留 bench.json 既有的總結選型")
    else:
        gate.info("--skip-final：沿用課中模型作為總結模型")
        if inclass:
            merge_bench({"summary_model_final": {
                "model": inclass["model"], "quant": inclass["quant"],
                "ctx": inclass["ctx"], "device": inclass["device"],
                "n_gpu_layers": 0 if inclass["device"] == "cpu" else -1,
                "no_think": inclass.get("no_think", False),
                "peak_mb": inclass["peak_mb"], "tps": inclass["tps"]}})

    merge_bench({"candidates_tried": tried_in + tried_fin})
    await llm.shutdown()
    print("\n結果已寫入 %s" % (ROOT / "data" / "bench.json"))
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-model", default=None, help="CT2 格式的 ASR 模型目錄")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--audio", default=None, help="量測用音檔（16kHz wav 最佳）")
    ap.add_argument("--skip-final", action="store_true",
                    help="跳過課後總結模型量測，沿用課中模型")
    ap.add_argument("--phase", choices=["both", "inclass", "final"], default="both",
                    help="只重跑其中一個階段（bench.json 會保留另一階段的結果）")
    ap.add_argument("--inclass-latency", type=float, default=None,
                    help="課中摘要的 P95 延遲上限（秒）。規格 §7.2 是 8，"
                         "但那假設使用者盯著螢幕等；若是邊聽邊寫可放寬換品質")
    args = ap.parse_args()
    if args.inclass_latency:
        global LATENCY_BUDGET_S
        LATENCY_BUDGET_S = args.inclass_latency
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
