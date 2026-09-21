"""環境變數、路徑常數，以及 bench.json 的載入。

規格 §0.1：所有與硬體能力相關的參數（模型選型、ctx、device）都從
data/bench.json 讀取，不 hard-code。§14.1：服務啟動時若 bench.json
不存在，拒絕啟動並提示先跑 bench。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
COURSES_DIR = ROOT / "courses"
# 上課用的投影片／講義丟這裡，系統自己去對應是哪一堂、對到第幾頁
MATERIALS_DIR = Path(os.getenv("LS_MATERIALS_DIR", ROOT / "materials"))
WEB_DIR = ROOT / "web"
BENCH_PATH = DATA_DIR / "bench.json"
DB_PATH = Path(os.getenv("LS_DB_PATH", DATA_DIR / "lecture.db"))

# ── 服務 ────────────────────────────────────────────────────────────────
HOST = os.getenv("LS_HOST", "127.0.0.1")
PORT = int(os.getenv("LS_PORT", "8000"))


def _find_llama_server() -> str:
    """LS_LLAMA_SERVER_BIN > tools/llama.cpp 下的解壓結果 > PATH。"""
    env = os.getenv("LS_LLAMA_SERVER_BIN")
    if env:
        return env
    local = ROOT / "tools" / "llama.cpp"
    if local.is_dir():
        for name in ("llama-server.exe", "llama-server"):
            direct = local / name
            if direct.exists():
                return str(direct)
            hits = sorted(local.rglob(name))
            if hits:
                return str(hits[0])
    return "llama-server"


# llama-server 子行程
LLAMA_SERVER_BIN = _find_llama_server()
LLAMA_HOST = os.getenv("LS_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LS_LLAMA_PORT", "8080"))
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
LLAMA_STARTUP_TIMEOUT_S = float(os.getenv("LS_LLAMA_STARTUP_TIMEOUT_S", "120"))

# ── 音訊 / VAD 切段（規格 §4.2）────────────────────────────────────────
SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512           # silero-vad 在 16kHz 只吃 512 sample（=32ms）
VAD_FRAME_MS = VAD_FRAME_SAMPLES * 1000 // SAMPLE_RATE
VAD_THRESHOLD = float(os.getenv("LS_VAD_THRESHOLD", "0.5"))
MIN_SILENCE_MS = int(os.getenv("LS_MIN_SILENCE_MS", "600"))
MIN_SEGMENT_MS = int(os.getenv("LS_MIN_SEGMENT_MS", "1000"))
# MAX_SEGMENT_MS 有一個規格沒寫、但不能違反的上限：30 秒。
#
# faster-whisper 以 30 秒為一個解碼窗口。當 condition_on_previous_text=False
# 時（規格 §4.3 要求必須關閉，否則長音訊會進入重複輸出迴圈），它在處理完
# 第一個窗口後就把 prompt_reset_since 推到 token 尾端，於是 initial_prompt
# 對第二個窗口之後完全失效。
#
# 也就是說規格 §4.3 的兩項要求在實作上互相牽制：關掉 condition_on_previous_text
# 會連帶讓「本專案準確度的最大槓桿」只作用在每段的前 30 秒。
# 只要切段上限守在 30 秒內，每次 transcribe 只有一個窗口，prompt 就全程有效。
MAX_SEGMENT_MS = min(30000, int(os.getenv("LS_MAX_SEGMENT_MS", "15000")))
PRE_ROLL_MS = int(os.getenv("LS_PRE_ROLL_MS", "200"))

# ── 摘要（規格 §6）─────────────────────────────────────────────────────
MIN_SECTION_S = float(os.getenv("LS_MIN_SECTION_S", "30"))   # 短於此則併入前一段
SUMMARY_TEMPERATURE = float(os.getenv("LS_SUMMARY_TEMPERATURE", "0.3"))
SUMMARY_MAX_TOKENS = int(os.getenv("LS_SUMMARY_MAX_TOKENS", "800"))
# 每幾字的逐字稿產生一個要點。數字越小筆記越詳細。
# 實測 180 時，6700 字的逐字稿約產出 20 個要點（原本寫死 6 則只有 3.7% 壓縮比）
CHARS_PER_BULLET = int(os.getenv("LS_CHARS_PER_BULLET", "180"))
# 手抄版的目標字數。原本寫死 250 字，使用者反映太少。
HANDCOPY_TARGET_CHARS = int(os.getenv("LS_HANDCOPY_CHARS", "700"))
FINAL_MAX_TOKENS = int(os.getenv("LS_FINAL_MAX_TOKENS", "3000"))

# ── 執行期防護（規格 §7.4）─────────────────────────────────────────────
VRAM_HEADROOM_MB = int(os.getenv("LS_VRAM_HEADROOM_MB", "400"))
VRAM_BUDGET_RATIO = float(os.getenv("LS_VRAM_BUDGET_RATIO", "0.85"))
STATS_INTERVAL_S = float(os.getenv("LS_STATS_INTERVAL_S", "5"))

# ── 其他 ───────────────────────────────────────────────────────────────
SAVE_AUDIO = os.getenv("LS_SAVE_AUDIO", "0") == "1"   # §10：預設不保存原始音訊
AUDIO_DIR = DATA_DIR / "audio"


class BenchMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelChoice:
    """bench.json 中一組被選定的摘要模型設定。"""
    model: str          # GGUF 檔路徑或名稱
    quant: str
    ctx: int
    device: str = "cuda"
    n_gpu_layers: int = -1
    peak_mb: float = 0.0
    tps: float = 0.0
    # Qwen3 的 hybrid thinking 模型要關掉思考鏈：它會吃掉延遲預算並污染 JSON 輸出
    no_think: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelChoice":
        return cls(
            model=d["model"],
            quant=d.get("quant", ""),
            ctx=int(d.get("ctx", 4096)),
            device=d.get("device", "cuda"),
            n_gpu_layers=int(d.get("n_gpu_layers", 0 if d.get("device") == "cpu" else -1)),
            peak_mb=float(d.get("peak_mb", 0.0)),
            tps=float(d.get("tps", 0.0)),
            no_think=bool(d.get("no_think", False)),
        )

    def resolve_model_path(self) -> str:
        p = Path(self.model)
        if not p.is_absolute():
            candidate = MODELS_DIR / self.model
            if candidate.exists():
                return str(candidate)
        return str(p)


@dataclass(frozen=True)
class Bench:
    raw: dict[str, Any]

    @property
    def gpu(self) -> dict[str, Any]:
        return self.raw.get("gpu", {})

    @property
    def available_mb(self) -> float:
        return float(self.gpu.get("available_mb", 0.0))

    @property
    def asr_model_path(self) -> str:
        model = self.raw.get("asr", {}).get("model")
        if not model:
            raise BenchMissingError("bench.json 缺少 asr.model，請重跑 scripts/bench_vram.py")
        p = Path(model)
        if not p.is_absolute() and (MODELS_DIR / model).exists():
            return str(MODELS_DIR / model)
        return str(p)

    @property
    def asr_compute_type(self) -> str:
        return self.raw.get("asr", {}).get("compute_type", "int8_float16")

    @property
    def asr_device(self) -> str:
        return self.raw.get("asr", {}).get("device", "cuda")

    @property
    def inclass_model(self) -> ModelChoice | None:
        d = self.raw.get("summary_model_inclass")
        return ModelChoice.from_dict(d) if d and d.get("model") else None

    @property
    def final_model(self) -> ModelChoice | None:
        d = self.raw.get("summary_model_final")
        return ModelChoice.from_dict(d) if d and d.get("model") else None


_bench_cache: Bench | None = None


_BENCH_HINT = (
    "服務啟動前必須先量測硬體能力（規格 §0.1、§14.1）：\n"
    "    python scripts/bench_vram.py\n"
    "    python scripts/verify_m0.py"
)


def load_bench(required: bool = True) -> Bench:
    """讀取 data/bench.json。

    required=True 時，檔案不存在或缺少 asr.model 都拒絕啟動——
    驗證腳本會把各自的量測結果併進同一個檔，所以「檔案存在」
    不代表選型已定案。
    """
    global _bench_cache
    if _bench_cache is not None:
        return _bench_cache
    if not BENCH_PATH.exists():
        if required:
            raise BenchMissingError("找不到 %s。\n%s" % (BENCH_PATH, _BENCH_HINT))
        _bench_cache = Bench(raw={})
        return _bench_cache
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        bench = Bench(raw=json.load(f))
    if required and not (bench.raw.get("asr") or {}).get("model"):
        raise BenchMissingError(
            "%s 缺少 asr.model —— 摘要模型選型尚未定案。\n%s"
            % (BENCH_PATH, _BENCH_HINT))
    _bench_cache = bench
    return _bench_cache


def reset_bench_cache() -> None:
    global _bench_cache
    _bench_cache = None
