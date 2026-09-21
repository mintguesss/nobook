"""faster-whisper 封裝：模型載入／卸載、推論、後處理（規格 §4.3、§4.4、§7.2）。"""
from __future__ import annotations

import gc
import logging
import re
import threading
import time

import numpy as np

from . import config, gpu

log = logging.getLogger(__name__)

# 規格 §4.4 (2)：Whisper 常見幻覺尾巴黑名單
HALLUCINATION_PATTERNS = [
    r"請不吝(點贊|按讚|點贊訂閱)",
    r"訂閱\s*[+＋]?\s*(點贊|按讚|分享)",
    r"字幕(由|志愿者|志願者|提供|製作)",
    r"(明鏡|點點欄目|點點栏目)",
    r"MING\s*PAO",
    r"謝謝(大家的)?(收看|觀看|觀賞)",
    r"下集(再見|見)",
    r"(转载|轉載)(自|请注明|請註明)",
    r"^\s*(嗯|呃|啊|喔|欸)+\s*$",
    r"^\s*[。，、,\.\?!？！\-—…\s]*$",
    r"Amara\.org",
    r"由.{0,8}字幕組",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]

MIN_TEXT_LEN = 2  # 規格 §4.4 (3)


class OpenCCConverter:
    """s2twp 繁簡轉換；opencc 缺失時退化為 no-op 並記警告。"""

    def __init__(self, cfg: str = "s2twp"):
        self._conv = None
        try:
            from opencc import OpenCC
            self._conv = OpenCC(cfg)
        except Exception as e:  # pragma: no cover
            log.warning("OpenCC 不可用（%s），繁簡轉換停用", e)

    def __call__(self, text: str) -> str:
        if self._conv is None:
            return text
        try:
            return self._conv.convert(text)
        except Exception:
            return text


def clean_text(text: str, converter=None) -> str:
    """後處理：s2twp → 幻覺過濾 → 長度過濾。回傳空字串代表應丟棄。"""
    if not text:
        return ""
    t = text.strip()
    if converter is not None:
        t = converter(t)
    t = re.sub(r"\s+", " ", t).strip()
    for rx in _HALLUCINATION_RE:
        if rx.search(t):
            log.debug("丟棄疑似幻覺輸出：%r", t)
            return ""
    stripped = re.sub(r"[\s\W_]", "", t, flags=re.UNICODE)
    if len(stripped) < MIN_TEXT_LEN:
        return ""
    return t


class ASRVramLow(RuntimeError):
    """VRAM 餘量不足，該段應推入延遲佇列（規格 §7.4）。"""


class ASREngine:
    """WhisperModel 的封裝。load/unload 可重入，供課後總結時釋放 VRAM。"""

    def __init__(self, model_path: str, device: str = "cuda",
                 compute_type: str = "int8_float16", gpu_index: int = 0):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.gpu_index = gpu_index
        self._model = None
        self._lock = threading.Lock()
        self._cc = OpenCCConverter()
        self.last_rtf = 0.0

    # 生命週期 ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel
            t0 = time.time()
            self._model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_index if self.device == "cuda" else 0,
            )
            log.info("ASR 模型載入完成（%.1fs）：%s", time.time() - t0, self.model_path)
            return self._model

    def unload(self) -> None:
        """規格 §7.2：CTranslate2 沒有顯式 unload，刪實例後靠 GC 釋放。"""
        with self._lock:
            if self._model is None:
                return
            self._model = None
        gc.collect()
        gc.collect()
        time.sleep(0.5)
        log.info("ASR 模型已卸載")

    # 推論 -------------------------------------------------------------
    def _check_vram(self) -> None:
        if self.device != "cuda":
            return
        free = gpu.free_mb(self.gpu_index)
        if free and free < config.VRAM_HEADROOM_MB:
            raise ASRVramLow(
                "VRAM 餘量 %.0fMB 低於門檻 %dMB" % (free, config.VRAM_HEADROOM_MB)
            )

    def transcribe(self, audio: np.ndarray, initial_prompt: str = "",
                   language: str = "zh", beam_size: int = 5):
        """audio: float32 numpy, 16kHz 單聲道。回傳 list[dict]。

        dict 欄位：start / end（相對本段起點的秒數）、text、avg_logprob。
        """
        self._check_vram()
        model = self.load()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        dur = len(audio) / float(config.SAMPLE_RATE)
        t0 = time.time()
        with self._lock:
            segments, _info = model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=False,               # 我們自己做了 VAD，不要重複
                condition_on_previous_text=False,  # 必須：避免長音訊重複輸出迴圈
                initial_prompt=initial_prompt or None,
                temperature=0.0,
                no_speech_threshold=0.6,
            )
            raw = list(segments)
        elapsed = time.time() - t0
        self.last_rtf = elapsed / dur if dur > 0 else 0.0

        out = []
        for s in raw:
            text = clean_text(s.text, self._cc)
            if not text:
                continue
            out.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": text,
                "avg_logprob": float(getattr(s, "avg_logprob", 0.0)),
            })
        return out

    def transcribe_joined(self, audio: np.ndarray, initial_prompt: str = "",
                          language: str = "zh", beam_size: int = 5):
        """把一段音訊的所有 whisper segment 併成單一結果，或 None。"""
        parts = self.transcribe(audio, initial_prompt, language, beam_size)
        if not parts:
            return None
        text = "".join(p["text"] for p in parts).strip()
        text = clean_text(text, None)
        if not text:
            return None
        logprobs = [p["avg_logprob"] for p in parts]
        return {
            "text": text,
            "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else 0.0,
            "rtf": self.last_rtf,
        }


_engine = None  # type: ASREngine | None


def get_engine():
    """依 bench.json 建立全域 ASREngine（不會自動 load 權重）。"""
    global _engine
    if _engine is None:
        bench = config.load_bench()
        _engine = ASREngine(
            model_path=bench.asr_model_path,
            device=bench.asr_device,
            compute_type=bench.asr_compute_type,
        )
    return _engine


def set_engine(engine) -> None:
    global _engine
    _engine = engine
