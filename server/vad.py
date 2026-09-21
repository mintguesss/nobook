"""silero-vad 封裝（規格 §4.2）。跑 CPU，不佔 VRAM。

介面刻意是「吃一個 512-sample 的 float32 frame，吐 0~1 機率」，
讓 audio_pipeline 的切段邏輯可以在測試中注入假的 VAD。
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class SileroVad:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._model = None
        self._torch = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        try:
            from silero_vad import load_silero_vad
            model = load_silero_vad()
        except ImportError:
            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                trust_repo=True, onnx=False,
            )
        torch.set_num_threads(1)
        model.eval()
        self._torch = torch
        self._model = model

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def __call__(self, frame: np.ndarray) -> float:
        self._ensure()
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            return float(self._model(t, self.sample_rate).item())


class EnergyVad:
    """無 torch 環境下的備援：短時能量門檻。

    只用於離線切段除錯與單元測試，正式服務一律用 SileroVad。
    """

    def __init__(self, sample_rate: int = 16000, floor: float = 0.01):
        self.sample_rate = sample_rate
        self.floor = floor

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))) + 1e-9)
        return min(1.0, rms / self.floor)


def make_vad(prefer_silero: bool = True, sample_rate: int = 16000):
    if not prefer_silero:
        return EnergyVad(sample_rate)
    vad = SileroVad(sample_rate)
    try:
        vad._ensure()
        return vad
    except Exception as e:
        log.warning("silero-vad 載入失敗（%s），退回能量式 VAD——僅供除錯，準確度會下降", e)
        return EnergyVad(sample_rate)
