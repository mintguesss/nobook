"""nvidia-smi 封裝：查詢 VRAM 總量／已用量（規格 §7.1、§7.4）。

刻意不依賴 torch/pynvml —— bench 腳本要能在裝任何深度學習套件之前跑。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

_SMI_CANDIDATES = [
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
]


def find_smi():
    env = os.getenv("LS_NVIDIA_SMI")
    if env and os.path.exists(env):
        return env
    for c in _SMI_CANDIDATES:
        p = shutil.which(c) if not os.path.sep in c else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


@dataclass
class GpuInfo:
    name: str
    total_mb: float
    used_mb: float

    @property
    def free_mb(self) -> float:
        return max(0.0, self.total_mb - self.used_mb)


def query(index: int = 0):
    """回傳 GpuInfo，查不到（無 GPU 或無 nvidia-smi）回傳 None。"""
    smi = find_smi()
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        return GpuInfo(name=parts[0], total_mb=float(parts[1]), used_mb=float(parts[2]))
    except ValueError:
        return None


def used_mb(index: int = 0) -> float:
    info = query(index)
    return info.used_mb if info else 0.0


def free_mb(index: int = 0) -> float:
    info = query(index)
    return info.free_mb if info else 0.0


def available_in_budget(index: int = 0):
    """回傳 (available_mb, baseline_used_mb)。

    available = total - baseline_used；baseline 是桌面環境等既有佔用，
    筆電的顯示輸出也吃這張卡，不能忽略（規格 §7.1）。
    """
    info = query(index)
    if not info:
        return (0.0, 0.0)
    return (info.total_mb - info.used_mb, info.used_mb)
