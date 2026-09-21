"""把 CUDA / cuDNN 的 DLL 目錄掛進 Windows 的 DLL 搜尋路徑。

規格 §2 要求 cuDNN 9（faster-whisper 依賴），這是最常見的失敗點。
在這台機器上 cuDNN 9 已隨 `torch` 的 wheel 附在 `torch/lib` 下，
但 Python 3.8+ 的 Windows 不再吃 PATH 來找 extension module 的相依 DLL，
所以 CTranslate2 看不到它們——必須顯式 `os.add_dll_directory`。

匯入 `server` 套件時自動執行一次，其他平台是 no-op。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_done = False
_added = []


def _candidate_dirs():
    """可能放著 cudnn*.dll / cublas*.dll 的目錄，依優先序。"""
    dirs = []

    env = os.getenv("LS_CUDA_DLL_DIR")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)

    for sp in sys.path:
        if not sp:
            continue
        base = Path(sp)
        # torch 的 wheel 自帶 cudnn64_9.dll 等
        dirs.append(base / "torch" / "lib")
        # nvidia-cudnn-cu12 / nvidia-cublas-cu12 等官方 pip 套件的擺法
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for pkg in sorted(nvidia.iterdir()):
                dirs.append(pkg / "bin")
                dirs.append(pkg / "lib")

    # 系統安裝的 CUDA Toolkit
    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        dirs.append(Path(cuda_path) / "bin")

    # llama.cpp 的 cudart 包也帶了一整套 CUDA runtime，
    # 沒裝 CUDA 版 torch 時可以靠它（cuDNN 仍需另外來源）
    dirs.append(Path(__file__).resolve().parent.parent / "tools" / "llama.cpp")

    return dirs


# CTranslate2 延遲載入這些庫。載入順序要由底層往上，否則相依解析會失敗。
_PRELOAD = [
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
]

_preloaded = []


def _preload(dirs):
    """以完整路徑把 CUDA 庫載進行程。

    只做 os.add_dll_directory 不夠：CTranslate2 是用裸檔名呼叫
    LoadLibrary 去延遲載入 cuBLAS/cuDNN，那條路徑不吃我們加的目錄，
    會得到 "Library cublas64_12.dll is not found or cannot be loaded"。
    但只要同名模組已經在行程裡，後續的 LoadLibrary 會直接拿到它、
    不再去檔案系統找。所以這裡先用絕對路徑載入一遍。
    """
    import ctypes
    for name in _PRELOAD:
        for d in dirs:
            p = Path(d) / name
            if not p.exists():
                continue
            try:
                ctypes.WinDLL(str(p))
                _preloaded.append(name)
            except OSError as e:
                log.debug("預載 %s 失敗：%s", p, e)
            break


def ensure(verbose: bool = False):
    """回傳實際掛上的目錄清單。重複呼叫只做一次。"""
    global _done
    if _done:
        return list(_added)
    _done = True
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []

    seen = set()
    for d in _candidate_dirs():
        try:
            resolved = d.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if not any(resolved.glob("cudnn*.dll")) and not any(resolved.glob("cublas*.dll")):
            continue
        try:
            os.add_dll_directory(str(resolved))
        except OSError as e:
            log.debug("無法掛載 DLL 目錄 %s：%s", resolved, e)
            continue
        _added.append(str(resolved))
        if verbose:
            log.info("已掛載 CUDA DLL 目錄：%s", resolved)

    if _added:
        _preload(_added)
    else:
        log.debug("找不到 cuDNN/cuBLAS 的 DLL 目錄；若 ASR 載入失敗請見 verify_m0 的修復建議")
    return list(_added)


def preloaded():
    """已成功預載的 CUDA 庫檔名，供 verify_m0 回報。"""
    return list(_preloaded)
