"""lecture-scribe 伺服端套件。

匯入時先把 CUDA / cuDNN 的 DLL 目錄掛進搜尋路徑，
否則 CTranslate2 在 Windows 上會找不到 cudnn64_9.dll（規格 §2）。
"""
from . import cuda_dlls as _cuda_dlls

_cuda_dlls.ensure()
