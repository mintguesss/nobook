"""閘門 G0 — 環境與模型可用性（規格 §11 M0、§14.2）。

每項獨立 try/except，失敗時印出具體修復建議。
"""
from __future__ import annotations

import ctypes
import os
import sys

from _common import Gate, find_fixture, header, load_bench

from server import config, gpu


def check_gpu(gate: Gate) -> None:
    info = gpu.query()
    if info is None:
        gate.check(False, "nvidia-smi 回報 GPU 名稱與 VRAM 總量",
                   "找不到 nvidia-smi。修復：安裝 NVIDIA 驅動，"
                   "或設環境變數 LS_NVIDIA_SMI 指向 nvidia-smi.exe")
        return
    gate.check(info.total_mb > 0, "nvidia-smi 回報 GPU 名稱與 VRAM 總量",
               "%s，total %.0fMB，目前已用 %.0fMB"
               % (info.name, info.total_mb, info.used_mb))


def check_cuda(gate: Gate) -> None:
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        gate.check(n > 0, "CTranslate2 偵測到 CUDA 裝置",
                   "device_count=%d%s" % (n, "" if n else
                   "；修復：安裝 CUDA 12.x runtime，並確認 faster-whisper 版本相容"))
    except ImportError:
        gate.check(False, "CTranslate2 偵測到 CUDA 裝置",
                   "未安裝 ctranslate2。修復：pip install -r requirements.txt")
    except Exception as e:
        gate.check(False, "CTranslate2 偵測到 CUDA 裝置", str(e))


def check_cudnn(gate: Gate) -> None:
    """faster-whisper 常見的失敗點，單獨驗（規格 §11 G0）。"""
    fix = ("修復：下載 cuDNN 9 for CUDA 12，把 bin/ 下的 cudnn*.dll 放進 PATH"
           "（或複製到 CUDA 的 bin 目錄）。"
           "Windows 上另一個常見做法是 pip install nvidia-cudnn-cu12")
    names = (["cudnn64_9.dll", "cudnn_ops64_9.dll"] if os.name == "nt"
             else ["libcudnn.so.9", "libcudnn_ops.so.9"])
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    errors = []
    for n in names:
        try:
            loader(n)
            gate.check(True, "cuDNN 9 可載入", "已載入 %s" % n)
            return
        except OSError as e:
            errors.append("%s: %s" % (n, e))
    # 退而求其次：問問 pip 套件在不在
    try:
        import nvidia.cudnn  # noqa: F401
        gate.check(True, "cuDNN 9 可載入",
                   "系統路徑找不到 dll，但已安裝 nvidia-cudnn-cu12 套件")
        return
    except ImportError:
        pass
    gate.check(False, "cuDNN 9 可載入", "; ".join(errors) + "。" + fix)


def check_asr(gate: Gate, bench: dict) -> str:
    model_path = (os.getenv("LS_ASR_MODEL")
                  or (bench.get("asr") or {}).get("model"))
    if not model_path:
        for name in ("breeze-asr-25-ct2", "Breeze-ASR-25-int8-CT2"):
            p = config.MODELS_DIR / name
            if p.exists():
                model_path = str(p)
                break
    if not model_path or not os.path.exists(model_path):
        gate.check(False, "ASR 模型能載入",
                   "找不到 CT2 格式的 Breeze-ASR-25。規格 §3.3：先找社群轉換版"
                   "（例如 phate334/Breeze-ASR-25-int8-CT2），"
                   "或自行 ct2-transformers-converter 轉換到 models/breeze-asr-25-ct2")
        return ""

    fixture = find_fixture("m0_sample.wav", "bench.wav", "lecture_3h.wav")
    if fixture is None:
        gate.skip("ASR 模型對測試 wav 產出非空文字",
                  "缺少 tests/fixtures/m0_sample.wav（任何一段中文語音，16kHz 單聲道即可）")
        gate.skip("ASR 輸出為繁體（無簡體字）", "同上，缺少測試音檔")
        return model_path

    try:
        import soundfile as sf
        from server.asr import ASREngine
        data, sr = sf.read(str(fixture), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import numpy as np
            idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
            data = np.interp(idx, np.arange(len(data)), data).astype("float32")
        data = data[:16000 * 60]
        engine = ASREngine(model_path,
                           compute_type=(bench.get("asr") or {}).get(
                               "compute_type", "int8_float16"))
        res = engine.transcribe_joined(data, "")
        text = (res or {}).get("text", "")
    except Exception as e:
        gate.check(False, "ASR 模型能載入", "%s：%s" % (model_path, e))
        gate.skip("ASR 輸出為繁體（無簡體字）", "ASR 載入失敗")
        return model_path

    gate.check(bool(text), "ASR 模型對測試 wav 產出非空文字",
               "%s → %r" % (fixture.name, text[:80]))
    check_traditional(gate, text)
    try:
        engine.unload()
    except Exception:
        pass
    return model_path


def check_traditional(gate: Gate, text: str) -> None:
    """s2twp 後應與原文相同——不同就代表原文含簡體字。"""
    if not text:
        gate.skip("ASR 輸出為繁體（無簡體字）", "沒有文字可檢查")
        return
    try:
        from opencc import OpenCC
    except ImportError:
        gate.check(False, "ASR 輸出為繁體（無簡體字）",
                   "未安裝 opencc。修復：pip install opencc-python-reimplemented")
        return
    converted = OpenCC("s2twp").convert(text)
    diffs = [(a, b) for a, b in zip(text, converted) if a != b]
    gate.check(not diffs, "ASR 輸出為繁體（無簡體字）",
               "全篇繁體" if not diffs
               else "發現 %d 處簡體：%s" % (len(diffs), diffs[:8]))


def check_bench(gate: Gate, bench: dict) -> None:
    if not bench:
        gate.check(False, "bench.json 已產生，摘要模型選型已定案",
                   "請先執行 python scripts/bench_vram.py")
        return
    inclass = bench.get("summary_model_inclass") or {}
    final = bench.get("summary_model_final") or {}
    ok = bool(inclass.get("model")) and bool(final.get("model"))
    detail = "課中：%s / 總結：%s" % (inclass.get("model", "（未定）"),
                                     final.get("model", "（未定）"))
    if not ok:
        detail += "。請先執行 python scripts/bench_vram.py"
    gate.check(ok, "bench.json 已產生，摘要模型選型已定案", detail)

    dev = bench.get("deviations") or {}
    if dev:
        gate.info("實測與規格書估計偏離超過 30%% 的項目（以實測為準）：%s"
                  % ", ".join(dev.keys()))


def check_llama_server(gate: Gate, bench: dict) -> None:
    import shutil
    binpath = config.LLAMA_SERVER_BIN
    found = binpath if os.path.isfile(binpath) else shutil.which(binpath)
    gate.check(bool(found), "llama-server 執行檔可用",
               found or ("找不到 %s。修復：python scripts/fetch_assets.py llama，"
                         "或自行安裝 llama.cpp 後設 LS_LLAMA_SERVER_BIN 指向它"
                         % binpath))
    inclass = (bench.get("summary_model_inclass") or {}).get("model")
    if inclass:
        p = config.MODELS_DIR / inclass
        gate.check(p.exists() or os.path.exists(inclass), "課中摘要模型 GGUF 檔存在",
                   str(p))


def main() -> int:
    header("verify_m0.py — 閘門 G0：環境與模型可用性（規格 §11 M0）")
    gate = Gate("G0")
    bench = load_bench()
    check_gpu(gate)
    check_cuda(gate)
    check_cudnn(gate)
    check_asr(gate, bench)
    check_llama_server(gate, bench)
    check_bench(gate, bench)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
