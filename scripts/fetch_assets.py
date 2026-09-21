"""下載外部資產：ASR 模型、llama.cpp 執行檔、摘要模型 GGUF。

規格書沒有這支（§3.3 只給了轉換指令），但把下載步驟腳本化才能重跑、續傳，
也讓 verify_m0 的修復建議有一個明確的執行入口。

用法：
    python scripts/fetch_assets.py              # 全部
    python scripts/fetch_assets.py asr llama    # 只抓指定項目
    python scripts/fetch_assets.py gguf --only 4b
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from _common import ROOT, header

MODELS = ROOT / "models"
TOOLS = ROOT / "tools"

# 規格 §3.3：先找社群已轉換好的 CT2 版本，不可用才自行轉換
ASR_REPO = "phate334/Breeze-ASR-25-int8-CT2"
ASR_DIR = MODELS / "breeze-asr-25-ct2"

# llama.cpp 預編譯版。CUDA 12.4 對應本機驅動的 CUDA 12.7（13.x 需要更新的驅動）
LLAMA_BUILD = "b10867"
LLAMA_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/%s/" % LLAMA_BUILD
LLAMA_ZIPS = [
    "llama-%s-bin-win-cuda-12.4-x64.zip" % LLAMA_BUILD,
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]

# 依 bench_vram.CANDIDATES 的優先序；先抓最可能被選中的
GGUFS = [
    ("4b",   "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("8b",   "unsloth/Qwen3-8B-GGUF",               "Qwen3-8B-Q4_K_M.gguf"),
    ("1.7b", "unsloth/Qwen3-1.7B-GGUF",             "Qwen3-1.7B-Q4_K_M.gguf"),
    ("4b0",  "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_0.gguf"),
]


def _mb(p):
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def fetch_asr() -> bool:
    header("ASR 模型：%s（規格 §3.3）" % ASR_REPO)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(
        repo_id=ASR_REPO,
        local_dir=str(ASR_DIR),
        allow_patterns=["*.json", "*.bin", "*.txt", "*.model"],
        max_workers=4,
    )
    shutil.rmtree(ASR_DIR / ".cache", ignore_errors=True)
    ok = (ASR_DIR / "model.bin").exists()
    print("  %s  model.bin %.0f MB  (%.0fs)"
          % ("OK" if ok else "FAIL", _mb(ASR_DIR / "model.bin"), time.time() - t0))
    return ok


def _download(url: str, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 5:
                last = now
                pct = (" %.0f%%" % (100 * done / total)) if total else ""
                print("    %.0f MB%s" % (done / 1e6, pct), flush=True)
    tmp.replace(out)
    print("  OK %s  %.0f MB  (%.0fs)" % (out.name, _mb(out), time.time() - t0))


def fetch_llama() -> bool:
    header("llama.cpp %s（CUDA 12.4 x64）" % LLAMA_BUILD)
    TOOLS.mkdir(exist_ok=True)
    dest = TOOLS / "llama.cpp"
    for name in LLAMA_ZIPS:
        zp = TOOLS / name
        if not zp.exists() or _mb(zp) < 1:
            _download(LLAMA_BASE + name, zp)
        else:
            print("  已存在 %s（%.0f MB）" % (name, _mb(zp)))
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
    exe = dest / "llama-server.exe"
    if not exe.exists():
        hits = list(dest.rglob("llama-server.exe"))
        if hits:
            exe = hits[0]
    print("  llama-server：%s" % (exe if exe.exists() else "找不到（解壓結構有變？）"))
    return exe.exists()


def fetch_gguf(only=None) -> bool:
    header("摘要模型 GGUF（規格 §7.2 候選）")
    from huggingface_hub import hf_hub_download
    MODELS.mkdir(exist_ok=True)
    ok = True
    for tag, repo, fname in GGUFS:
        if only and tag not in only:
            continue
        target = MODELS / fname
        if target.exists() and _mb(target) > 100:
            print("  已存在 %s（%.0f MB）" % (fname, _mb(target)))
            continue
        print("  下載 %s ← %s" % (fname, repo), flush=True)
        t0 = time.time()
        try:
            p = hf_hub_download(repo_id=repo, filename=fname,
                                local_dir=str(MODELS))
        except Exception as e:
            print("  FAIL %s：%s" % (fname, e))
            ok = False
            continue
        print("  OK %s  %.0f MB  (%.0fs)" % (fname, _mb(Path(p)), time.time() - t0))
    shutil.rmtree(MODELS / ".cache", ignore_errors=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=None,
                    choices=["asr", "llama", "gguf"],
                    help="不指定就全部抓")
    ap.add_argument("--only", default="",
                    help="gguf 專用，逗號分隔：4b,8b,1.7b,4b0")
    args = ap.parse_args()
    what = args.what or ["asr", "llama", "gguf"]
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None

    results = {}
    if "asr" in what:
        results["asr"] = fetch_asr()
    if "llama" in what:
        results["llama"] = fetch_llama()
    if "gguf" in what:
        results["gguf"] = fetch_gguf(only)

    header("下載結果")
    for k, v in results.items():
        print("  %-6s %s" % (k, "OK" if v else "FAIL"))
    if results.get("llama"):
        print("\n提醒：把 llama-server 加進 PATH，或設定環境變數")
        print("  set LS_LLAMA_SERVER_BIN=%s"
              % (TOOLS / "llama.cpp" / "llama-server.exe"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
