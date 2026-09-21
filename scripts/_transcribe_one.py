"""把一個 wav 全文轉錄出來，供建立課程術語表用。

用法：python scripts/_transcribe_one.py <wav> <輸出txt> [--prompt "..."]
"""
import io, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf
from server.asr import ASREngine
from server.config import load_bench

wav, out = sys.argv[1], sys.argv[2]
prompt = ""
if "--prompt" in sys.argv:
    prompt = sys.argv[sys.argv.index("--prompt") + 1]

b = load_bench(required=False)
a, sr = sf.read(wav, dtype="float32")
e = ASREngine(b.asr_model_path, compute_type=b.asr_compute_type)
e.load()
parts, t0, CH = [], time.time(), 120 * sr
for i in range(0, len(a), CH):
    r = e.transcribe_joined(a[i:i + CH], prompt)
    parts.append("[%02d:%02d] %s" % (i // sr // 60, i // sr % 60, (r or {}).get("text", "")))
    print("  %d/%d 分鐘" % (min((i + CH) // sr // 60, len(a) // sr // 60),
                            len(a) // sr // 60), flush=True)
io.open(out, "w", encoding="utf-8").write("\n\n".join(parts))
print("完成 %.0fs  RTF %.3f  →  %s" % (time.time() - t0, (time.time() - t0) / (len(a) / sr), out))
