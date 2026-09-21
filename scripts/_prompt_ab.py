"""有／無 initial_prompt 的 A/B 對照，走正式 VAD 管線。

用法：python scripts/_prompt_ab.py <wav> <course_id> [分鐘數]

必須走 VAD 切段：faster-whisper 在 condition_on_previous_text=False 時，
處理完第一個 30 秒窗口就丟掉 initial_prompt，整段直接丟會把效果稀釋掉。
"""
import io, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf
from server import courses
from server.asr import ASREngine
from server.audio_pipeline import VadSegmenter, default_vad
from server.config import load_bench

wav, cid = sys.argv[1], sys.argv[2]
mins = float(sys.argv[3]) if len(sys.argv) > 3 else 0

course = courses.load_course(cid)
a, sr = sf.read(wav, dtype="float32")
if mins:
    a = a[:int(mins * 60 * sr)]
seg = VadSegmenter(default_vad())
segs = [s.audio for s in seg.feed(a)] + [s.audio for s in seg.flush()]
print("VAD 切出 %d 段，平均 %.1fs，最長 %.1fs"
      % (len(segs), sum(len(x) for x in segs) / len(segs) / sr,
         max(len(x) for x in segs) / sr), flush=True)

b = load_bench(required=False)
e = ASREngine(b.asr_model_path, compute_type=b.asr_compute_type)
e.load()


def run(prompt, tag):
    t0 = time.time()
    out = []
    for s in segs:
        r = e.transcribe_joined(s, prompt)
        if r:
            out.append(r["text"])
    txt = "".join(out)
    io.open("data/ab_%s_%s.txt" % (cid, tag), "w", encoding="utf-8").write(txt)
    print("  %s 完成 %.0fs" % (tag, time.time() - t0), flush=True)
    return txt


t_no = run("", "noprompt")
t_yes = run(course.asr_prompt, "withprompt")

cn = {g: t_no.lower().count(g.lower()) for g in course.glossary}
cy = {g: t_yes.lower().count(g.lower()) for g in course.glossary}
print("\n%-24s %8s %8s %7s" % ("術語", "無prompt", "有prompt", "變化"))
for g in course.glossary:
    d = cy[g] - cn[g]
    print("%-24s %8d %8d %+7d%s" % (g, cn[g], cy[g], d, "  <<" if d else ""))
A, B = sum(cn.values()), sum(cy.values())
print("\n合計 %d -> %d  (%+.1f%%)   規格門檻 +10%%" % (A, B, 100.0 * (B - A) / A if A else 0))
