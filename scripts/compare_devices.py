"""比較同一段課堂內容在不同收音裝置下的 ASR 表現。

規格 §12 把收音品質列為「對準確度影響最大的單一變因，超過任何模型選擇」，
但那是規格作者的判斷，沒有實測。這支腳本用兩份重疊的錄音（手機 vs 平板）
做受控對照：內容相同、時間對齊、同一個模型與 prompt，只差收音裝置。

前置：tests/fixtures/ 下要有 phone_clip.wav 與 tablet_clip.wav
（由 align 步驟切出的同一時間窗）。

用法：
    python scripts/compare_devices.py
"""
from __future__ import annotations

import io
import sys

import numpy as np
import soundfile as sf

from _common import ROOT, Gate, header

from server import courses
from server.asr import ASREngine
from server.audio_pipeline import VadSegmenter, default_vad

FIX = ROOT / "tests" / "fixtures"


def vad_segments(audio):
    seg = VadSegmenter(default_vad())
    return [s.audio for s in seg.feed(audio)] + [s.audio for s in seg.flush()]


def transcribe(engine, segs, prompt):
    out = []
    for s in segs:
        r = engine.transcribe_joined(s, prompt)
        if r:
            out.append(r["text"])
    return "".join(out)


def main() -> int:
    header("compare_devices.py — 收音裝置對 ASR 準確度的影響（規格 §12）")
    gate = Gate("DEVICE")

    course = courses.get_course("scm-2026")
    clips = {}
    for tag, name in (("手機", "phone_clip.wav"), ("平板", "tablet_clip.wav")):
        p = FIX / name
        if not p.exists():
            gate.check(False, "找到 %s 的音檔" % tag, "缺少 %s" % p)
            return gate.finish()
        clips[tag], _ = sf.read(str(p), dtype="float32")

    from server.config import load_bench
    bench = load_bench(required=False)
    engine = ASREngine(bench.asr_model_path, compute_type=bench.asr_compute_type)
    engine.load()

    results = {}
    for tag, audio in clips.items():
        segs = vad_segments(audio)
        txt = transcribe(engine, segs, course.asr_prompt)
        low = txt.lower()
        hits = {g: low.count(g.lower()) for g in course.glossary}
        results[tag] = {"text": txt, "segs": len(segs), "hits": sum(hits.values()),
                        "per_term": hits, "chars": len(txt)}
        io.open(str(ROOT / "data" / ("device_%s.txt" % tag)), "w",
                encoding="utf-8").write(txt)
        gate.info("%s：VAD %d 段、逐字稿 %d 字、術語命中 %d 次"
                  % (tag, len(segs), len(txt), sum(hits.values())))

    a, b = results["手機"], results["平板"]
    gate.info("")
    gate.info("逐字稿長度  手機 %d 字 / 平板 %d 字（%+.0f%%）"
              % (a["chars"], b["chars"],
                 100.0 * (b["chars"] - a["chars"]) / max(1, a["chars"])))
    gate.info("術語命中    手機 %d 次 / 平板 %d 次" % (a["hits"], b["hits"]))
    diff = [(g, a["per_term"][g], b["per_term"][g]) for g in course.glossary
            if a["per_term"][g] != b["per_term"][g]]
    if diff:
        gate.info("有差異的術語：")
        for g, x, y in diff:
            gate.info("    %-14s 手機 %d / 平板 %d" % (g, x, y))

    # 這支腳本的用途是提供比較資料，不是判定通過與否——
    # 「哪個裝置比較好」沒有可以事先寫死的門檻。
    gate.check(True, "兩個裝置的逐字稿都已產出，可供人工比對",
               "結果在 data/device_手機.txt 與 data/device_平板.txt")
    engine.unload()
    print("\n請人工比對兩份逐字稿：ASR 的字錯率無法自動判定，"
          "\n但術語命中數與逐字稿長度可以當作參考指標。")
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
