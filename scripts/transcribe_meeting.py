"""把會議錄音檔轉成標好講者的逐字稿。

    python scripts/transcribe_meeting.py <錄音檔或資料夾>
    python scripts/transcribe_meeting.py <資料夾> --terms 術語.yaml --speakers 3

輸出放在錄音旁邊的「逐字稿」資料夾：<檔名>.txt 給人看，<檔名>.jsonl 給後續整理用。
不碰資料庫，伺服器開著也能跑，只要沒有同時在錄音。

--terms 是 YAML，三個欄位都可省略：
    asr_prompt: 接在預設提示後面的文字，寫這批會議常講的詞
    glossary: [負載卸載, 生成式]      # 讀音接近時自動改回來
    corrections: {深層式: 生成式}     # 固定錯字對照，一律取代

知道人數就給 --speakers，比自動判斷準。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from _common import header

from server import config, courses, diarize, term_fix
from server.asr import ASREngine, clean_text
from server.audio_pipeline import VadSegmenter
from server.vad import make_vad

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".webm"}
# 同一人的相鄰片段併成一行，逐字稿才不會碎成一句一行
MERGE_GAP_S = 3.0
MERGE_MAX_CHARS = 300


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def load_terms(course, path):
    """把 --terms 疊到會議的預設設定上。"""
    if not path:
        return course
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    course.asr_prompt = courses._truncate_prompt(
        (course.asr_prompt.strip() + " " + str(raw.get("asr_prompt") or "")).strip(),
        course.id)
    course.glossary = [str(g) for g in (raw.get("glossary") or [])]
    course.corrections = {str(k): str(v)
                          for k, v in (raw.get("corrections") or {}).items()}
    return course


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def transcribe_segment(engine, fixer, seg, prompt, turns, log):
    """一個 VAD 段落 → 依講者切開的列。"""
    rows = []
    for part in engine.transcribe(seg.audio, prompt, word_timestamps=True):
        words = [(seg.start_s + a, seg.start_s + b, w) for a, b, w in part["words"]]
        pieces = (diarize.split_words_by_speaker(words, turns) if turns and words
                  else [(seg.start_s + part["start"], seg.start_s + part["end"],
                         part["text"], None)])
        for start, end, text, speaker in pieces:
            text = clean_text(text, engine._cc)
            if not text:
                continue
            if fixer.enabled:
                text, fixes = fixer.fix(text)
                log.extend(fixes)
            rows.append({"start": start, "end": end, "speaker": speaker,
                         "text": text, "avg_logprob": part["avg_logprob"]})
    return rows


def merge_rows(rows):
    out = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev and prev["speaker"] == r["speaker"]
                and r["start"] - prev["end"] <= MERGE_GAP_S
                and len(prev["text"]) + len(r["text"]) <= MERGE_MAX_CHARS):
            chars = len(prev["text"]), len(r["text"])
            prev["avg_logprob"] = ((prev["avg_logprob"] * chars[0]
                                    + r["avg_logprob"] * chars[1]) / sum(chars))
            prev["text"] += r["text"]
            prev["end"] = r["end"]
        else:
            out.append(dict(r))
    return out


def write_outputs(rows, txt: Path, jsonl: Path):
    lines = []
    for r in rows:
        who = "講者 %s：" % (r["speaker"] or "?")
        lines.append("[%s] %s%s" % (fmt_ts(r["start"]), who, r["text"]))
    tmp = txt.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    tmp.replace(txt)       # .txt 最後才出現，代表這份已完整寫完


def process(path, out_dir, engine, course, vad, args):
    from faster_whisper.audio import decode_audio
    txt, jsonl = out_dir / (path.stem + ".txt"), out_dir / (path.stem + ".jsonl")
    if txt.exists() and not args.force:
        print("  略過（已存在，要重做加 --force）：%s" % path.name, flush=True)
        return
    t0 = time.time()
    audio = decode_audio(str(path), sampling_rate=config.SAMPLE_RATE)
    minutes = len(audio) / config.SAMPLE_RATE / 60
    print("  %s（%.1f 分鐘）" % (path.name, minutes), flush=True)

    turns = []
    if not args.no_speakers:
        turns = diarize.diarize(audio, vad, num_speakers=args.speakers)
        print("    講者 %d 人（%.0fs）" % (len({t[2] for t in turns}), time.time() - t0),
              flush=True)

    segmenter = VadSegmenter(vad)
    segmenter.reset()
    segments = []
    for i in range(0, len(audio), config.SAMPLE_RATE):
        segments += segmenter.feed(audio[i:i + config.SAMPLE_RATE])
    segments += segmenter.flush()

    fixer = term_fix.TermFixer(course.glossary, corrections=course.corrections)
    fixes, rows = [], []
    for seg in segments:
        rows += transcribe_segment(engine, fixer, seg, course.asr_prompt, turns, fixes)
    rows = merge_rows(rows)
    write_outputs(rows, txt, jsonl)
    print("    %d 行、%d 字、修正術語 %d 處，共 %.0fs"
          % (len(rows), sum(len(r["text"]) for r in rows), len(fixes), time.time() - t0),
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="錄音檔或資料夾")
    ap.add_argument("--terms", help="這批會議的術語 YAML")
    ap.add_argument("--speakers", type=int, default=0, help="與會人數，不給就自動判斷")
    ap.add_argument("--no-speakers", action="store_true", help="不標講者")
    ap.add_argument("--force", action="store_true", help="已轉過的也重做")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    header("transcribe_meeting.py — 會議錄音轉逐字稿")
    if not args.no_speakers and not diarize.available():
        print("講者辨識不可用：先跑 python scripts/fetch_assets.py diarize，"
              "或加 --no-speakers")
        return 1
    src = Path(args.path)
    audio_files = find_audio(src)
    if not audio_files:
        print("找不到錄音檔：%s" % src)
        return 1
    out_dir = (src.parent if src.is_file() else src) / "逐字稿"
    out_dir.mkdir(parents=True, exist_ok=True)

    course = load_terms(courses.load_course("meeting"), args.terms)
    bench = config.load_bench(required=True)
    engine = ASREngine(bench.asr_model_path, device=bench.asr_device,
                       compute_type=bench.asr_compute_type)
    engine.load()
    vad = make_vad()
    for path in audio_files:
        process(path, out_dir, engine, course, vad, args)
    print("輸出：%s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
