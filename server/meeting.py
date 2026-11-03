"""把一整段錄音轉成標好講者的逐字稿列。網頁上傳與命令列工具共用。"""
from __future__ import annotations

import dataclasses
import re

from . import config, courses, diarize
from .asr import clean_text
from .audio_pipeline import VadSegmenter

# 同一人的相鄰片段併成一行，逐字稿才不會碎成一句一行
MERGE_GAP_S = 3.0
MERGE_MAX_CHARS = 300

_TERM_SPLIT_RE = re.compile(r"[,，、;；\n]+")


def parse_terms(text: str):
    """使用者輸入的術語 → (glossary, corrections)。

    用頓號或逗號分開；寫成「聽錯的=正確的」就是固定錯字對照。
    """
    glossary, corrections = [], {}
    for item in _TERM_SPLIT_RE.split(text or ""):
        item = item.strip()
        if not item:
            continue
        wrong, sep, right = item.partition("=")
        if sep and wrong.strip() and right.strip():
            corrections[wrong.strip()] = right.strip()
            glossary.append(right.strip())
        elif not sep:
            glossary.append(item)
    return glossary, corrections


def with_terms(course, glossary, corrections, extra_prompt=""):
    """把這批錄音的術語疊到課程設定上，回傳新的設定。"""
    words = "、".join(dict.fromkeys(glossary))
    parts = [course.asr_prompt.strip(), extra_prompt.strip(),
             ("常見詞：" + words + "。") if words else ""]
    return dataclasses.replace(
        course,
        asr_prompt=courses._truncate_prompt(" ".join(p for p in parts if p), course.id),
        glossary=list(dict.fromkeys(glossary)),
        corrections=dict(corrections),
    )


def split_segments(audio, vad):
    """整段音訊 → VAD 切好的段落。"""
    segmenter = VadSegmenter(vad)
    segmenter.reset()
    segments = []
    for i in range(0, len(audio), config.SAMPLE_RATE):
        segments += segmenter.feed(audio[i:i + config.SAMPLE_RATE])
    segments += segmenter.flush()
    return segments


def transcribe_segment(engine, fixer, seg, prompt, turns, log):
    """一個 VAD 段落 → 依講者切開的列。"""
    rows = []
    for part in engine.transcribe(seg.audio, prompt, word_timestamps=bool(turns)):
        words = [(seg.start_s + a, seg.start_s + b, w)
                 for a, b, w in part.get("words", [])]
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
