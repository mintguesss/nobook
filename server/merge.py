"""把同一天同一門課的幾段錄音合併成一堂，重新產生總筆記。

為什麼需要：一堂三小時的課常常被拆成好幾段錄（中途休息、手機睡著、
換教室重連）。分開看的話每一段各有一份筆記，但老師的論述是跨段的，
拆開之後的筆記接不起來，複習時也要在四五筆紀錄之間跳來跳去。

合併會產生一筆**新的**紀錄，原本那幾筆留著不動——使用者自己決定要不要
刪。合併失敗時原始資料也不會受影響。
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from . import audio_store, config, courses, export, storage, term_fix

log = logging.getLogger(__name__)

# 兩段錄音之間補一段靜音，聽的時候才知道換段了，時間軸也對得起來
GAP_S = 2.0


class MergeError(Exception):
    pass


def _same_day(rows) -> bool:
    return len({(r["started_at"] or "")[:10] for r in rows}) == 1


def pick_rows(ids):
    rows = []
    for sid in ids:
        r = storage.get_session(sid)
        if r is None:
            raise MergeError("找不到 session：%s" % sid[:8])
        rows.append(r)
    if len(rows) < 2:
        raise MergeError("至少要選兩堂才需要合併")
    if len({r["course_id"] for r in rows}) != 1:
        raise MergeError("只能合併同一門課的紀錄")
    # 還沒結束的課不能合併：錄音檔還開著，讀進來是半個檔；時長也還不知道，
    # 排不出時間軸。先在那堂按「結束並產生筆記」再回來合併。
    open_rows = [r for r in rows if not r.get("ended_at")]
    if open_rows:
        raise MergeError(
            "有 %d 堂還沒結束（%s），請先把它結束再合併"
            % (len(open_rows), "、".join(r["id"][:8] for r in open_rows)))
    rows.sort(key=lambda r: r["started_at"])
    return rows


def concat_audio(rows, out_id: str, course=None):
    """把幾段 FLAC 接成一個檔。缺檔就跳過那一段，不讓整個合併失敗。

    回傳 (輸出路徑或 None, 每一段的起始秒數, 總長度秒數)。
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        log.warning("沒有 soundfile，略過錄音合併")
        return None, None, 0.0

    paths = [(r, r.get("audio_path")) for r in rows]
    have = [(r, p) for r, p in paths if p and Path(p).exists()]
    if not have:
        return None, None, 0.0
    if len(have) < len(paths):
        log.warning("有 %d 段沒有錄音檔，合併後的音訊會少那幾段",
                    len(paths) - len(have))

    out_dir = Path(config.AUDIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 合併檔的起始時間跟來源第一段一樣，不加後綴就會撞名
    out_path = audio_store.build_filename(course, rows[0]["started_at"],
                                          out_id, out_dir, suffix="合併")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(GAP_S * config.SAMPLE_RATE), dtype="int16")
    offsets = {}
    written = 0
    with sf.SoundFile(str(out_path), mode="w",
                      samplerate=config.SAMPLE_RATE, channels=1,
                      format="FLAC", subtype="PCM_16") as out:
        for i, (r, p) in enumerate(have):
            offsets[r["id"]] = written / config.SAMPLE_RATE
            if i:
                out.write(gap)
                written += len(gap)
            with sf.SoundFile(str(p)) as src:
                if src.samplerate != config.SAMPLE_RATE:
                    raise MergeError(
                        "取樣率不一致（%s 是 %d Hz），無法合併"
                        % (Path(p).name, src.samplerate))
                while True:
                    block = src.read(1 << 16, dtype="int16")
                    if not len(block):
                        break
                    out.write(block)
                    written += len(block)
    return str(out_path), offsets, written / config.SAMPLE_RATE


def _span_of(r) -> float:
    """這堂課的長度。duration_s 可能是 0 或缺，就用逐字稿的最後一句。"""
    d = r.get("duration_s") or 0.0
    segs = storage.list_segments(r["id"])
    return max(d, segs[-1]["end_s"] if segs else 0.0)


def _offsets_from_duration(rows):
    """沒有錄音檔時，用每段的長度排時間軸。"""
    offsets, cur = {}, 0.0
    for r in rows:
        offsets[r["id"]] = cur
        cur += _span_of(r) + GAP_S
    return offsets, cur


async def merge_sessions(ids, summarizer) -> dict:
    """合併並重新產生筆記，回傳新紀錄的摘要資訊。"""
    rows = pick_rows(ids)
    if not _same_day(rows):
        log.warning("要合併的紀錄不是同一天，仍照時間順序接起來")
    course = courses.get_course(rows[0]["course_id"])
    new_id = str(uuid.uuid4())

    audio_path, offsets, written_s = concat_audio(rows, new_id, course)
    if offsets is None:
        offsets, _ = _offsets_from_duration(rows)
    else:
        # 沒有錄音檔的那幾段要接在**已寫入音訊的結尾**之後。
        # 接在 max(offsets) 之後是錯的——那是最後一段的「起點」，
        # 會讓沒錄音的那段跟前一段的時間軸整個重疊。
        cur = written_s
        for r in rows:
            if r["id"] not in offsets:
                offsets[r["id"]] = cur + GAP_S
                cur += GAP_S + _span_of(r)

    # 1. 逐字稿搬過去，時間往後平移
    storage.create_session(new_id, course.id, rows[0]["started_at"])
    total = 0.0
    n_seg = 0
    fixer = term_fix.for_course(course)
    merged_text = []
    for r in rows:
        off = offsets.get(r["id"], 0.0)
        for g in storage.list_segments(r["id"]):
            text = g["text"]
            if fixer.enabled:
                text, _ = fixer.fix(text)
            storage.insert_segment(new_id, g["start_s"] + off,
                                   g["end_s"] + off, text, g.get("avg_logprob"))
            merged_text.append(text)
            total = max(total, g["end_s"] + off)
            n_seg += 1
    if not n_seg:
        storage.delete_session(new_id)
        raise MergeError("這幾堂都沒有逐字稿，沒有東西可以合併")

    # 2. 重新產生段落摘要。不沿用原本的段落：原本的邊界是按鈕按下去的
    #    時間點，跨段之後那些邊界沒有意義了。
    text = "".join(merged_text).strip()
    spans = await summarizer.summarize_span(course, [], text, 0.0, total)
    sections = [{"start_s": sp["start_s"], "end_s": sp["end_s"],
                 "title": sp["title"], "bullets": sp["bullets"],
                 "summary": sp.get("summary", ""),
                 "groups": sp.get("groups") or [],
                 "user_note": None} for sp in spans]
    storage.replace_sections(new_id, sections)

    # 3. 總筆記
    row = storage.get_session(new_id)
    notes = await summarizer.summarize_handcopy(course, sections,
                                                model_key="final")
    handcopy_md = export.build_handcopy(course, row, notes)
    final = await summarizer.summarize_final(course, sections)
    final_md = export.build_markdown(course, row, sections, final)
    storage.finish_session(new_id, rows[-1]["ended_at"] or rows[-1]["started_at"],
                           total, final_md, handcopy_md, audio_path)
    storage.set_merged_from(new_id, [r["id"] for r in rows])

    return {"id": new_id, "course_id": course.id, "merged_from": [r["id"] for r in rows],
            "segments": n_seg, "chars": len(text), "sections": len(sections),
            "duration_s": round(total, 1), "audio": bool(audio_path),
            "handcopy_chars": len(handcopy_md), "final_chars": len(final_md)}


def mergeable_groups(limit: int = 200):
    """找出可以合併的組：同一天、同一門課、兩筆以上。

    合併結果跟它的來源同一天、同一門課，所以要把已經合併過的排掉，
    否則合併完那一組還留在清單上，再按一次會把合併結果再合併進去。
    """
    groups = {}
    done = set()      # 已經被合併掉的來源組合
    for r in storage.list_sessions(limit):
        if r.get("merged_from"):
            try:
                done.add(frozenset(json.loads(r["merged_from"])))
            except (TypeError, ValueError):
                pass
            continue      # 合併結果本身不是合併素材
        key = (r["course_id"], (r["started_at"] or "")[:10])
        groups.setdefault(key, []).append(r)
    out = []
    for (cid, day), rows in groups.items():
        if len(rows) < 2:
            continue
        if frozenset(r["id"] for r in rows) in done:
            continue
        rows.sort(key=lambda r: r["started_at"])
        out.append({"course_id": cid, "date": day,
                    "ids": [r["id"] for r in rows],
                    "count": len(rows),
                    "total_s": round(sum(r.get("duration_s") or 0 for r in rows), 1)})
    out.sort(key=lambda g: g["date"], reverse=True)
    return out
