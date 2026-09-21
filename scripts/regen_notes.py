"""用目前的設定重新產生既有課堂的筆記。

改了 prompt、術語表或模型之後，舊的紀錄仍是用舊設定產生的。
這支腳本從**逐字稿**重新跑一次完整流程，不是拿舊的段落摘要再加工——
舊摘要已經被寫死的 6 則上限壓過一次，從那裡開始等於繼承了資訊損失。

會做的事：
  1. 用課程術語表修正逐字稿的近似音錯誤（term_fix）
  2. 依原本的段落邊界重新產生段落摘要（要點數隨長度調整）
  3. 重新產生手抄版與完整版
  4. 寫回資料庫

用法：
    python scripts/regen_notes.py                 # 全部已結束的課
    python scripts/regen_notes.py 7e11bfd0        # 指定 session（前綴即可）
    python scripts/regen_notes.py --course se-2026
    python scripts/regen_notes.py --dry-run       # 只看會改什麼，不寫入
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from _common import Gate, header

from server import config, courses, export, storage, term_fix
from server.llm_manager import LLMManager, LLMUnavailable
from server.summarizer import Summarizer


def pick_sessions(args):
    rows = storage.list_sessions(500)
    out = []
    for r in rows:
        if args.course and r["course_id"] != args.course:
            continue
        if args.ids and not any(r["id"].startswith(p) for p in args.ids):
            continue
        if not args.include_unfinished and not r["ended_at"]:
            continue
        out.append(r)
    return out


async def regen_one(summarizer, row, dry_run=False, verbose=True):
    sid = row["id"]
    course = courses.get_course(row["course_id"])
    segs = storage.list_segments(sid)
    old_secs = storage.list_sections(sid)
    if not segs:
        return {"id": sid, "skipped": "沒有逐字稿"}

    # 1. 近似音修正
    fixer = term_fix.for_course(course)
    fixed_segs = []
    all_corr = []
    for g in segs:
        t = g["text"]
        if fixer.enabled:
            t, corr = fixer.fix(t)
            all_corr.extend(corr)
        fixed_segs.append(dict(g, text=t))

    # 2. 依原邊界重新產生段落摘要。
    # 舊邊界只反映當時按過幾次「即時整理」，不保證蓋住整堂課：mmdb-2026 有一堂
    # 32,368 字的逐字稿，舊紀錄只有一則涵蓋前 219 秒的段落，照舊邊界重生會把
    # 96% 的內容丟掉。所以這裡把沒被覆蓋的空檔補成獨立的段落（user_note 留空，
    # 標記是屬於當初按下去的那一刻，不該被拉到補出來的段落上）。
    tail = fixed_segs[-1]["end_s"] + 1
    bounds = sorted(
        ((s["start_s"], s["end_s"], s.get("user_note")) for s in old_secs),
        key=lambda b: b[0])
    filled = []
    prev_end = 0.0
    for st, en, note in bounds:
        if st > prev_end + 1:
            filled.append((prev_end, st, None))
        filled.append((st, en, note))
        prev_end = max(prev_end, en)
    if tail > prev_end + 1:
        filled.append((prev_end, tail, None))
    bounds = filled or [(0.0, tail, None)]

    new_secs = []
    for i, (st, en, note) in enumerate(bounds):
        chunk = [g for g in fixed_segs if st <= g["start_s"] < en]
        if not chunk and i == len(bounds) - 1:
            chunk = [g for g in fixed_segs if g["start_s"] >= st]
        text = "".join(g["text"] for g in chunk).strip()
        if not text:
            continue
        spans = await summarizer.summarize_span(course, new_secs[-2:], text, st, en)
        for j, sp in enumerate(spans):
            new_secs.append({"start_s": sp["start_s"], "end_s": sp["end_s"],
                             "title": sp["title"], "bullets": sp["bullets"],
                             "summary": sp.get("summary", ""),
                             "groups": sp.get("groups") or [],
                             "user_note": note if j == 0 else None})
        if verbose:
            print("     第 %d 段（%d 字）→ %d 塊、共 %d 個要點"
                  % (i + 1, len(text), len(spans),
                     sum(len(x["bullets"]) for x in spans)), flush=True)

    if not new_secs:
        return {"id": sid, "skipped": "重新摘要後沒有內容"}

    # 3. 手抄版與完整版
    notes = await summarizer.summarize_handcopy(course, new_secs, model_key="inclass")
    handcopy_md = export.build_handcopy(course, row, notes)
    final = await summarizer.summarize_final(course, new_secs)
    final_md = export.build_markdown(course, row, new_secs, final)

    old_bul = sum(len(s["bullets"]) for s in old_secs)
    new_bul = sum(len(s["bullets"]) for s in new_secs)
    result = {
        "id": sid, "course": course.name,
        "corrections": all_corr,
        "old_bullets": old_bul, "new_bullets": new_bul,
        "old_handcopy": len(row.get("handcopy_md") or ""),
        "new_handcopy": len(handcopy_md),
        "old_final": len(row.get("final_md") or ""),
        "new_final": len(final_md),
    }

    if not dry_run:
        # 逐字稿的修正也寫回去，之後看紀錄才是修正後的版本
        for old, new in zip(segs, fixed_segs):
            if old["text"] != new["text"]:
                storage.update_segment_text(old["id"], new["text"])
        storage.replace_sections(sid, new_secs)
        storage.finish_session(sid, row["ended_at"] or "", row["duration_s"] or 0.0,
                               final_md, handcopy_md, row.get("audio_path"))
    return result


async def main_async(args) -> int:
    header("regen_notes.py — 用目前的設定重新產生筆記")
    gate = Gate("REGEN")
    storage.init()
    rows = pick_sessions(args)
    if not rows:
        gate.check(False, "找到要重生的課堂", "沒有符合條件的 session")
        return gate.finish()
    gate.info("共 %d 堂%s" % (len(rows), "（--dry-run，不會寫入）" if args.dry_run else ""))

    bench = config.load_bench(required=False)
    llm = LLMManager(bench)
    summarizer = Summarizer(llm)
    ok = 0
    try:
        for r in rows:
            print("\n  %s  %s  %s" % (r["id"][:8], r["started_at"][5:16], r["course_id"]),
                  flush=True)
            t0 = time.time()
            try:
                res = await regen_one(summarizer, storage.get_session(r["id"]),
                                      args.dry_run)
            except LLMUnavailable as e:
                gate.check(False, "%s 重生" % r["id"][:8], str(e))
                continue
            if res.get("skipped"):
                gate.info("     略過：%s" % res["skipped"])
                continue
            ok += 1
            c = res["corrections"]
            uniq = {}
            for a, b, _ in c:
                uniq[(a, b)] = uniq.get((a, b), 0) + 1
            gate.info("     要點 %d → %d ／ 手抄版 %d → %d 字 ／ 完整版 %d → %d 字（%.0fs）"
                      % (res["old_bullets"], res["new_bullets"],
                         res["old_handcopy"], res["new_handcopy"],
                         res["old_final"], res["new_final"], time.time() - t0))
            if uniq:
                gate.info("     近似音修正：%s"
                          % "、".join("%s→%s×%d" % (a, b, n) for (a, b), n in uniq.items()))
    finally:
        await llm.shutdown()
        storage.close()

    gate.check(ok == len(rows), "全部重生成功", "%d/%d" % (ok, len(rows)))
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="session id 前綴；不給就全部")
    ap.add_argument("--course", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-unfinished", action="store_true")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
