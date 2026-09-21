"""閘門 G3 — 期末總結與模型切換（規格 §11 M3、§14.5）。

用假的 15 則 section 資料直接呼叫總結流程，不需要跑滿三小時。
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import re
import sys
import time

from _common import Gate, header, load_bench, merge_bench

from server import config, courses, export, gpu
from server.llm_manager import LLMManager, LLMUnavailable
from server.summarizer import Summarizer, fmt_ts

TITLES = [
    "課程導論與評分方式", "監督式與非監督式學習的差別", "線性迴歸的最小平方解",
    "梯度下降的收斂性", "learning rate 的選擇與 scheduling",
    "momentum 與 Adam 的差別", "overfitting 的成因",
    "L1 與 L2 regularization", "cross validation 的做法",
    "混淆矩陣與召回率", "ROC 曲線與 AUC", "決策樹與資訊增益",
    "隨機森林與 bagging", "特徵工程實務", "下週小考範圍",
]


def fake_sections(n=15):
    out = []
    for i in range(n):
        start = i * 600.0
        out.append({
            "seq": i + 1, "row_id": i + 1,
            "start_s": start, "end_s": start + 580.0,
            "title": TITLES[i % len(TITLES)],
            "bullets": [
                "%s 的定義與適用情境" % TITLES[i % len(TITLES)],
                "老師舉了一個 sklearn 的實作例子說明",
                "提醒這個概念與前一節的 %s 有關" % TITLES[(i - 1) % len(TITLES)],
            ],
            "user_note": "這題會考" if i in (3, 9) else None,
        })
    return out


def check_asr_unload(gate, args) -> None:
    """規格 §14.5：nvidia-smi 在卸載前後各取一次，中間 gc.collect() + 3 秒。"""
    bench = load_bench()
    model_path = args.asr_model or (bench.get("asr") or {}).get("model")
    if not model_path:
        gate.skip("ASR 卸載後 VRAM 釋放 >= 載入時佔用的 80%",
                  "bench.json 沒有 asr.model，請先跑 bench_vram.py")
        return
    from server.asr import ASREngine
    baseline = gpu.used_mb()
    engine = ASREngine(model_path,
                       compute_type=(bench.get("asr") or {}).get(
                           "compute_type", "int8_float16"))
    try:
        engine.load()
    except Exception as e:
        gate.check(False, "ASR 卸載後 VRAM 釋放 >= 載入時佔用的 80%",
                   "模型載入失敗：%s" % e)
        return
    time.sleep(2)
    loaded = gpu.used_mb()
    occupied = loaded - baseline

    engine.unload()
    gc.collect()
    time.sleep(3)
    after = gpu.used_mb()
    freed = loaded - after
    ratio = (freed / occupied) if occupied > 0 else 0.0
    gate.check(ratio >= 0.80, "ASR 卸載後 VRAM 釋放 >= 載入時佔用的 80%",
               "載入佔 %.0fMB，卸載釋放 %.0fMB（%.0f%%）——"
               "CTranslate2 沒有顯式 unload API，這裡驗證 GC 真的有效"
               % (occupied, freed, ratio * 100))
    merge_bench({"m3": {"asr_occupied_mb": round(occupied, 1),
                        "asr_freed_mb": round(freed, 1),
                        "asr_freed_ratio": round(ratio, 3)}})


async def check_switch_and_final(gate, args) -> None:
    bench = config.load_bench(required=False)
    if bench.inclass_model is None or bench.final_model is None:
        gate.skip("llama-server 模型切換成功", "bench.json 缺少摘要模型選型")
        gate.skip("15 則 section 的總結生成 < 90 秒", "同上")
        gate.skip("匯出的 Markdown 有效且完整", "同上")
        gate.skip("切回課中模型後可正常再開一個 session", "同上")
        return

    llm = LLMManager(bench)
    summarizer = Summarizer(llm)
    sections = fake_sections(15)
    course = courses.get_course(args.course)
    md = ""
    try:
        # 1. 切到總結模型 -----------------------------------------
        try:
            await llm.ensure("final")
            switch_ok, switch_s = True, llm.last_switch_s
        except LLMUnavailable as e:
            gate.check(False, "llama-server 模型切換成功且不需重啟主服務", str(e))
            return
        gate.check(switch_ok and llm.current == "final",
                   "llama-server 模型切換成功且不需重啟主服務",
                   "切到 final（%s）耗時 %.1fs" % (bench.final_model.model, switch_s))

        # 2. 總結生成 ---------------------------------------------
        t0 = time.monotonic()
        final = await summarizer.summarize_final(course, sections)
        elapsed = time.monotonic() - t0
        gate.check(elapsed < 90, "15 則 section 的總結生成 < 90 秒",
                   "耗時 %.1fs%s" % (elapsed,
                   "（模型輸出格式異常，已走降級路徑）" if final.get("degraded") else ""))

        # 3. Markdown 匯出 ----------------------------------------
        session = {"id": "verify-m3", "started_at": "2026-09-09T09:00:00+08:00"}
        md = export.build_markdown(course, session, sections, final)
        check_markdown(gate, md, sections)

        # 4. 切回課中模型 -----------------------------------------
        t1 = time.monotonic()
        try:
            await llm.ensure("inclass")
            back_s = time.monotonic() - t1
            ok = llm.current == "inclass" and await llm.health()
        except LLMUnavailable as e:
            ok, back_s = False, 0.0
            gate.info("切回失敗：%s" % e)
        gate.check(ok, "切回課中模型後可正常再開一個 session",
                   "切回耗時 %.1fs，/health 正常" % back_s)
        merge_bench({"m3": {"switch_to_final_s": round(switch_s, 2),
                            "switch_back_s": round(back_s, 2),
                            "final_summary_s": round(elapsed, 2)}})
    finally:
        await llm.shutdown()
    if args.write_md and md:
        out = config.DATA_DIR / "verify_m3_sample.md"
        out.write_text(md, encoding="utf-8")
        gate.info("樣張已寫到 %s" % out)


def check_markdown(gate, md: str, sections) -> None:
    # 語法有效性
    try:
        from markdown_it import MarkdownIt
        MarkdownIt().parse(md)
        parse_ok, detail = True, "markdown-it-py 解析無誤"
    except ImportError:
        parse_ok, detail = False, "未安裝 markdown-it-py（pip install markdown-it-py）"
    except Exception as e:
        parse_ok, detail = False, str(e)
    gate.check(parse_ok, "匯出的 Markdown 語法有效", detail)

    # 時間戳格式
    stamps = re.findall(r"`\[(\d{2}:\d{2}:\d{2})\]`", md)
    expected = [fmt_ts(s["start_s"]) for s in sections]
    gate.check(stamps == expected, "時間戳格式正確（HH:MM:SS）",
               "找到 %d 個時間戳，前三個 %s" % (len(stamps), stamps[:3]))

    # 所有 section 都在
    missing = [s["title"] for s in sections if s["title"] not in md]
    gate.check(not missing, "所有 section 都在匯出結果中",
               "%d/%d 段" % (len(sections) - len(missing), len(sections))
               + ("，缺少：%s" % missing if missing else ""))

    # 使用者標註只在有 note 的段落出現
    notes = md.count("> 使用者標註：")
    expect_notes = sum(1 for s in sections if s.get("user_note"))
    gate.check(notes == expect_notes, "使用者標註僅在有 note 的段落輸出",
               "%d 則（預期 %d）" % (notes, expect_notes))


async def main_async(args) -> int:
    header("verify_m3.py — 閘門 G3：期末總結與模型切換（規格 §11 M3）")
    gate = Gate("G3")
    check_asr_unload(gate, args)
    await check_switch_and_final(gate, args)
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--course", default="ml-2026")
    ap.add_argument("--write-md", action="store_true",
                    help="把總結樣張寫到 data/verify_m3_sample.md")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
