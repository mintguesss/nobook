"""把別台錄好的錄音匯進來：重新轉逐字稿、重新產生筆記。

用在兩台電腦之間交接——只搬錄音檔（筆記不搬），到這台用這台的模型
重新跑一次。流程跟上課時完全一樣：VAD 切段 → ASR（帶課程術語表當
initial_prompt）→ 近似音修正 → 段落筆記 → 手抄版 → 完整版。

用法：
    python scripts/import_audio.py <交接資料夾>
    python scripts/import_audio.py <交接資料夾> --force    # 已存在的也重做

交接資料夾裡要有 manifest.json 與 audio/*.flac（由另一台打包產生）。
**先把伺服器關掉**：這支會自己載 ASR 和摘要模型，跟伺服器搶 GPU 與 port。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from _common import Gate, header

from server import config, courses, export, storage, term_fix
from server.audio_pipeline import AudioPipeline
from server.llm_manager import LLMManager, LLMUnavailable
from server.summarizer import Summarizer

# 離線灌音訊時的節流門檻。pipeline 的佇列滿了會「丟段落」而不是等——
# 上課時音訊是即時進來的所以永遠不會滿，但離線一次灌 72 分鐘一定爆，
# 而且丟掉的段落不會有任何錯誤，逐字稿就默默少了幾段。
MAX_QUEUE = 8
FEED_SECONDS = 30


def server_running() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/health" % config.PORT,
                                    timeout=2):
            return True
    except Exception:
        return False


def load_flac(path: Path):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        raise ValueError("%s 的取樣率是 %d，不是 %d" % (path.name, sr, config.SAMPLE_RATE))
    return np.ascontiguousarray(audio)


async def transcribe(entry, course, audio, engine):
    """用跟上課一樣的 pipeline 轉逐字稿，直接寫進資料庫。"""
    fixer = term_fix.for_course(course)
    count = {"n": 0, "chars": 0}

    def on_result(seg, res):
        text = res["text"]
        if fixer.enabled:
            text, _ = fixer.fix(text)
        storage.insert_segment(entry["id"], seg.start_s, seg.end_s, text,
                               res["avg_logprob"])
        count["n"] += 1
        count["chars"] += len(text)

    pipe = AudioPipeline(engine, initial_prompt=course.asr_prompt,
                         on_result=on_result)
    pipe.start()
    step = FEED_SECONDS * config.SAMPLE_RATE
    total = len(audio)
    t0 = time.time()
    last_pct = -1
    for i in range(0, total, step):
        await pipe.feed(audio[i:i + step])
        while pipe.queue_depth > MAX_QUEUE:
            await asyncio.sleep(0.05)
        pct = int((i + step) * 100 / total)
        if pct // 10 != last_pct // 10:
            print("     轉錄 %3d%%（%d 段）" % (min(pct, 100), count["n"]), flush=True)
            last_pct = pct
    await pipe.flush()
    await pipe.drain()
    await pipe.stop()
    if pipe.dropped:
        print("     [!!] 佇列溢出丟了 %d 段" % pipe.dropped, flush=True)
    return count, time.time() - t0


async def summarize(entry, course, summarizer):
    segs = storage.list_segments(entry["id"])
    if not segs:
        return None
    text = "".join(g["text"] for g in segs).strip()
    spans = await summarizer.summarize_span(course, [], text, 0.0, segs[-1]["end_s"])
    sections = [{"start_s": sp["start_s"], "end_s": sp["end_s"],
                 "title": sp["title"], "bullets": sp["bullets"],
                 "summary": sp.get("summary", ""),
                 "groups": sp.get("groups") or [],
                 "user_note": None} for sp in spans]
    storage.replace_sections(entry["id"], sections)
    row = storage.get_session(entry["id"])
    notes = await summarizer.summarize_handcopy(course, sections, model_key="final")
    handcopy_md = export.build_handcopy(course, row, notes)
    final = await summarizer.summarize_final(course, sections)
    final_md = export.build_markdown(course, row, sections, final)
    return sections, handcopy_md, final_md


async def main_async(args) -> int:
    header("import_audio.py — 匯入錄音、重轉逐字稿、重產筆記")
    gate = Gate("IMPORT")
    src = Path(args.folder)
    man_path = src / "manifest.json"
    if not man_path.exists():
        gate.check(False, "找到 manifest.json", str(man_path))
        return gate.finish()
    if server_running() and not args.ignore_server:
        gate.check(False, "伺服器已關閉",
                   "偵測到伺服器在跑。這支會自己載模型，請先關掉伺服器再執行")
        return gate.finish()

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    storage.init()
    audio_dir = Path(config.AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)

    from server.asr import ASREngine
    bench = config.load_bench(required=True)
    engine = ASREngine(bench.asr_model_path, device=bench.asr_device,
                       compute_type=bench.asr_compute_type)
    llm = LLMManager(bench)
    summarizer = Summarizer(llm)

    todo = []
    for e in manifest:
        if storage.get_session(e["id"]) and not args.force:
            gate.info("  %s 已經在資料庫裡，略過（要重做加 --force）" % e["id"][:8])
            continue
        todo.append(e)
    if not todo:
        gate.check(True, "沒有需要匯入的", "全部都已存在")
        return gate.finish()

    ok = 0
    try:
        # 1. 全部先轉逐字稿（只載一次 ASR）
        engine.load()
        for e in todo:
            course = courses.get_course(e["course_id"])
            flac = src / "audio" / e["file"]
            print("\n  %s  %s  %s  %.0f 分鐘"
                  % (e["id"][:8], e["started_at"][5:16], course.name,
                     (e.get("duration_s") or 0) / 60), flush=True)
            dst = audio_dir / e["file"]
            if not dst.exists():
                shutil.copy2(flac, dst)
            if storage.get_session(e["id"]):
                storage.delete_session(e["id"])      # --force：整筆重建
            storage.create_session(e["id"], course.id, e["started_at"])
            audio = load_flac(dst)
            cnt, secs = await transcribe(e, course, audio, engine)
            e["_dur"] = len(audio) / config.SAMPLE_RATE
            e["_dst"] = str(dst)
            print("     逐字稿 %d 段、%d 字（%.0fs，RTF %.2f）"
                  % (cnt["n"], cnt["chars"], secs, secs / max(e["_dur"], 1)),
                  flush=True)
        # 2. 卸載 ASR 讓出 VRAM，再輪流產筆記
        engine.unload()
        for e in todo:
            course = courses.get_course(e["course_id"])
            t0 = time.time()
            res = await summarize(e, course, summarizer)
            if res is None:
                gate.check(False, "%s 產生筆記" % e["id"][:8], "沒有逐字稿")
                continue
            sections, handcopy_md, final_md = res
            ended = e.get("ended_at") or datetime.datetime.now().astimezone().isoformat()
            storage.finish_session(e["id"], ended, e["_dur"], final_md,
                                   handcopy_md, e["_dst"])
            ok += 1
            gate.info("  %s  段落 %d、手抄版 %d 字、完整版 %d 字（%.0fs）"
                      % (e["id"][:8], len(sections), len(handcopy_md),
                         len(final_md), time.time() - t0))
    except LLMUnavailable as ex:
        gate.check(False, "摘要模型可用", str(ex))
    finally:
        try:
            engine.unload()
        except Exception:
            pass
        await llm.shutdown()
        storage.close()

    gate.check(ok == len(todo), "全部匯入完成", "%d/%d" % (ok, len(todo)))
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="交接資料夾（含 manifest.json 與 audio/）")
    ap.add_argument("--force", action="store_true", help="已存在的也整筆重做")
    ap.add_argument("--ignore-server", action="store_true",
                    help="伺服器開著也照跑（只在伺服器閒置、確定沒在錄音時用）")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
