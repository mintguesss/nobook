"""網頁上傳的錄音檔：分塊接收，再排隊轉成逐字稿。

分塊是為了 iPhone：Safari 一鎖螢幕或切走，進行中的請求就會斷掉。一次傳整個檔的
話要從頭來過；分塊的話回到畫面問一下伺服器收到哪裡，接著傳就好。

轉錄狀態存在 sessions.job（JSON），伺服器重啟後會自己接著做。一次只處理一個檔，
而且有人在現場錄音時先停下來讓路——現場錄音錯過就沒了，上傳的檔晚點轉沒關係。
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from . import audio_store, config, courses, diarize, meeting, storage, term_fix, ws_session
from .asr import ASRVramLow
from .audio_pipeline import is_oom

log = logging.getLogger(__name__)

UPLOAD_DIR = config.DATA_DIR / "uploads"
MAX_UPLOAD_BYTES = int(os.getenv("LS_UPLOAD_MAX_MB", "2048")) * 1024 * 1024
STALE_UPLOAD_S = 48 * 3600
# iPhone 語音備忘錄可能給 .m4a、.qta；通話錄音 App 常見 .caf
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".mp4",
              ".webm", ".mov", ".qta", ".caf", ".aif", ".aiff", ".wma"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PROGRESS_EVERY = 5       # 每轉幾段更新一次進度


class UploadError(ValueError):
    """訊息會直接顯示給使用者。"""


class _Cancelled(Exception):
    pass


# ── 接收檔案 ──────────────────────────────────────────────────────────
def _paths(upload_id: str):
    if not _ID_RE.match(upload_id or ""):
        raise UploadError("上傳編號不正確")
    return UPLOAD_DIR / (upload_id + ".json"), UPLOAD_DIR / (upload_id + ".part")


def create(filename: str, size: int, course_id: str, speakers: int = 0,
           terms: str = "", note: str = "", recorded_at: str = "") -> dict:
    import shutil
    ext = Path(filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise UploadError("這個檔案不是支援的錄音格式（%s）" % (ext or "沒有副檔名"))
    if size <= 0:
        raise UploadError("這個檔案是空的")
    if size > MAX_UPLOAD_BYTES:
        raise UploadError("檔案太大，上限 %d GB" % (MAX_UPLOAD_BYTES // (1024 ** 3)))
    courses.get_course(course_id)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # 轉成 FLAC 後也差不多大，留兩倍才不會轉到一半把硬碟塞滿
    if shutil.disk_usage(str(UPLOAD_DIR)).free < size * 2:
        raise UploadError("伺服器硬碟空間不夠")
    upload_id = uuid.uuid4().hex
    meta_path, part_path = _paths(upload_id)
    meta = {"filename": Path(filename).name, "ext": ext, "size": size,
            "course_id": course_id, "speakers": max(0, int(speakers or 0)),
            "terms": (terms or "")[:1000], "note": (note or "").strip()[:40],
            "recorded_at": _clean_time(recorded_at)}
    part_path.touch()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"id": upload_id, "received": 0}


def _clean_time(value: str) -> str:
    """檔案的修改時間就是開會時間的最佳估計；格式不對就用現在。"""
    try:
        ts = datetime.datetime.fromisoformat((value or "").replace("Z", "+00:00"))
        return ts.astimezone().isoformat()
    except ValueError:
        return datetime.datetime.now().astimezone().isoformat()


def received(upload_id: str) -> int:
    meta_path, part_path = _paths(upload_id)
    if not meta_path.exists():
        raise UploadError("找不到這次上傳，請重新選檔")
    return part_path.stat().st_size


def _load_meta(upload_id: str) -> dict:
    meta_path, _ = _paths(upload_id)
    if not meta_path.exists():
        raise UploadError("找不到這次上傳，請重新選檔")
    return json.loads(meta_path.read_text(encoding="utf-8"))


async def write_chunk(upload_id: str, offset: int, stream) -> int:
    """從 offset 接著寫。前一塊只寫到一半就斷線時，截掉重複的部分。"""
    meta = _load_meta(upload_id)
    _, part_path = _paths(upload_id)
    have = part_path.stat().st_size
    if offset > have:
        return have            # 前端照回傳值重新對齊
    with open(part_path, "r+b") as f:
        f.seek(offset)
        f.truncate()
        size = offset
        async for chunk in stream:
            size += len(chunk)
            if size > meta["size"]:
                raise UploadError("收到的資料比檔案還大")
            f.write(chunk)
    return size


def discard(upload_id: str) -> None:
    for p in _paths(upload_id):
        p.unlink(missing_ok=True)


def complete(upload_id: str) -> dict:
    """收齊了：建立紀錄並排進佇列。"""
    meta = _load_meta(upload_id)
    meta_path, part_path = _paths(upload_id)
    if part_path.stat().st_size != meta["size"]:
        raise UploadError("檔案還沒傳完")
    session_id = str(uuid.uuid4())
    source = UPLOAD_DIR / (session_id + meta["ext"])
    os.replace(part_path, source)
    storage.create_session(session_id, meta["course_id"], meta["recorded_at"])
    storage.set_note(session_id, meta["note"] or Path(meta["filename"]).stem[:40])
    storage.set_job(session_id, {
        "status": "queued", "stage": "queued", "progress": 0, "source": str(source),
        "speakers": meta["speakers"], "terms": meta["terms"],
        "diarize": meta["course_id"] == "meeting", "queued_at": time.time()})
    meta_path.unlink(missing_ok=True)
    enqueue(session_id)
    return {"session_id": session_id}


def remove_source(session_id: str) -> None:
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.glob(session_id + ".*"):
            p.unlink(missing_ok=True)


def clean_stale() -> None:
    """放棄的上傳不要永遠佔著硬碟。"""
    if not UPLOAD_DIR.exists():
        return
    now = time.time()
    for p in list(UPLOAD_DIR.glob("*.part")):
        if now - p.stat().st_mtime > STALE_UPLOAD_S:
            discard(p.stem)


# ── 轉錄佇列 ──────────────────────────────────────────────────────────
_queue = None          # type: asyncio.Queue | None
_cancelled = set()
_deps = {}


def enqueue(session_id: str) -> None:
    if _queue is not None:
        _queue.put_nowait(session_id)


def cancel(session_id: str) -> None:
    _cancelled.add(session_id)


def retranscribe(session_id: str, speakers: int) -> None:
    """用保存的錄音重轉，例如講者人數判斷錯了、或現場錄的要補標講者。"""
    s = storage.get_session(session_id)
    if s is None:
        raise UploadError("找不到這筆紀錄")
    job = load_job(s)
    if job.get("status") in ("queued", "running"):
        raise UploadError("這筆正在處理中")
    source = job.get("source") if job.get("status") == "failed" else None
    source = source if source and os.path.exists(source) else s.get("audio_path")
    if not source or not os.path.exists(source):
        raise UploadError("這筆沒有保存錄音，沒辦法重轉")
    storage.set_job(session_id, {
        "status": "queued", "stage": "queued", "progress": 0, "source": source,
        "speakers": max(0, int(speakers or 0)), "terms": job.get("terms", ""),
        "diarize": s["course_id"] == "meeting", "queued_at": time.time()})
    enqueue(session_id)


def load_job(session: dict) -> dict:
    try:
        return json.loads(session.get("job") or "{}") or {}
    except ValueError:
        return {}


def live_recording() -> bool:
    return any(s.ws is not None and not s.ended
               for s in ws_session.all_sessions().values())


def _update(session_id: str, job: dict, **fields) -> None:
    job.update(fields)
    storage.set_job(session_id, job)


async def _yield_to_live(session_id: str, job: dict) -> None:
    resume_stage = job.get("stage")
    while live_recording():
        if job.get("stage") != "waiting_live":
            _update(session_id, job, stage="waiting_live")
        await asyncio.sleep(3)
    if job.get("stage") == "waiting_live":
        _update(session_id, job, stage=resume_stage)


def _check_cancel(session_id: str) -> None:
    if session_id in _cancelled:
        raise _Cancelled()


async def _transcribe_with_retry(engine, llm, *args):
    """顯存被摘要模型佔住時先把它關掉再試。逐字稿優先（規格 §7.4）。"""
    try:
        return await asyncio.to_thread(meeting.transcribe_segment, engine, *args)
    except Exception as e:
        if not (isinstance(e, ASRVramLow) or is_oom(e)):
            raise
        log.warning("轉錄時 VRAM 不足，卸載摘要模型後重試：%s", e)
        await llm.shutdown()
        await asyncio.sleep(1.0)
        return await asyncio.to_thread(meeting.transcribe_segment, engine, *args)


async def _prepare_audio(session_id, s, job, course):
    """解碼；來源不是已保存的 FLAC 就先存一份，之後播放和重轉都用它。"""
    from faster_whisper.audio import decode_audio
    source = job["source"]
    audio = await asyncio.to_thread(decode_audio, source,
                                    sampling_rate=config.SAMPLE_RATE)
    if source == s.get("audio_path"):
        return audio, source
    import soundfile as sf
    dest = audio_store.build_filename(course, s["started_at"], session_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(sf.write, str(dest), audio, config.SAMPLE_RATE,
                            format="FLAC", subtype="PCM_16")
    storage.set_audio_path(session_id, str(dest))
    Path(source).unlink(missing_ok=True)
    _update(session_id, job, source=str(dest))
    return audio, str(dest)


async def _run_job(session_id: str) -> None:
    engine, llm, finish_notes = _deps["engine"], _deps["llm"], _deps["finish_notes"]
    s = storage.get_session(session_id)
    if s is None:
        return
    job = load_job(s)
    if job.get("status") not in ("queued", "running"):
        return
    from .vad import make_vad
    course = courses.get_course(s["course_id"])
    glossary, corrections = meeting.parse_terms(job.get("terms", ""))
    course = meeting.with_terms(course, glossary, corrections)
    fixer = term_fix.TermFixer(course.glossary, corrections=course.corrections)

    _update(session_id, job, status="running", stage="decoding", progress=0, error=None)
    await _yield_to_live(session_id, job)
    audio, audio_path = await _prepare_audio(session_id, s, job, course)
    duration = len(audio) / float(config.SAMPLE_RATE)
    _update(session_id, job, duration_s=duration, started_ts=time.time())
    # VAD 帶狀態，不能跟現場錄音共用
    vad = await asyncio.to_thread(make_vad)

    turns = []
    if job.get("diarize") and diarize.available():
        _check_cancel(session_id)
        _update(session_id, job, stage="speakers")
        turns = await asyncio.to_thread(diarize.diarize, audio, vad,
                                        job.get("speakers") or 0)

    _check_cancel(session_id)
    _update(session_id, job, stage="transcribing", progress=0)
    segments = await asyncio.to_thread(meeting.split_segments, audio, vad)
    rows, fixes = [], []
    for i, seg in enumerate(segments):
        _check_cancel(session_id)
        await _yield_to_live(session_id, job)
        rows += await _transcribe_with_retry(
            engine, llm, fixer, seg, course.asr_prompt, turns, fixes)
        if i % PROGRESS_EVERY == 0:
            _update(session_id, job, progress=round(i / max(1, len(segments)), 3))

    rows = meeting.merge_rows(rows)
    storage.replace_segments(session_id, rows)
    speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
    _update(session_id, job, status="done", stage="done", progress=1,
            found_speakers=len(speakers), fixes=len(fixes),
            empty=not rows)

    fresh = storage.get_session(session_id)
    if job.get("diarize") or not rows:
        # 會議的整理格式另外定，這裡不套課堂筆記的 prompt
        storage.finish_session(
            session_id, fresh.get("ended_at") or _now_iso(), duration,
            fresh.get("final_md"), fresh.get("handcopy_md"), audio_path)
    else:
        try:
            await finish_notes(session_id, fresh)
        except Exception as e:
            # 逐字稿已經存好了，筆記失敗可以從詳細頁重試
            log.warning("上傳檔產生筆記失敗：%s", e)
            _update(session_id, job, notes_error=str(e))


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _friendly_error(e: Exception) -> str:
    text = str(e).lower()
    if any(k in text for k in ("invalid data", "invaliddata", "no audio", "decode",
                               "av.error", "end of file")):
        return "這個檔案讀不出聲音，可能已損毀或不是錄音檔"
    if is_oom(e) or isinstance(e, ASRVramLow):
        return "伺服器的顯示卡記憶體不夠，請稍後重試"
    return "處理時發生問題：%s" % e


async def _worker() -> None:
    while True:
        session_id = await _queue.get()
        try:
            await _run_job(session_id)
        except _Cancelled:
            log.info("已取消轉錄：%s", session_id)
        except Exception as e:
            log.exception("轉錄失敗：%s", session_id)
            s = storage.get_session(session_id)
            if s is not None:
                job = load_job(s)
                _update(session_id, job, status="failed", stage="failed",
                        error=_friendly_error(e))
        finally:
            _cancelled.discard(session_id)
            _queue.task_done()
        # 佇列清空就把 ASR 卸載，顯存還給下一堂課的摘要
        if _queue.empty() and not ws_session.all_sessions():
            await asyncio.to_thread(_deps["engine"].unload)


def start(engine, llm, finish_notes):
    """啟動 worker，並把重啟前沒做完的接回來。回傳 task 供關閉時取消。"""
    global _queue
    _queue = asyncio.Queue()
    _deps.update(engine=engine, llm=llm, finish_notes=finish_notes)
    clean_stale()
    for row in storage.list_unfinished_jobs():
        job = load_job(row)
        if job.get("status") in ("queued", "running"):
            enqueue(row["id"])
    return asyncio.create_task(_worker(), name="upload-worker")
