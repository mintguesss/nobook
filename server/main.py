"""FastAPI app、路由註冊（規格 §3.2、§5、§10）。"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import (asr, audio_pipeline, config, courses, export, storage,
               term_fix, ws_session)
from .llm_manager import LLMManager, LLMUnavailable
from .summarizer import Summarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("nobook")

state = {"engine": None, "llm": None, "summarizer": None, "bench": None}


async def _idle_unload_loop(llm) -> None:
    """課上完之後把 llama-server 佔的 VRAM 還出去。

    有進行中的課就不動——課中按鈕之間本來就會隔很久，那不叫閒置。
    """
    idle_s = config.LLAMA_IDLE_UNLOAD_S
    if idle_s <= 0:
        return
    while True:
        await asyncio.sleep(min(60.0, idle_s / 2))
        try:
            if ws_session.all_sessions():
                llm.last_used = time.monotonic()
                continue
            await llm.unload_if_idle(idle_s)
        except Exception as e:     # 這條路掛掉不能影響服務
            log.warning("閒置卸載檢查失敗：%s", e)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 規格 §14.1：bench.json 不存在就拒絕啟動
    bench = config.load_bench(required=True)
    state["bench"] = bench
    storage.init()
    state["engine"] = asr.get_engine()
    llm = LLMManager(bench)
    state["llm"] = llm
    state["summarizer"] = Summarizer(llm)
    log.info("ASR 模型：%s（%s / %s）", bench.asr_model_path,
             bench.asr_device, bench.asr_compute_type)
    inclass = bench.inclass_model
    log.info("課中摘要模型：%s", inclass.model if inclass else "（未選型，摘要停用）")
    # 先把 silero-vad 載進來，第一堂課的第一段才不會多等模型載入
    asyncio.get_running_loop().run_in_executor(None, audio_pipeline.default_vad)
    idle_task = asyncio.create_task(_idle_unload_loop(llm))
    try:
        yield
    finally:
        idle_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await idle_task
        with contextlib.suppress(Exception):
            await llm.shutdown()
        with contextlib.suppress(Exception):
            state["engine"].unload()
        storage.close()


app = FastAPI(title="Lecture Scribe", lifespan=lifespan)


# ── WebSocket（規格 §5）────────────────────────────────────────────────
@app.websocket("/ws/session")
async def ws_endpoint(ws: WebSocket, course_id: str = Query("general")):
    await ws_session.handle_connection(
        ws, course_id, state["engine"], state["summarizer"], state["llm"])


# ── REST（規格 §10）───────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    llm = state["llm"]
    return {
        "ok": True,
        "asr_loaded": state["engine"].loaded if state["engine"] else False,
        "llm_model": llm.current if llm else None,
        "llm_alive": (await llm.health()) if llm else False,
        "summary_degraded": llm.degraded if llm else True,
        "active_sessions": list(ws_session.all_sessions().keys()),
    }


@app.get("/api/courses")
async def api_courses():
    return courses.list_courses()


@app.get("/api/sessions")
async def api_sessions(limit: int = 100):
    return storage.list_sessions(limit)


@app.get("/api/sessions/{session_id}")
async def api_session(session_id: str):
    s = storage.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    import os
    audio = s.get("audio_path")
    # 還在重連佇列裡的才接得回去。伺服器重啟過的話記憶體裡什麼都沒有，
    # 只能收尾——這兩種情況前端要顯示不同的按鈕，不能讓使用者按了才發現。
    live = ws_session.get_session(session_id)
    return {"session": s,
            "has_audio": bool(audio and os.path.exists(audio)),
            "resumable": bool(live is not None and not live.ended
                              and not s.get("ended_at")),
            "sections": storage.list_sections(session_id),
            "segments": storage.list_segments(session_id)}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """刪除一堂課：逐字稿、摘要、錄音檔一起清掉。"""
    import os
    if storage.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    audio = storage.delete_session(session_id)
    removed_audio = False
    if audio and os.path.exists(audio):
        try:
            os.remove(audio)
            removed_audio = True
        except OSError as e:
            log.warning("刪除錄音檔失敗 %s：%s", audio, e)
    return {"ok": True, "audio_removed": removed_audio}


@app.post("/api/sessions/{session_id}/finish")
async def api_finish(session_id: str):
    """把一堂沒有正常結束的課收尾，產生筆記。

    使用者離開一陣子回來，紀錄顯示「未結束」，但既不能接續錄音也不能結束，
    那堂課的逐字稿就永遠卡在那裡。發生原因有兩種，兩種都要能救：

      1. session 還在重連佇列裡（手機睡著、網路斷掉）——直接跑它自己的
         結束流程，錄音檔才會被正常關閉。
      2. 伺服器重啟過，記憶體裡什麼都沒有了——只能從資料庫的逐字稿重建。
    """
    s = storage.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    if s.get("ended_at"):
        return {"ok": True, "already_ended": True}

    live = ws_session.get_session(session_id)
    if live is not None and not live.ended:
        try:
            await live.handle_end()
        finally:
            ws_session.drop_session(session_id)
        s = storage.get_session(session_id)
        return {"ok": True, "source": "live",
                "sections": len(storage.list_sections(session_id)),
                "has_handcopy": bool(s.get("handcopy_md")),
                "has_final": bool(s.get("final_md"))}

    ws_session.drop_session(session_id)
    n = await _finish_from_db(session_id, s)
    return dict({"ok": True, "source": "db"}, **n)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().astimezone().isoformat()


async def _finish_from_db(session_id: str, s: dict) -> dict:
    """伺服器重啟後只剩資料庫時，從逐字稿把筆記補出來。"""
    segs = storage.list_segments(session_id)
    if not segs:
        storage.finish_session(session_id, _now_iso(), 0.0, None, None,
                               s.get("audio_path"))
        return {"sections": 0, "has_handcopy": False, "has_final": False,
                "note": "沒有逐字稿，只把狀態標成已結束"}

    course = courses.get_course(s["course_id"])
    summarizer = state["summarizer"]
    old = storage.list_sections(session_id)
    covered = max([x["end_s"] for x in old] or [0.0])
    tail = [g for g in segs if g["start_s"] >= covered]
    new_secs = [dict(x) for x in old]
    if tail:
        text = "".join(g["text"] for g in tail).strip()
        fixer = term_fix.for_course(course)
        if fixer.enabled:
            text, _ = fixer.fix(text)
        spans = await summarizer.summarize_span(
            course, new_secs[-2:], text, covered, segs[-1]["end_s"])
        for sp in spans:
            new_secs.append({"start_s": sp["start_s"], "end_s": sp["end_s"],
                             "title": sp["title"], "bullets": sp["bullets"],
                             "summary": sp.get("summary", ""),
                             "groups": sp.get("groups") or [],
                             "user_note": None})
        storage.replace_sections(session_id, new_secs)

    notes = await summarizer.summarize_handcopy(course, new_secs,
                                                model_key="final")
    handcopy_md = export.build_handcopy(course, s, notes)
    final = await summarizer.summarize_final(course, new_secs)
    final_md = export.build_markdown(course, s, new_secs, final)
    duration = s.get("duration_s") or segs[-1]["end_s"]
    storage.finish_session(session_id, _now_iso(), duration,
                           final_md, handcopy_md, s.get("audio_path"))
    return {"sections": len(new_secs), "has_handcopy": True,
            "has_final": True,
            "recovered_chars": sum(len(g["text"]) for g in tail)}


@app.get("/api/materials")
async def api_materials():
    """教材有哪些，照課程資料夾分。

    放在 materials/ 根目錄（沒分資料夾）的檔案會被列成「未分類」，
    而且**不會拿來對照**——攤平的話每堂課都會去比對所有教材，
    實測管理資訊系統那堂跑去對照生產與作業管理的投影片。
    """
    from . import materials
    files = materials.list_materials()
    groups = {}
    for f in files:
        groups.setdefault(f["course_id"] or "", []).append(f)
    known = {c["id"]: c["name"] for c in courses.list_courses()}
    names = {c["name"] for c in courses.list_courses()}
    out = []
    for key, fs in sorted(groups.items()):
        out.append({"folder": key,
                    "course": known.get(key) or (key if key in names else None),
                    "known": bool(key) and (key in known or key in names),
                    "files": fs})
    return {"dir": str(config.MATERIALS_DIR), "groups": out,
            "hint": "把投影片放到 materials/<課程代號或課名>/ 底下"}


@app.post("/api/sessions/{session_id}/align")
async def api_align(session_id: str):
    """把這堂課的逐字稿對到投影片的頁碼。

    純 CPU，不用模型，一堂課一兩秒。結果存在 session 上，之後直接讀。
    """
    from . import materials
    s = storage.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    segs = storage.list_segments(session_id)
    if not segs:
        raise HTTPException(status_code=400, detail="這堂沒有逐字稿")
    # 只比對這門課自己的教材。不限定的話每堂課都會去對照所有課的投影片。
    course = courses.get_course(s["course_id"])
    if not materials.list_materials(course=course):
        raise HTTPException(
            status_code=400,
            detail=r"這門課還沒有教材。把投影片放到 %s\%s\ 底下"
                   % (config.MATERIALS_DIR, course.name or course.id))

    res = await asyncio.to_thread(materials.align_all, segs, None,
                                  materials.MATCH_THRESHOLD, course)
    payload = {
        "coverage": res["coverage"],
        "matched": res["matched"],
        "chunks": len(res["chunks"]),
        "materials": res["materials"],
        "pages": materials.page_ranges(res),
    }
    storage.set_alignment(session_id, payload)
    return payload


@app.get("/api/sessions/{session_id}/align")
async def api_align_get(session_id: str):
    if storage.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return storage.get_alignment(session_id) or {}


@app.delete("/api/sessions/{session_id}/align")
async def api_align_clear(session_id: str):
    storage.set_alignment(session_id, None)
    return {"ok": True}


@app.get("/api/merge/groups")
async def api_merge_groups():
    """可以合併的組：同一天、同一門課、兩筆以上。"""
    from . import merge
    return merge.mergeable_groups()


@app.post("/api/sessions/merge")
async def api_merge(payload: dict = Body(...)):
    """把幾段錄音接成一堂，重新產生總筆記。

    產生的是一筆新紀錄，原本那幾筆留著不動——合併壞掉的話原始逐字稿
    還在，使用者也可以自己決定要不要刪掉舊的。
    """
    from . import merge
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or len(ids) < 2:
        raise HTTPException(status_code=400, detail="請至少選兩堂")
    try:
        return await merge.merge_sessions(ids, state["summarizer"])
    except merge.MergeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/storage")
async def api_storage():
    """磁碟用量，讓前端能回答「錄音能存多久」。"""
    import os
    import shutil
    n_ses, n_seg, db_bytes = storage.storage_usage()
    audio_dir = config.AUDIO_DIR
    audio_bytes = 0
    audio_files = 0
    if audio_dir.exists():
        for f in audio_dir.glob("*.flac"):
            audio_bytes += f.stat().st_size
            audio_files += 1
    free = shutil.disk_usage(str(config.DATA_DIR)).free
    return {"sessions": n_ses, "segments": n_seg, "db_bytes": db_bytes,
            "audio_files": audio_files, "audio_bytes": audio_bytes,
            "disk_free_bytes": free, "save_audio": config.SAVE_AUDIO}


@app.get("/api/sessions/{session_id}/audio")
async def api_audio(session_id: str):
    """回放整堂課的錄音（規格 §10 的選用保存）。"""
    from fastapi.responses import FileResponse
    s = storage.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    path = s.get("audio_path")
    if not path or not __import__("os").path.exists(path):
        raise HTTPException(status_code=404, detail="這堂課沒有保存錄音")
    return FileResponse(path, media_type="audio/flac",
                        filename="%s.flac" % session_id)


def _notes_markdown(s, session_id, which: str) -> str:
    """which: full=完整版、handcopy=手抄版。缺的話即時組一份。"""
    if which == "handcopy":
        md = s.get("handcopy_md")
        if md:
            return md
    md = s.get("final_md")
    if md and which == "full":
        # 完整版是課後整理時存下來的，那時還沒對照投影片，所以頁碼要後補
        return export.annotate_slides(md, storage.get_alignment(session_id))
    course = courses.get_course(s["course_id"])
    return export.build_markdown(
        course, s, storage.list_sections(session_id),
        {"overview": ["（本堂尚未執行課後整理）"], "open_questions": []},
        align=storage.get_alignment(session_id))


def _filename(s, which: str, ext: str) -> str:
    course = s.get("course_id", "lecture")
    try:
        course = courses.get_course(s["course_id"]).name
    except Exception:
        pass
    date = (s.get("started_at") or "")[:10]
    label = {"full": "完整版", "handcopy": "手抄版",
             "transcript": "逐字稿"}.get(which, which)
    return "%s_%s_%s.%s" % (course, date, label, ext)


@app.get("/api/sessions/{session_id}/export")
async def api_export(
    session_id: str,
    format: str = Query("md", pattern="^(md|docx|txt|json)$"),
    which: str = Query("full", pattern="^(full|handcopy|transcript)$"),
):
    """匯出筆記。

    format=md 給 Markdown 原文；format=docx 轉成 Word 的原生樣式
    （Heading/List Bullet），不是把 # 和 - 原樣塞進去——那樣在 Word 裡
    既不能用導覽窗格跳轉，複製到別的文件也帶不走格式。
    """
    from fastapi.responses import Response
    from urllib.parse import quote

    s = storage.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")

    if format == "json":
        return JSONResponse(content=json.loads(export.build_json(
            s, storage.list_segments(session_id),
            storage.list_sections(session_id))))

    if which == "transcript" or format == "txt":
        body = export.build_txt(s, storage.list_segments(session_id))
        which = "transcript"
    else:
        body = _notes_markdown(s, session_id, which)

    name = _filename(s, which, "docx" if format == "docx" else
                     ("txt" if format == "txt" else "md"))
    disp = "attachment; filename*=UTF-8''%s" % quote(name)

    if format == "docx":
        from . import docx_export
        try:
            data = docx_export.markdown_to_docx_bytes(body, title=name)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="伺服器未安裝 python-docx，無法輸出 Word 檔")
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument."
                       "wordprocessingml.document",
            headers={"Content-Disposition": disp})

    media = ("text/plain; charset=utf-8" if format == "txt"
             else "text/markdown; charset=utf-8")
    return Response(content=body.encode("utf-8"), media_type=media,
                    headers={"Content-Disposition": disp})


@app.post("/api/llm/ensure/{model_key}")
async def api_llm_ensure(model_key: str):
    """手動預熱／切換摘要模型（course 前先按一次可省掉第一則摘要的冷啟動）。"""
    if model_key not in ("inclass", "final"):
        raise HTTPException(status_code=400, detail="model_key must be inclass|final")
    try:
        await state["llm"].ensure(model_key)
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"ok": True, "current": state["llm"].current,
            "switch_s": state["llm"].last_switch_s}


@app.get("/api/bench")
async def api_bench():
    return state["bench"].raw if state["bench"] else {}


# ── 靜態 PWA ──────────────────────────────────────────────────────────
if config.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    @app.get("/")
    async def _no_web():
        return HTMLResponse("<h1>web/ 尚未建立</h1>", status_code=500)


def run():
    import uvicorn
    uvicorn.run("server.main:app", host=config.HOST, port=config.PORT,
                ws_max_size=16 * 1024 * 1024, log_level="info")


if __name__ == "__main__":
    run()
