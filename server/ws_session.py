"""WebSocket 連線處理與 session 狀態機（規格 §5、§6）。"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from . import (asr, audio_store, config, courses, export, gpu, storage,
               term_fix, verify_notes)
from .audio_pipeline import AudioPipeline
from .llm_manager import LLMUnavailable

log = logging.getLogger(__name__)

HEADER = struct.Struct("<II")     # uint32 seq + uint32 sampleCount（小端序，§4.1）
HEADER_SIZE = HEADER.size


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class LectureSession:
    """一堂課的完整狀態。斷線重連時由 registry 取回同一個實例。"""

    def __init__(self, session_id, course, engine, summarizer, llm, vad=None):
        self.id = session_id
        self.course = course
        self.engine = engine
        self.summarizer = summarizer
        self.llm = llm

        self.started_at = _now_iso()
        self.ws = None
        self.paused = False
        self.ended = False

        self.sections = []        # [{seq,row_id,start_s,end_s,title,bullets,user_note}]
        self.cursor = 0.0         # 上次按鈕的時間戳（秒）
        self.buffer = []          # cursor 之後尚未摘要的 segments
        self.seen_seqs = set()
        self.audio_pos = 0.0      # 已收音訊的總長度（秒）
        self.summarizing = False  # mark 去抖動旗標（規格 §6.2）
        self.notes_busy = False   # 手抄版筆記生成中，同樣要去抖動
        self.summary_degraded = False
        self.last_error = None

        self.pipeline = AudioPipeline(
            engine=engine,
            initial_prompt=course.asr_prompt,
            on_result=self._on_segment,
            on_error=self._on_pipeline_error,
            on_oom=self._on_oom,
            vad=vad,
        )
        # 規格 §10：原始音訊預設不保存；開了 LS_SAVE_AUDIO 才錄，
        # 用途是回頭確認 ASR 有沒有聽錯、以及日後換模型重跑
        self.recorder = (audio_store.AudioRecorder(session_id, course=course,
                                                   started_at=self.started_at)
                         if config.SAVE_AUDIO else None)
        # 近似音自動修正：靠課程術語表把「電動成本」改回「變動成本」。
        # 每次修正都記下來，方便事後檢查有沒有改錯。
        self.fixer = term_fix.for_course(course)
        self.term_corrections = []
        self.audio_path = None
        self._send_lock = asyncio.Lock()
        self._tasks = set()
        self._stats_task = None

    def start_stats(self) -> None:
        """每 5 秒推一次 stats（規格 §5.2）。重連時沿用同一個 task。"""
        if self._stats_task is None or self._stats_task.done():
            self._stats_task = asyncio.create_task(_stats_loop(self))

    def stop_stats(self) -> None:
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None

    # 傳送 -------------------------------------------------------------
    async def send(self, payload: dict) -> None:
        ws = self.ws
        if ws is None:
            return
        async with self._send_lock:
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                log.debug("送出事件失敗（可能已斷線）：%s", e)
                self.ws = None

    async def send_error(self, code: str, message: str, fatal: bool = False) -> None:
        self.last_error = {"code": code, "message": message}
        await self.send({"type": "error", "code": code,
                         "message": message, "fatal": fatal})

    def spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # 音訊 -------------------------------------------------------------
    async def handle_audio(self, data: bytes) -> None:
        if len(data) < HEADER_SIZE:
            return
        seq, sample_count = HEADER.unpack_from(data, 0)
        payload = data[HEADER_SIZE:]
        if seq in self.seen_seqs:      # 重連補送的重複 chunk（規格 §5.3）
            return
        self.seen_seqs.add(seq)
        expected = sample_count * 2
        if expected and len(payload) != expected:
            log.warning("chunk seq=%d 長度不符：宣告 %d sample，實得 %d bytes",
                        seq, sample_count, len(payload))
        if self.paused or not payload:
            return
        pcm = pcm16_to_float32(payload)
        self.audio_pos += len(pcm) / float(config.SAMPLE_RATE)
        if self.recorder is not None:
            await asyncio.to_thread(self.recorder.write, pcm)
        await self.pipeline.feed(pcm, fed_at=time.monotonic())

    async def _on_segment(self, seg, res) -> None:
        text = res["text"]
        if self.fixer.enabled:
            text, corr = self.fixer.fix(text)
            if corr:
                self.term_corrections.extend(corr)
                log.info("近似音修正：%s",
                         "、".join("%s→%s" % (a, b) for a, b, _ in corr))
        row_id = await asyncio.to_thread(
            self._store_segment, seg.start_s, seg.end_s,
            text, res["avg_logprob"])
        item = {"id": row_id, "start": seg.start_s, "end": seg.end_s,
                "text": text, "avg_logprob": res["avg_logprob"]}
        self.buffer.append(item)
        await self.send({"type": "segment", **item})

    def _store_segment(self, start_s, end_s, text, avg_logprob):
        """磁碟寫入失敗不能影響即時串流（規格 §14.6）。"""
        try:
            return storage.insert_segment(self.id, start_s, end_s, text, avg_logprob)
        except Exception as e:
            log.error("segment 寫入資料庫失敗（逐字稿續行）：%s", e)
            return -1

    async def _on_pipeline_error(self, code, message, fatal) -> None:
        await self.send_error(code, message, fatal)

    async def _on_oom(self) -> None:
        """規格 §7.4：CUDA OOM → 卸載摘要模型 → 重試 ASR → 告知已降級。"""
        log.error("偵測到 CUDA OOM，卸載摘要模型以保住逐字稿")
        self.summary_degraded = True
        try:
            await self.llm.shutdown()
        except Exception as e:
            log.error("卸載摘要模型失敗：%s", e)
        await self.send_error(
            "VRAM_OOM", "VRAM 不足，摘要功能已降級；逐字稿繼續運作", fatal=False)

    # 摘要 -------------------------------------------------------------
    async def handle_mark(self, note=None) -> None:
        if self.ended:
            return
        if self.summarizing:
            # 去抖動：前一次摘要完成前的重複 mark 直接忽略（規格 §6.2）
            log.info("mark 去抖動：上一則摘要尚在生成中，忽略")
            return
        span = self.audio_pos - self.cursor
        merge = span < config.MIN_SECTION_S and bool(self.sections)
        section_seq = (self.sections[-1]["seq"] if merge else len(self.sections) + 1)
        self.summarizing = True
        await self.send({"type": "summary_pending", "section_id": section_seq})
        self.spawn(self._run_section_summary(section_seq, note, merge))

    async def _run_section_summary(self, section_seq, note, merge) -> None:
        try:
            # 讓佇列裡的音訊先轉完，摘要才不會漏掉剛講的內容
            await self.pipeline.flush()
            try:
                await asyncio.wait_for(self.pipeline.drain(), timeout=30)
            except asyncio.TimeoutError:
                log.warning("等待 ASR 佇列清空逾時，以現有 buffer 生成摘要")

            if merge:
                # 30 秒內二次按鈕：不新開段落，把 buffer 併回前一則重新生成。
                # 逐字稿從資料庫重讀（涵蓋前一則的全部內容，不只 buffer）。
                prev = self.sections[-1]
                start_s = prev["start_s"]
                transcript = (self._transcript_from_db(start_s, self.audio_pos)
                              or "".join(s["text"] for s in self.buffer).strip())
                context = self.sections[-3:-1]
                user_note = note or prev.get("user_note")
            else:
                start_s = self.cursor
                transcript = "".join(s["text"] for s in self.buffer).strip()
                context = self.sections[-2:]
                user_note = note

            if not transcript:
                await self.send_error("EMPTY_BUFFER", "這段沒有可摘要的內容", False)
                return

            # 逐字稿可能很長（整堂課只按一次按鈕時），summarize_span 會
            # 依 context 上限切塊，每塊各自成為一則 section。
            # 不切的話 llama-server 會無聲截斷——實測 32,368 字的課
            # 整段被丟掉，只留下最前面 4 分鐘。
            spans = await self.summarizer.summarize_span(
                self.course, context, transcript, start_s, self.audio_pos)
            if not spans:
                await self.send_error("SUMMARY_FAILED", "這段整理不出內容", False)
                return
            result = {"title": spans[0]["title"], "bullets": spans[0]["bullets"],
                      "summary": spans[0].get("summary", ""),
                      "groups": spans[0].get("groups") or [],
                      "degraded": any(x.get("degraded") for x in spans)}
            extra_spans = spans[1:]
            # 切成多塊時，第一則 section 只涵蓋第一塊的時間範圍；
            # 用 self.audio_pos 會讓它跟後面每一塊重疊，之後照邊界重生
            # 就會重複計算同一段逐字稿。單塊時 spans[0]["end_s"] 本來
            # 就等於 self.audio_pos。
            end_s = spans[0]["end_s"]
            if merge:
                prev = self.sections[-1]
                prev.update({"end_s": end_s, "title": result["title"],
                             "bullets": result["bullets"],
                             "summary": result.get("summary", ""),
                             "groups": result.get("groups") or [],
                             "user_note": user_note})
                await asyncio.to_thread(self._update_section_row, prev)
            else:
                row_id = await asyncio.to_thread(
                    self._insert_section_row, section_seq, start_s, end_s,
                    result["title"], result["bullets"], user_note,
                    result.get("summary", ""), result.get("groups"))
                self.sections.append({
                    "seq": section_seq, "row_id": row_id, "start_s": start_s,
                    "end_s": end_s, "title": result["title"],
                    "bullets": result["bullets"],
                    "summary": result.get("summary", ""),
                    "groups": result.get("groups") or [],
                    "user_note": user_note,
                })

            # 主段落先送，再送切出來的後續段落，前端才會照時間順序排。
            # （原本放在迴圈後面，start 取 self.sections[-1] 會拿到最後一塊
            #   的起點，主段落的時間軸整個錯位。）
            # batch：同一次按鈕產生的段落共用一個編號。逐字稿太長時一次
            # 會切出好幾段，前端要能把它們收在同一個容器裡，不然按五次
            # 就變成一串二十幾個平鋪的段落。
            await self.send({
                "type": "summary",
                "section_id": section_seq,
                "batch": section_seq,
                "batch_size": 1 + len(extra_spans),
                "batch_index": 1,
                "title": result["title"],
                "bullets": result["bullets"],
                "summary": result.get("summary", ""),
                "groups": result.get("groups") or [],
                "start": start_s,
                "end": end_s,
                "user_note": user_note,
            })

            # 切出多塊時，第一塊沿用原本的 section，其餘接在後面
            for sp in extra_spans:
                seq = len(self.sections) + 1
                row = await asyncio.to_thread(
                    self._insert_section_row, seq, sp["start_s"], sp["end_s"],
                    sp["title"], sp["bullets"], None,
                    sp.get("summary", ""), sp.get("groups"))
                self.sections.append({
                    "seq": seq, "row_id": row, "start_s": sp["start_s"],
                    "end_s": sp["end_s"], "title": sp["title"],
                    "bullets": sp["bullets"],
                    "summary": sp.get("summary", ""),
                    "groups": sp.get("groups") or [],
                    "user_note": None})
                await self.send({
                    "type": "summary", "section_id": seq,
                    "batch": section_seq,
                    "batch_size": 1 + len(extra_spans),
                    "batch_index": 2 + extra_spans.index(sp),
                    "title": sp["title"],
                    "bullets": sp["bullets"],
                    "summary": sp.get("summary", ""),
                    "groups": sp.get("groups") or [],
                    "start": sp["start_s"],
                    "end": sp["end_s"], "user_note": None})

            self.cursor = extra_spans[-1]["end_s"] if extra_spans else end_s
            self.buffer = []
            if result.get("degraded"):
                self.summary_degraded = True
        except LLMUnavailable as e:
            self.summary_degraded = True
            await self.send_error("LLM_UNAVAILABLE",
                                  "摘要服務不可用，逐字稿持續運作：%s" % e, False)
        except Exception as e:
            log.exception("段落摘要失敗")
            await self.send_error("SUMMARY_FAILED", str(e), False)
        finally:
            self.summarizing = False

    def _transcript_from_db(self, start_s, end_s):
        try:
            rows = storage.segments_between(self.id, start_s, end_s)
        except Exception:
            return None
        return "".join(r["text"] for r in rows).strip()

    def _insert_section_row(self, seq, start_s, end_s, title, bullets, note,
                            summary="", groups=None):
        try:
            return storage.insert_section(self.id, seq, start_s, end_s,
                                          title, bullets, note,
                                          summary=summary, groups=groups)
        except Exception as e:
            log.error("section 寫入資料庫失敗：%s", e)
            return -1

    def _update_section_row(self, sec):
        if sec.get("row_id", -1) < 0:
            return
        try:
            storage.update_section(sec["row_id"], sec["start_s"], sec["end_s"],
                                   sec["title"], sec["bullets"],
                                   sec.get("user_note"),
                                   summary=sec.get("summary", ""),
                                   groups=sec.get("groups"))
        except Exception as e:
            log.error("section 更新資料庫失敗：%s", e)

    # 隨時產生手抄版筆記 ----------------------------------------------
    async def handle_notes(self) -> None:
        """課堂進行中就給出可以直接手抄的筆記。

        規格沒有這個動作——它假設下課後才慢慢整理。但實際使用情境是
        「一下課就要交手寫筆記」，所以必須隨時能叫出來。

        刻意**不**切到 final 模型：那要卸載 ASR 又要 10–20 秒冷啟動，
        會中斷正在進行的逐字稿。課中用課中模型換取不中斷。
        """
        if self.notes_busy:
            return
        self.notes_busy = True
        await self.send({"type": "notes_pending"})
        try:
            # 把還沒摘要的 buffer 也納進來，否則剛講的內容不會出現在筆記裡
            sections = list(self.sections)
            if self.buffer:
                tail = "".join(s["text"] for s in self.buffer).strip()
                if tail:
                    sections = sections + [{
                        "start_s": self.cursor, "end_s": self.audio_pos,
                        "title": "（尚未標記的最新內容）",
                        "bullets": [tail[:400]], "user_note": None,
                    }]
            if not sections:
                await self.send_error("NO_CONTENT",
                                      "還沒有任何內容可以整理", False)
                return
            notes = await self.summarizer.summarize_handcopy(
                self.course, sections, model_key="inclass")
            if notes.get("degraded"):
                self.summary_degraded = True
            md = export.build_handcopy(
                self.course, {"started_at": self.started_at}, notes)
            check = self._check_notes(notes)
            await self.send({"type": "notes", "markdown": md,
                             "sections": len(sections), "check": check})
        except LLMUnavailable as e:
            await self.send_error("LLM_UNAVAILABLE",
                                  "摘要服務不可用：%s" % e, False)
        except Exception as e:
            log.exception("產生手抄版筆記失敗")
            await self.send_error("NOTES_FAILED", str(e), False)
        finally:
            self.notes_busy = False

    def _check_notes(self, notes):
        """接地檢查：摘要的每一點在逐字稿裡找不找得到依據。

        抓的是「模型自己補的知識」，不是「老師講錯」或「ASR 聽成別的詞」——
        後兩者從逐字稿本身看不出來，只能靠人回去聽錄音。
        """
        try:
            claims = []
            for t in (notes.get("topics") or []):
                claims.extend(t.get("points") or [])
            claims.extend(notes.get("exam") or [])
            if not claims:
                return None
            rows = storage.list_segments(self.id)
            transcript = "".join(r["text"] for r in rows)
            return verify_notes.summarize_check(claims, transcript, rows)
        except Exception as e:
            log.warning("接地檢查失敗（不影響筆記）：%s", e)
            return None

    # 期末總結（規格 §6.4）--------------------------------------------
    async def handle_end(self) -> None:
        if self.ended:
            return
        self.ended = True
        await self.send({"type": "final_summary_pending"})
        try:
            md = await self._run_final_summary()
            await self.send({"type": "final_summary", "session_id": self.id,
                             "markdown": md})
        except LLMUnavailable as e:
            md = await self._fallback_markdown()
            await self.send_error("LLM_UNAVAILABLE",
                                  "總結模型不可用，已輸出未經合成的筆記：%s" % e, False)
            await self.send({"type": "final_summary", "session_id": self.id,
                             "markdown": md})
        except Exception as e:
            log.exception("期末總結失敗")
            md = await self._fallback_markdown()
            await self.send_error("FINAL_SUMMARY_FAILED", str(e), False)
            await self.send({"type": "final_summary", "session_id": self.id,
                             "markdown": md})

    async def _run_final_summary(self) -> str:
        # 0. 若還有段落摘要在跑，等它做完再收尾，避免兩個摘要 task 搶 sections
        for _ in range(600):
            if not self.summarizing:
                break
            await asyncio.sleep(0.1)

        # 1. buffer 非空 → 先自動生成最後一則段落摘要
        await self.pipeline.flush()
        try:
            await asyncio.wait_for(self.pipeline.drain(), timeout=60)
        except asyncio.TimeoutError:
            log.warning("結束時等待 ASR 佇列清空逾時")
        if self.buffer:
            self.summarizing = False
            section_seq = len(self.sections) + 1
            await self._run_section_summary(section_seq, None, merge=False)

        # 2. 收掉錄音檔、卸載 ASR 釋放 VRAM（規格 §7.2）
        if self.recorder is not None:
            self.audio_path = await asyncio.to_thread(self.recorder.close)
        await self.pipeline.stop()
        await asyncio.to_thread(self.engine.unload)

        # 3./4. 切到 8B 總結模型並生成
        final = await self.summarizer.summarize_final(self.course, self.sections)
        if final.get("degraded"):
            self.summary_degraded = True

        # 5. 寫入資料庫、產出 Markdown。
        #    手抄版一併用 final 模型產一份——下課要交的就是這個。
        handcopy_md = None
        try:
            notes = await self.summarizer.summarize_handcopy(
                self.course, self.sections, model_key="final")
            handcopy_md = export.build_handcopy(
                self.course, {"started_at": self.started_at}, notes)
            await self.send({"type": "notes", "markdown": handcopy_md,
                             "sections": len(self.sections),
                             "check": self._check_notes(notes)})
        except Exception as e:
            log.warning("期末手抄版產生失敗（完整筆記不受影響）：%s", e)
        md = await self._compose_markdown(final)
        self._handcopy_md = handcopy_md
        await asyncio.to_thread(self._finish_row, md)

        # 6. 切回課中模型待命
        try:
            await self.llm.ensure("inclass")
        except LLMUnavailable as e:
            log.warning("切回課中模型失敗（下次 session 會重試）：%s", e)
        return md

    async def _compose_markdown(self, final) -> str:
        session = {"id": self.id, "started_at": self.started_at}
        return export.build_markdown(self.course, session, self.sections, final)

    async def _fallback_markdown(self) -> str:
        md = await self._compose_markdown(
            {"overview": ["（總結模型不可用，以下為各段落原始筆記）"],
             "open_questions": []})
        await asyncio.to_thread(self._finish_row, md)
        return md

    def _finish_row(self, md):
        try:
            storage.finish_session(self.id, _now_iso(), self.audio_pos, md,
                                   getattr(self, "_handcopy_md", None),
                                   self.audio_path)
        except Exception as e:
            log.error("session 收尾寫入失敗：%s", e)

    # 統計 -------------------------------------------------------------
    def stats_payload(self) -> dict:
        return {
            "type": "stats",
            "vram_used_mb": round(gpu.used_mb(), 1),
            "queue_depth": self.pipeline.queue_depth,
            "rtf": round(self.pipeline.last_rtf, 3),
            "audio_s": round(self.audio_pos, 1),
            "sections": len(self.sections),
            "summary_degraded": self.summary_degraded,
        }

    async def close(self) -> None:
        self.stop_stats()
        try:
            await self.pipeline.stop()
        except Exception as e:      # 收工失敗不該蓋掉真正的斷線原因
            log.warning("關閉 session %s 時發生例外（已忽略）：%s", self.id, e)


# ── registry：支援斷線重連（規格 §5.3）─────────────────────────────────
_sessions = {}


def get_session(session_id):
    return _sessions.get(session_id)


def all_sessions():
    return dict(_sessions)


def drop_session(session_id) -> None:
    _sessions.pop(session_id, None)


async def handle_connection(ws, course_id: str, engine, summarizer, llm,
                            vad=None) -> None:
    """WebSocket 端點主迴圈。單一連線可服務一個 session（含重連接續）。"""
    await ws.accept()
    session = None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if session is not None:
                    await session.handle_audio(msg["bytes"])
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                evt = json.loads(text)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")

            if etype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if etype == "start":
                if session is not None:
                    continue
                cid = evt.get("course_id") or course_id
                course = courses.get_course(cid)
                session = LectureSession(str(uuid.uuid4()), course, engine,
                                         summarizer, llm, vad=vad)
                session.ws = ws
                _sessions[session.id] = session
                await asyncio.to_thread(storage.create_session, session.id,
                                        course.id, session.started_at)
                session.pipeline.start()
                await session.send({"type": "session_started",
                                    "session_id": session.id,
                                    "started_at": session.started_at})
                session.start_stats()
                continue

            if etype == "resume" and evt.get("session_id"):
                # 重連：接回既有 session（規格 §5.3）
                existing = _sessions.get(evt["session_id"])
                if existing is None:
                    await ws.send_text(json.dumps({
                        "type": "error", "code": "SESSION_NOT_FOUND",
                        "message": "找不到該 session，請重新開始錄音", "fatal": True},
                        ensure_ascii=False))
                    continue
                session = existing
                session.ws = ws
                session.paused = False
                session.start_stats()
                # next_seq：接回來的客戶端要從這個編號之後開始送。
                # 頁面重開後 state.seq 會歸零，而伺服器用 seen_seqs 去重——
                # 不給的話新送的音訊全部會被當成重複丟掉，錄音看起來在跑
                # 但一個字都不會進來。
                await session.send({"type": "session_started",
                                    "session_id": session.id,
                                    "started_at": session.started_at,
                                    "next_seq": (max(session.seen_seqs) + 1
                                                 if session.seen_seqs else 0)})
                continue

            if session is None:
                continue

            if etype == "mark":
                await session.handle_mark(evt.get("note"))
            elif etype == "pause":
                session.paused = True
                await session.pipeline.flush()
            elif etype == "resume":
                session.paused = False
            elif etype == "notes":
                session.spawn(session.handle_notes())
            elif etype == "end":
                await session.handle_end()
                drop_session(session.id)
                break
    except Exception as e:
        log.exception("WebSocket 連線異常結束：%s", e)
    finally:
        if session is not None:
            session.ws = None
            if session.ended:
                session.stop_stats()
                await session.close()
                drop_session(session.id)
        try:
            await ws.close()
        except Exception:
            pass


async def _stats_loop(session) -> None:
    """每 5 秒推一次 stats（規格 §5.2）。"""
    try:
        while True:
            await asyncio.sleep(config.STATS_INTERVAL_S)
            if session.ws is None:
                return
            await session.send(session.stats_payload())
    except asyncio.CancelledError:
        return
