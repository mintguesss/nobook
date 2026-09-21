"""selftest 用的 WebSocket 端到端測試：真的跑 FastAPI 路由與 ws_session 狀態機。

用假的 ASR 引擎與假的摘要服務，所以不需要模型、GPU 或 llama-server。
"""
from __future__ import annotations

import contextlib
import struct
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, WebSocket

from server import config, storage, ws_session
from server.summarizer import Summarizer

HEADER = struct.Struct("<II")
CHUNK = 4000        # 250ms @16kHz


class FakeEngine:
    def __init__(self):
        self.n = 0
        self.loaded = True
        self.unloaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False
        self.unloaded = True

    def transcribe_joined(self, audio, prompt="", *a, **k):
        self.n += 1
        return {"text": "逐字稿第%d段" % self.n, "avg_logprob": -0.3, "rtf": 0.12}


class FakeLLM:
    base_url = "http://127.0.0.1:0"
    current = "inclass"
    degraded = False

    async def ensure(self, key):
        self.current = key

    async def shutdown(self):
        pass

    async def health(self):
        return True


class FakeSummarizer:
    # summarize_span 直接借用真的實作，切塊邏輯才是真的被測到，
    # 而不是被替身繞過去。它只用到 summarize_section 與 manager.current_ctx。
    summarize_span = Summarizer.summarize_span
    _max_prompt_chars = Summarizer._max_prompt_chars

    def __init__(self):
        self.n = 0
        self.last_transcript = None
        self.manager = None      # current_ctx 取不到就用預設 4096

    async def summarize_section(self, course, context, transcript):
        self.n += 1
        self.last_transcript = transcript
        return {"title": "段落%d" % self.n,
                "bullets": ["要點%d-1" % self.n, "要點%d-2" % self.n],
                "degraded": False}

    async def summarize_final(self, course, sections):
        return {"overview": ["這堂課的整體概述"], "open_questions": ["待釐清一"],
                "degraded": False}

    async def summarize_handcopy(self, course, sections, model_key="inclass"):
        # 沒有這個方法的話，ws_session 會吞掉 AttributeError，
        # 手抄版那條路在端到端測試裡等於完全沒跑到。
        return {"topics": [{"title": "手抄主題", "points": ["手抄要點"]}],
                "exam": [], "degraded": False}


class Speech:
    """每 2 秒語音、1 秒靜音，確保切得出段。"""

    def __init__(self):
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, frame):
        self.i += 1
        cycle = int(3000 / config.VAD_FRAME_MS)
        return 1.0 if (self.i % cycle) < int(2000 / config.VAD_FRAME_MS) else 0.0


def build_app(engine, summarizer, llm, vad):
    app = FastAPI()

    @app.websocket("/ws/session")
    async def ws(ws: WebSocket, course_id: str = Query("ml-2026")):
        await ws_session.handle_connection(ws, course_id, engine, summarizer,
                                           llm, vad=vad)
    return app


def audio_frame(seq, seconds=0.25, loud=True):
    n = int(seconds * config.SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / config.SAMPLE_RATE
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)) if loud else np.zeros(n, np.float32)
    pcm = np.clip(sig * 32768, -32768, 32767).astype("<i2")
    return HEADER.pack(seq, len(pcm)) + pcm.tobytes()


def run(gate):
    from fastapi.testclient import TestClient

    engine = FakeEngine()
    summ = FakeSummarizer()
    llm = FakeLLM()
    app = build_app(engine, summ, llm, Speech())

    tmp = tempfile.TemporaryDirectory()
    storage.init(Path(tmp.name) / "ws.db")
    # TestClient 必須進 context：不進的話每次 websocket_connect 會各自開一個
    # event loop，而 session 持有的 asyncio.Queue／Task 是綁定 loop 的，
    # 重連測試就會炸在 "attached to a different loop"。進 context 後整個
    # client 共用一個 portal，才符合正式環境 uvicorn 單一 event loop 的行為。
    stack = contextlib.ExitStack()
    try:
        client = stack.enter_context(TestClient(app))
        with client.websocket_connect("/ws/session?course_id=ml-2026") as ws:
            ws.send_json({"type": "start", "course_id": "ml-2026",
                          "client_ts": 0})
            started = ws.receive_json()
            gate.check(started["type"] == "session_started" and started["session_id"],
                       "WS：start 事件建立 session 並回送 session_started",
                       started.get("session_id", ""))
            sid = started["session_id"]

            gate.check(storage.get_session(sid) is not None,
                       "WS：session 已寫入資料庫")

            ws.send_json({"type": "ping"})
            gate.check(ws.receive_json()["type"] == "pong", "WS：ping/pong 保活")

            # 送 40 秒音訊（> MIN_SECTION_S 30 秒），收 segment 事件
            seq = 0
            for _ in range(160):
                ws.send_bytes(audio_frame(seq))
                seq += 1
            ws.send_json({"type": "mark", "note": "這題會考"})

            events = collect(ws, until={"summary"}, limit=400)
            segs = [e for e in events if e["type"] == "segment"]
            gate.check(len(segs) > 0, "WS：binary 音訊 → segment 事件下推",
                       "%d 段，首段 %r" % (len(segs), segs[0]["text"] if segs else ""))
            gate.check(any(e["type"] == "summary_pending" for e in events),
                       "WS：mark 立即回送 summary_pending（前端顯示 loading）")
            summary = [e for e in events if e["type"] == "summary"][0]
            gate.check(summary["section_id"] == 1
                       and summary["user_note"] == "這題會考",
                       "WS：summary 事件帶 section_id 與使用者標註",
                       "section_id=%s note=%r" % (summary["section_id"],
                                                  summary["user_note"]))
            gate.check(summ.last_transcript and "逐字稿第" in summ.last_transcript,
                       "WS：摘要 prompt 收到 buffer 全文",
                       repr(summ.last_transcript[:40]))

            # 去抖動：連按 3 次 -----------------------------------------
            for _ in range(160):
                ws.send_bytes(audio_frame(seq))
                seq += 1
            n_before = summ.n
            for _ in range(3):
                ws.send_json({"type": "mark"})
            collect(ws, until={"summary"}, limit=400)
            gate.check(summ.n == n_before + 1,
                       "WS：100ms 內 3 次 mark 只產生 1 則 section（去抖動）",
                       "summarize_section 呼叫 %d 次" % (summ.n - n_before))

            # 短 buffer 合併：只送 12 秒就再按 -------------------------
            for _ in range(48):
                ws.send_bytes(audio_frame(seq))
                seq += 1
            ws.send_json({"type": "mark", "note": "補充"})
            evts = collect(ws, until={"summary"}, limit=400)
            merged = [e for e in evts if e["type"] == "summary"][0]
            gate.check(merged["section_id"] == 2,
                       "WS：30 秒內二次按鈕併入前一則，section_id 不遞增",
                       "section_id=%s" % merged["section_id"])
            gate.check(storage.count_sections(sid) == 2,
                       "WS：資料庫 sections 列數不增加",
                       "%d 列" % storage.count_sections(sid))

            # 接回既有 session 時要拿到 next_seq ----------------------
            # 沒有的話頁面重開後 seq 從 0 起算，全部撞上 seen_seqs 被丟掉
            with client.websocket_connect(
                    "/ws/session?course_id=ml-2026") as ws2:
                ws2.send_json({"type": "resume", "session_id": sid})
                evts = collect(ws2, until={"session_started"}, limit=50)
                st = [e for e in evts if e["type"] == "session_started"]
                gate.check(bool(st) and st[0].get("next_seq", 0) >= seq,
                           "WS：resume 回傳 next_seq，接在已收過的編號之後",
                           "next_seq=%s 已送到 seq=%d"
                           % (st[0].get("next_seq") if st else None, seq))

            # pause / resume ------------------------------------------
            # 先把前一階段還在 ASR 佇列裡的音訊排乾。不排的話那些結果會
            # 在 pause 之後才回來，被誤判成「pause 期間產生的逐字稿」，
            # 這個檢查就變成看運氣的。
            ws.send_json({"type": "ping"})
            collect(ws, until={"pong"}, limit=200)
            ws.send_json({"type": "pause"})
            for _ in range(20):
                ws.send_bytes(audio_frame(seq))
                seq += 1
            ws.send_json({"type": "resume"})
            ws.send_json({"type": "ping"})
            drained = collect(ws, until={"pong"}, limit=200)
            gate.check(not any(e["type"] == "segment" for e in drained),
                       "WS：pause 期間不產生逐字稿")

        # 斷線重連（規格 §5.3）：上面的 with 區塊結束＝連線中斷 ------
        gate.check(sid in ws_session.all_sessions(),
                   "WS：連線中斷後 session 保留在 registry 等待重連")
        held = ws_session.get_session(sid)
        gate.check(held is not None and held.ws is None,
                   "WS：中斷時只清掉 ws，session 狀態不動")
        sections_before = len(held.sections) if held else -1
        pos_before = held.audio_pos if held else -1

        with client.websocket_connect("/ws/session?course_id=ml-2026") as ws:
            ws.send_json({"type": "resume", "session_id": sid})
            back = ws.receive_json()
            gate.check(back["type"] == "session_started"
                       and back["session_id"] == sid,
                       "WS：resume 帶 session_id 接回同一個 session",
                       "session_id=%s" % back.get("session_id"))
            gate.check(len(held.sections) == sections_before,
                       "WS：重連後既有摘要不遺失",
                       "%d 則" % len(held.sections))

            # 補送斷線期間的 buffer，其中前 20 筆刻意重送（測 seq 去重）
            backlog = [audio_frame(s) for s in range(seq, seq + 120)]
            seq += 120
            for f in backlog[:20] + backlog:
                ws.send_bytes(f)
            ws.send_json({"type": "ping"})
            collect(ws, until={"pong"}, limit=400)
            gained = held.audio_pos - pos_before
            gate.check(abs(gained - 30.0) < 0.6,
                       "WS：補送的音訊被 seq 去重，只算一次",
                       "重連後累計 +%.1fs（送了 140 筆、其中 20 筆重複，"
                       "應只算 120 筆＝30.0s）" % gained)

            # 結束並取得完整筆記 ---------------------------------------
            ws.send_json({"type": "end"})
            fin = collect(ws, until={"final_summary"}, limit=400)
            md = [e for e in fin if e["type"] == "final_summary"][0]["markdown"]
            gate.check(any(e["type"] == "final_summary_pending" for e in fin),
                       "WS：end 先回送 final_summary_pending")
            gate.check("# 機器學習" in md and "## 本堂重點" in md
                       and "### 段落" in md,
                       "WS：final_summary 回送完整 Markdown",
                       md.splitlines()[0] if md else "")
            gate.check("> 使用者標註：這題會考" in md,
                       "WS：使用者標註出現在最終筆記")
            gate.check("這堂課的整體概述" in md and "待釐清一" in md,
                       "WS：走總結模型的正常路徑（非降級輸出）")
            gate.check(engine.unloaded,
                       "WS：期末總結前卸載 ASR 釋放 VRAM（規格 §6.4 步驟 2）")

        row = storage.get_session(sid)
        gate.check(row and row["final_md"] and row["ended_at"],
                   "WS：結束後 final_md 與 ended_at 已落地")
        gate.check(sid not in ws_session.all_sessions(),
                   "WS：session 結束後從 registry 移除，不留殘留狀態")
    finally:
        stack.close()
        storage.close()
        tmp.cleanup()


def collect(ws, until, limit=200):
    """收事件直到看到 until 中的型別（或達 limit）。"""
    out = []
    for _ in range(limit):
        msg = ws.receive_json()
        out.append(msg)
        if msg.get("type") in until:
            break
    return out
