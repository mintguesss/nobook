"""閘門 G4 — 韌性與體驗，混沌測試（規格 §11 M4、§14.6）。

每個案例的模式都是：建立正常串流 → 注入故障 → 斷言逐字稿仍持續產出。

預設用假的 ASR 引擎（故障注入與逐字稿續行邏輯不需要真模型，且可重複跑）；
加 --real-asr 改用 bench.json 指定的真模型。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

from _common import Gate, header, load_bench, merge_bench

from server import config, courses, storage, ws_session
from server.asr import ASRVramLow
from server.llm_manager import LLMUnavailable
from server.summarizer import Summarizer

CHUNK_SAMPLES = int(0.25 * config.SAMPLE_RATE)


class FakeEngine:
    """可注入故障的 ASR 替身。永遠回傳可辨識的文字，方便斷言逐字稿續行。"""

    def __init__(self):
        self.calls = 0
        self.fail_next = None       # 下一次呼叫要拋的例外
        self.loaded = True

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def transcribe_joined(self, audio, initial_prompt="", *a, **k):
        self.calls += 1
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        return {"text": "第 %d 段逐字稿內容" % self.calls,
                "avg_logprob": -0.3, "rtf": 0.1}


class FakeWs:
    """收集下行事件的假 WebSocket。ws=None 即代表斷線。"""

    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        import json
        self.sent.append(json.loads(text))

    def types(self):
        return [e.get("type") for e in self.sent]

    def errors(self):
        return [e for e in self.sent if e.get("type") == "error"]


class FakeLLM:
    """llm_manager 替身，可被 _force_kill。"""

    def __init__(self):
        self.dead = False
        self.base_url = "http://127.0.0.1:0"
        self.current = "inclass"
        self.degraded = False

    async def ensure(self, key):
        if self.dead:
            raise LLMUnavailable("llama-server 已被 _force_kill()")
        self.current = key

    async def shutdown(self):
        self.dead = True

    async def health(self):
        return not self.dead

    def _force_kill(self):
        self.dead = True
        self.degraded = True


class FakeSummarizer:
    # 見 _wstest.py：切塊邏輯用真的，不要在替身裡再寫一份
    summarize_span = Summarizer.summarize_span
    _max_prompt_chars = Summarizer._max_prompt_chars

    def __init__(self, llm):
        self.manager = llm

    async def summarize_section(self, course, context, transcript):
        await self.manager.ensure("inclass")
        return {"title": "測試段落", "bullets": ["要點一", "要點二"], "degraded": False}

    async def summarize_final(self, course, sections):
        await self.manager.ensure("final")
        return {"overview": ["整體概述"], "open_questions": [], "degraded": False}


def make_audio(seconds=6.0):
    n = int(seconds * config.SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / config.SAMPLE_RATE
    # 交替的語音／靜音，讓 VAD 切得出段落
    env = (np.sin(2 * np.pi * 0.4 * t) > -0.2).astype(np.float32)
    return (0.25 * np.sin(2 * np.pi * 200 * t) * env).astype(np.float32)


class AlwaysSpeech:
    """固定回傳 1.0 的 VAD 替身，讓切段行為可預測。"""

    def __init__(self, pattern_s=2.0):
        self.pattern_s = pattern_s
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, frame):
        self.i += 1
        # 每 2 秒語音、0.8 秒靜音，確保會觸發 MIN_SILENCE_MS 切段
        cycle = int((self.pattern_s + 0.8) * 1000 / config.VAD_FRAME_MS)
        speech = int(self.pattern_s * 1000 / config.VAD_FRAME_MS)
        return 1.0 if (self.i % cycle) < speech else 0.0


async def make_session(engine=None, llm=None):
    storage.init(config.DATA_DIR / "chaos.db")
    course = courses.get_course("ml-2026")
    engine = engine or FakeEngine()
    llm = llm or FakeLLM()
    s = ws_session.LectureSession("chaos-" + str(int(time.time() * 1000)),
                                  course, engine, FakeSummarizer(llm), llm,
                                  vad=AlwaysSpeech())
    s.ws = FakeWs()
    storage.create_session(s.id, course.id, s.started_at)
    s.pipeline.start()
    return s


async def feed(session, seconds, audio):
    pos = 0
    from server.ws_session import HEADER
    seq = getattr(session, "_test_seq", 0)
    for _ in range(int(seconds / 0.25)):
        if pos + CHUNK_SAMPLES > len(audio):
            pos = 0
        chunk = audio[pos:pos + CHUNK_SAMPLES]
        pos += CHUNK_SAMPLES
        pcm16 = np.clip(chunk * 32768, -32768, 32767).astype("<i2")
        await session.handle_audio(HEADER.pack(seq, len(pcm16)) + pcm16.tobytes())
        seq += 1
        await asyncio.sleep(0)
    session._test_seq = seq


def transcript_count(session):
    return sum(1 for e in session.ws.sent if e.get("type") == "segment")


# ── 案例 ───────────────────────────────────────────────────────────────
async def case_kill_llm(gate, engine=None):
    """串流中途 kill llama-server：逐字稿不中斷，回送降級 error 事件。"""
    llm = FakeLLM()
    s = await make_session(engine=engine, llm=llm)
    audio = make_audio()
    try:
        await feed(s, 8, audio)
        await s.pipeline.drain()
        before = transcript_count(s)

        llm._force_kill()                       # 注入故障（規格 §14.6）
        await s.handle_mark("kill 測試")
        for _ in range(100):
            if not s.summarizing:
                break
            await asyncio.sleep(0.05)

        await feed(s, 8, audio)
        await s.pipeline.drain()
        after = transcript_count(s)

        gate.check(after > before, "kill llama-server 後逐字稿不中斷",
                   "故障前 %d 段 → 故障後 %d 段" % (before, after))
        codes = [e["code"] for e in s.ws.errors()]
        gate.check("LLM_UNAVAILABLE" in codes, "回送降級 error 事件",
                   "收到 error：%s" % (codes or "無"))
    finally:
        await s.pipeline.stop()


async def case_cuda_oom(gate, real_vram_probe):
    """人為觸發 CUDA OOM：不崩潰，摘要降級，逐字稿續行。"""
    engine = FakeEngine()
    llm = FakeLLM()
    s = await make_session(engine=engine, llm=llm)
    audio = make_audio()
    try:
        await feed(s, 6, audio)
        await s.pipeline.drain()
        before = transcript_count(s)

        # 注入一次 CUDA OOM；規格 §7.4 要求卸載摘要模型 → 重試 ASR → 續行
        engine.fail_next = RuntimeError(
            "CUDA failed with error out of memory")
        await feed(s, 8, audio)
        await s.pipeline.drain()
        after = transcript_count(s)

        gate.check(after > before, "CUDA OOM 後逐字稿續行（重試成功）",
                   "OOM 前 %d 段 → OOM 後 %d 段，pipeline 記錄 %d 次 OOM"
                   % (before, after, s.pipeline.oom_events))
        gate.check(s.pipeline.oom_events == 1 and llm.dead,
                   "OOM 時卸載摘要模型並標記降級",
                   "oom_events=%d，llm.dead=%s，summary_degraded=%s"
                   % (s.pipeline.oom_events, llm.dead, s.summary_degraded))
        codes = [e["code"] for e in s.ws.errors()]
        gate.check("VRAM_OOM" in codes, "回送 VRAM_OOM error 事件",
                   "收到 error：%s" % (codes or "無"))

        # VRAM 餘量不足時的延遲佇列（規格 §7.4）
        engine.fail_next = ASRVramLow("VRAM 餘量 100MB 低於門檻 400MB")
        n_before = transcript_count(s)
        await feed(s, 8, audio)
        await s.pipeline.drain()
        gate.check(transcript_count(s) > n_before,
                   "VRAM 餘量不足時推入延遲佇列而非炸掉 session",
                   "延後的段落已在後續補上，逐字稿共 %d 段" % transcript_count(s))
    finally:
        await s.pipeline.stop()
    if real_vram_probe:
        await probe_real_oom(gate)


async def probe_real_oom(gate):
    """選用：真的把 VRAM 吃滿再釋放，確認服務不崩（需 torch+CUDA）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            gate.info("--real-oom：torch 沒有 CUDA 支援，跳過真實 VRAM 耗盡測試")
            return
    except ImportError:
        gate.info("--real-oom：未安裝 torch，跳過真實 VRAM 耗盡測試")
        return
    blobs = []
    try:
        while True:
            blobs.append(torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32,
                                     device="cuda"))
    except RuntimeError as e:
        gate.check("out of memory" in str(e).lower(), "真實 VRAM 耗盡可被攔截為 OOM",
                   str(e)[:100])
    finally:
        blobs.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


async def case_network_drop(gate, engine=None):
    """網路中斷 30 秒再恢復：音訊補送成功，逐字稿無缺口。"""
    s = await make_session(engine=engine)
    audio = make_audio()
    try:
        await feed(s, 6, audio)
        await s.pipeline.drain()
        before = transcript_count(s)

        # 斷線：前端繼續錄音並在本地 buffer 累積（規格 §5.3）
        s.ws = None
        from server.ws_session import HEADER
        buffered = []
        seq = s._test_seq
        pos = 0
        for _ in range(int(30 / 0.25)):
            if pos + CHUNK_SAMPLES > len(audio):
                pos = 0
            pcm16 = np.clip(audio[pos:pos + CHUNK_SAMPLES] * 32768,
                            -32768, 32767).astype("<i2")
            pos += CHUNK_SAMPLES
            buffered.append(HEADER.pack(seq, len(pcm16)) + pcm16.tobytes())
            seq += 1
        s._test_seq = seq

        # 重連並補送 buffer，其中前 20 筆刻意重送（測 seq 去重）
        s.ws = FakeWs()
        resend = buffered[:20] + buffered
        for frame in resend:
            await s.handle_audio(frame)
        await s.pipeline.flush()
        await s.pipeline.drain()

        after = transcript_count(s)
        expected_s = 30.0
        got_s = s.audio_pos
        gate.check(after > 0, "重連後逐字稿續行", "補送後產出 %d 段" % after)
        gate.check(abs(got_s - (6.0 + expected_s)) < 1.0,
                   "音訊補送無缺漏、無重複（重送 20 筆被 seq 去重）",
                   "session 累計音訊 %.1fs（預期約 %.1fs）" % (got_s, 6.0 + expected_s))
        gate.check(len(s.seen_seqs) == s._test_seq,
                   "seq 連續性正確", "已見 %d 個 seq，最大 seq %d"
                   % (len(s.seen_seqs), s._test_seq - 1))
    finally:
        await s.pipeline.stop()


async def case_disk_failure(gate, engine=None):
    """磁碟寫入失敗：不影響即時串流，僅記錄錯誤。"""
    s = await make_session(engine=engine)
    audio = make_audio()
    original = storage.insert_segment
    try:
        await feed(s, 6, audio)
        await s.pipeline.drain()
        before = transcript_count(s)

        def boom(*a, **k):
            raise OSError("模擬磁碟寫入失敗：disk full")

        storage.insert_segment = boom
        await feed(s, 8, audio)
        await s.pipeline.drain()
        after = transcript_count(s)
        gate.check(after > before, "磁碟寫入失敗時即時串流不受影響",
                   "故障前 %d 段 → 故障後 %d 段（segment 事件仍持續下推）"
                   % (before, after))
        ids = [e["id"] for e in s.ws.sent if e.get("type") == "segment"][-1:]
        gate.check(ids == [-1], "寫入失敗的 segment 以 id=-1 標記，不中斷流程",
                   "最後一段 id=%s" % (ids or "無"))
    finally:
        storage.insert_segment = original
        await s.pipeline.stop()


def build_engine(args):
    """--real-asr 時改用 bench.json 指定的真模型。

    CUDA OOM 案例一定要用 FakeEngine（故障必須可注入），其餘案例可用真模型。
    """
    if not args.real_asr:
        return None
    bench = load_bench()
    path = (bench.get("asr") or {}).get("model")
    if not path:
        print("  --real-asr 但 bench.json 沒有 asr.model，退回 FakeEngine")
        return None
    from server.asr import ASREngine
    return ASREngine(path, compute_type=(bench.get("asr") or {}).get(
        "compute_type", "int8_float16"))


async def main_async(args) -> int:
    header("verify_m4.py — 閘門 G4：韌性與體驗（混沌測試，規格 §11 M4）")
    gate = Gate("G4")
    engine = build_engine(args)
    gate.info("ASR 引擎：%s"
              % ("真實模型（OOM 案例仍用 FakeEngine 注入故障）" if engine
                 else "FakeEngine（故障注入用）"))
    await case_kill_llm(gate, engine)
    await case_cuda_oom(gate, args.real_oom)
    await case_network_drop(gate, engine)
    await case_disk_failure(gate, engine)
    merge_bench({"m4": {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "passed": len(gate.failed) == 0}})
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-asr", action="store_true",
                    help="改用 bench.json 指定的真實 ASR 模型")
    ap.add_argument("--real-oom", action="store_true",
                    help="額外做一次真實 VRAM 耗盡測試（需 torch+CUDA）")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
