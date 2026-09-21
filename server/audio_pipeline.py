"""ring buffer → VAD → 切段 → ASR 佇列（規格 §4.2）。

切段邏輯（VadSegmenter）刻意寫成純同步、可離線重放，
好讓規格 §13 步驟 3「用 wav 驗證切點合理」不需要啟服務。
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from . import config

log = logging.getLogger(__name__)

FRAME = config.VAD_FRAME_SAMPLES          # 512 samples = 32ms @16kHz
SR = config.SAMPLE_RATE
FRAME_MS = FRAME * 1000 // SR


class Segment:
    __slots__ = ("start_s", "end_s", "audio", "fed_at")

    def __init__(self, start_s, end_s, audio, fed_at=0.0):
        self.start_s = start_s
        self.end_s = end_s
        self.audio = audio
        # 最後一塊音訊被餵入的時刻，用於量測端到端延遲（規格 §14.3）
        self.fed_at = fed_at

    @property
    def duration_s(self):
        return self.end_s - self.start_s

    def __repr__(self):
        return "<Segment %.2f-%.2f (%.2fs)>" % (
            self.start_s, self.end_s, self.duration_s)


class VadSegmenter:
    """逐 32ms frame 判斷語音活動並依規格 §4.2 的規則切段。

    feed() 回傳這次餵入所產出的 Segment 清單（通常是 0 或 1 個）。
    """

    def __init__(self, vad, threshold=None, min_silence_ms=None,
                 min_segment_ms=None, max_segment_ms=None,
                 pre_roll_ms=None, sample_rate=SR):
        self.vad = vad
        self.sr = sample_rate
        self.threshold = config.VAD_THRESHOLD if threshold is None else threshold
        self.min_silence_ms = (config.MIN_SILENCE_MS if min_silence_ms is None
                               else min_silence_ms)
        self.min_segment_ms = (config.MIN_SEGMENT_MS if min_segment_ms is None
                               else min_segment_ms)
        self.max_segment_ms = (config.MAX_SEGMENT_MS if max_segment_ms is None
                               else max_segment_ms)
        self.pre_roll_ms = config.PRE_ROLL_MS if pre_roll_ms is None else pre_roll_ms

        self._pre_roll_frames = max(1, int(self.pre_roll_ms * self.sr / 1000 / FRAME))
        self._tail = []               # 最近的靜音 frame，供 pre-roll 使用
        self._cur = []                # 目前這段已累積的 frame
        self._cur_start_sample = 0
        self._in_speech = False
        self._silence_ms = 0
        self._residual = np.zeros(0, dtype=np.float32)
        self._total_samples = 0       # 已消化的 sample 數（切齊 frame）
        # 短於 MIN_SEGMENT_MS 的片段暫存，併入下一段（規格 §4.2）
        self._carry = None            # (start_sample, list[frame])

    # 公用 -------------------------------------------------------------
    @property
    def position_s(self):
        return self._total_samples / float(self.sr)

    def reset(self):
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        self._tail = []
        self._cur = []
        self._in_speech = False
        self._silence_ms = 0
        self._residual = np.zeros(0, dtype=np.float32)
        self._carry = None

    def feed(self, pcm, fed_at=None):
        """餵入任意長度的 float32 音訊，回傳這次產出的 Segment 清單。"""
        if fed_at is None:
            fed_at = time.monotonic()
        out = []
        buf = np.concatenate([self._residual, np.asarray(pcm, dtype=np.float32)])
        n_frames = len(buf) // FRAME
        for i in range(n_frames):
            frame = buf[i * FRAME:(i + 1) * FRAME]
            seg = self._push_frame(frame, fed_at)
            if seg is not None:
                out.append(seg)
        self._residual = buf[n_frames * FRAME:].copy()
        return out

    def flush(self, fed_at=None):
        """串流結束／暫停時把手上的片段吐出來（不受 MIN_SEGMENT_MS 限制）。"""
        if fed_at is None:
            fed_at = time.monotonic()
        if not self._cur and self._carry is None:
            return []
        seg = self._emit(force=True, fed_at=fed_at)
        return [seg] if seg is not None else []

    # 內部 -------------------------------------------------------------
    def _push_frame(self, frame, fed_at):
        prob = float(self.vad(frame))
        is_speech = prob >= self.threshold
        self._total_samples += FRAME

        if not self._in_speech:
            if is_speech:
                # 起段：把 pre-roll 的靜音 frame 也帶進來，避免吃掉字首
                pre = self._tail[-self._pre_roll_frames:] if self._pre_roll_frames else []
                self._cur = list(pre) + [frame]
                self._cur_start_sample = (self._total_samples - FRAME
                                          - len(pre) * FRAME)
                self._in_speech = True
                self._silence_ms = 0
                self._tail = []
            else:
                self._tail.append(frame)
                if len(self._tail) > self._pre_roll_frames:
                    self._tail.pop(0)
            return None

        # 語音中
        self._cur.append(frame)
        if is_speech:
            self._silence_ms = 0
        else:
            self._silence_ms += FRAME_MS

        cur_ms = len(self._cur) * FRAME_MS
        if self._silence_ms >= self.min_silence_ms:
            return self._emit(force=False, fed_at=fed_at)
        if cur_ms >= self.max_segment_ms:
            # 強制切段：可能切在句中，規格 §12 已接受此風險
            return self._emit(force=True, fed_at=fed_at, keep_speaking=True)
        return None

    def _emit(self, force, fed_at, keep_speaking=False):
        frames = self._cur
        start_sample = self._cur_start_sample
        self._cur = []
        self._in_speech = keep_speaking
        self._silence_ms = 0
        if keep_speaking:
            self._cur_start_sample = start_sample + len(frames) * FRAME
        else:
            self._tail = []

        if self._carry is not None:
            carry_start, carry_frames = self._carry
            frames = carry_frames + frames
            start_sample = carry_start
            self._carry = None

        if not frames:
            return None

        dur_ms = len(frames) * FRAME_MS
        if dur_ms < self.min_segment_ms and not force:
            # 太短：併入下一段，不單獨送 ASR
            self._carry = (start_sample, frames)
            return None

        audio = np.concatenate(frames).astype(np.float32)
        end_sample = start_sample + len(audio)
        return Segment(start_sample / float(self.sr),
                       end_sample / float(self.sr), audio, fed_at)


class AudioPipeline:
    """把切好的段推入 asyncio.Queue，由單一 worker 序列化送進 ASR。

    規格 §4.2：不要開多個 worker 並行推論——8GB VRAM 承受不住，
    而且會打亂逐字稿順序。
    """

    def __init__(self, engine, initial_prompt="", on_result=None,
                 on_error=None, on_oom=None, vad=None, max_queue=64):
        self.engine = engine
        self.initial_prompt = initial_prompt
        self.on_result = on_result
        self.on_error = on_error
        # 規格 §7.4：CUDA OOM 時的降級回呼（卸載摘要模型後重試 ASR）
        self.on_oom = on_oom
        self.segmenter = VadSegmenter(vad if vad is not None else default_vad())
        self.queue = asyncio.Queue(maxsize=max_queue)
        self._task = None
        self._deferred = []            # VRAM 不足時的延遲佇列（規格 §7.4）
        self.last_rtf = 0.0
        self.processed = 0
        self.dropped = 0
        self.oom_events = 0

    @property
    def queue_depth(self):
        return self.queue.qsize() + len(self._deferred)

    def start(self):
        if self._task is None:
            self.segmenter.reset()
            self._task = asyncio.create_task(self._worker(), name="asr-worker")

    async def stop(self):
        """收工。這裡在 WebSocket 的 finally 裡被呼叫，
        不論發生什麼都不能往外拋——否則會蓋掉真正的斷線原因。"""
        task = self._task
        self._task = None
        if task is None:
            return
        try:
            await self.queue.put(None)
            await asyncio.wait_for(task, timeout=30)
        except asyncio.TimeoutError:
            task.cancel()
        except Exception as e:
            log.warning("停止 ASR worker 時發生例外（已忽略）：%s", e)
            task.cancel()

    async def feed(self, pcm, fed_at=None):
        segs = await asyncio.to_thread(self.segmenter.feed, pcm, fed_at)
        for s in segs:
            await self._enqueue(s)

    async def flush(self):
        segs = await asyncio.to_thread(self.segmenter.flush)
        for s in segs:
            await self._enqueue(s)

    async def drain(self):
        """等待佇列中所有段落處理完成。"""
        await self.queue.join()

    async def _enqueue(self, seg):
        try:
            self.queue.put_nowait(seg)
        except asyncio.QueueFull:
            self.dropped += 1
            log.error("ASR 佇列已滿，丟棄片段 %s（累計 %d）", seg, self.dropped)
            await self._report_error(
                "ASR_QUEUE_FULL", "ASR 處理速度跟不上，已丟棄一段音訊")

    async def _worker(self):
        from .asr import ASRVramLow
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            try:
                await self._process(item, ASRVramLow)
            except Exception as e:   # 單段失敗不能拖垮整個 session
                log.exception("ASR 處理片段失敗：%s", e)
                await self._report_error("ASR_FAILED", str(e))
            finally:
                self.queue.task_done()

    async def _process(self, seg, ASRVramLow):
        # 先把先前因 VRAM 不足延後的段補上
        if self._deferred:
            pending, self._deferred = self._deferred, []
            for d in pending:
                await self._transcribe(d, ASRVramLow, allow_defer=False)
        await self._transcribe(seg, ASRVramLow, allow_defer=True)

    async def _transcribe(self, seg, ASRVramLow, allow_defer):
        try:
            res = await asyncio.to_thread(
                self.engine.transcribe_joined, seg.audio, self.initial_prompt)
        except ASRVramLow as e:
            if allow_defer:
                self._deferred.append(seg)
                log.warning("VRAM 餘量不足，片段 %s 推入延遲佇列：%s", seg, e)
                await self._report_error("VRAM_LOW", str(e), fatal=False)
                return
            raise
        except Exception as e:
            if not is_oom(e):
                raise
            # 規格 §7.4：卸載摘要模型 → 重試 ASR → 回送降級通知。
            # 逐字稿是核心功能，任何資源衝突下都犧牲摘要保逐字稿。
            self.oom_events += 1
            log.error("ASR 推論遇到 CUDA OOM，卸載摘要模型後重試：%s", e)
            if self.on_oom is not None:
                await maybe_await(self.on_oom)
            await asyncio.sleep(1.0)
            res = await asyncio.to_thread(
                self.engine.transcribe_joined, seg.audio, self.initial_prompt)
        self.processed += 1
        if res is None:
            return
        self.last_rtf = res.get("rtf", 0.0)
        if self.on_result is not None:
            await maybe_await(self.on_result, seg, res)

    async def _report_error(self, code, message, fatal=False):
        if self.on_error is not None:
            await maybe_await(self.on_error, code, message, fatal)


_OOM_MARKERS = ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed",
                "cudnn_status_alloc_failed", "failed to allocate")


def is_oom(exc) -> bool:
    """CTranslate2 的 OOM 是普通 RuntimeError，只能靠訊息辨識。"""
    return any(m in str(exc).lower() for m in _OOM_MARKERS)


async def maybe_await(fn, *args):
    r = fn(*args)
    if asyncio.iscoroutine(r):
        await r


_vad_singleton = None


def default_vad():
    """VAD 模型全域共用一份（載入成本高），狀態由各 segmenter reset。"""
    global _vad_singleton
    if _vad_singleton is None:
        from .vad import make_vad
        _vad_singleton = make_vad()
    return _vad_singleton
