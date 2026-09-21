"""閘門 G2 — 按鈕摘要（規格 §11 M2、§14.4）。

以 Python WebSocket 客戶端連上真實服務，餵入音訊並在指定時間點送 mark。
需要服務已啟動：
    python -m server.main
或加 --start-server 讓本腳本自己拉起來。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _common import (Gate, ROOT, find_fixture, header, load_bench, merge_bench,
                     percentile)

from server import config, courses, gpu

HEADER = struct.Struct("<II")
CHUNK_S = 0.25
CHUNK_SAMPLES = int(CHUNK_S * config.SAMPLE_RATE)

SAMPLE_TRANSCRIPT = (
    "那我們今天要講的是 gradient descent 的收斂性。首先在凸函數的情況下，"
    "只要 learning rate 選得夠小，它一定會收斂到全域最小值。可是深度學習的 "
    "loss surface 並不是凸的，所以我們只能保證收斂到局部最小值或是鞍點。"
    "接下來講 regularization，L2 regularization 其實就是在 loss 後面加一個 "
    "weight decay 項，它可以有效抑制 overfitting。"
)


class WsClient:
    """最小 WebSocket 客戶端，收集下行事件。"""

    def __init__(self, url):
        self.url = url
        self.ws = None
        self.events = []
        self.stats = []
        self._recv_task = None
        self._seq = 0
        self.session_id = None

    async def __aenter__(self):
        import websockets
        self.ws = await websockets.connect(self.url, max_size=32 * 1024 * 1024)
        self._recv_task = asyncio.create_task(self._recv_loop())
        return self

    async def __aexit__(self, *exc):
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws:
            await self.ws.close()
        return False

    async def _recv_loop(self):
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    continue
                evt = json.loads(msg)
                evt["_t"] = time.monotonic()
                if evt.get("type") == "stats":
                    self.stats.append(evt)
                else:
                    self.events.append(evt)
                if evt.get("type") == "session_started":
                    self.session_id = evt["session_id"]
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def send(self, obj):
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def send_audio(self, pcm_float):
        pcm16 = np.clip(pcm_float * 32768.0, -32768, 32767).astype("<i2")
        frame = HEADER.pack(self._seq, len(pcm16)) + pcm16.tobytes()
        self._seq += 1
        await self.ws.send(frame)

    async def wait_for(self, etype, timeout=60.0, since=0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for e in self.events[since:]:
                if e.get("type") == etype:
                    return e
            await asyncio.sleep(0.05)
        return None

    def count(self, etype):
        return sum(1 for e in self.events if e.get("type") == etype)


def load_wav(path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        idx = np.linspace(0, len(data) - 1, int(len(data) * config.SAMPLE_RATE / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype("float32")
    return np.ascontiguousarray(data, dtype=np.float32)


async def feed_seconds(client, audio, pos, seconds, realtime=True):
    """按真實節奏送 seconds 秒音訊，回傳新的 pos。"""
    n_chunks = int(seconds / CHUNK_S)
    t0 = time.monotonic()
    for i in range(n_chunks):
        if pos + CHUNK_SAMPLES > len(audio):
            pos = 0
        await client.send_audio(audio[pos:pos + CHUNK_SAMPLES])
        pos += CHUNK_SAMPLES
        if realtime:
            delay = t0 + (i + 1) * CHUNK_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
    return pos


async def wait_server(base_http, timeout=180.0):
    import httpx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(base_http + "/api/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


class AttachedLLM:
    """只接上既有的 llama-server，永遠不 spawn 也不 kill。

    給 Summarizer 用的最小介面：ensure() 是 no-op，因為模型已經由
    主服務載好了。
    """

    def __init__(self, base_url=None):
        self.base_url = base_url or config.LLAMA_BASE_URL
        self.current = "inclass"
        self.degraded = False

    async def ensure(self, model_key):
        return

    async def shutdown(self):
        return


# ── 各項檢查 ───────────────────────────────────────────────────────────
async def check_json_rate(gate, n=20):
    """直接呼叫 summarizer 20 次，統計 JSON 解析成功率（規格 §14.4）。"""
    from server.llm_manager import LLMUnavailable
    from server.summarizer import Summarizer

    # 接上主服務已經起好的 llama-server，不要自己再起一個：
    # 同一個 port 會撞，而且 8GB VRAM 塞不下兩份模型。
    s = Summarizer(AttachedLLM())
    course = courses.get_course("ml-2026")
    ok = 0
    degraded = 0
    for _ in range(n):
        try:
            res = await s.summarize_section(course, [], SAMPLE_TRANSCRIPT)
        except LLMUnavailable as e:
            gate.check(False, "LLM 輸出 JSON 解析成功率 >= 95%",
                       "llama-server 不可用：%s" % e)
            return
        if res.get("degraded"):
            degraded += 1
        else:
            ok += 1
    rate = ok / float(n)
    gate.check(rate >= 0.95, "LLM 輸出 JSON 解析成功率 >= 95%（20 次）",
               "成功 %d／%d（%.0f%%），降級路徑觸發 %d 次" % (ok, n, rate * 100, degraded))
    await check_degrade_path(gate, course)
    merge_bench({"m2": {"json_parse_rate": round(rate, 3)}})


async def check_degrade_path(gate, course):
    """主動餵壞回應，確認降級路徑真的會走（規格 §6.3）。

    不能靠「這 20 次剛好有幾次失敗」來驗：模型表現好的時候一次都不會失敗，
    斷言就變成 0 == 0，看起來 PASS 其實什麼都沒測到。
    """
    from server.summarizer import Summarizer

    s = Summarizer(AttachedLLM())
    calls = []

    async def broken_chat(system, user, max_tokens, temperature, schema=None):
        calls.append(user)
        return "抱歉，我沒辦法用 JSON 回答，這裡是一段純文字說明。"

    s._chat = broken_chat
    try:
        res = await s.summarize_section(course, [], SAMPLE_TRANSCRIPT)
    except Exception as e:
        gate.check(False, "解析失敗時降級為單一 bullet，不拋例外",
                   "反而拋了 %r" % (e,))
        return
    ok = (res.get("degraded") is True and bool(res.get("title"))
          and len(res.get("bullets") or []) >= 1)
    gate.check(ok, "解析失敗時降級為單一 bullet，不拋例外",
               "degraded=%s，title=%r，bullet=%r"
               % (res.get("degraded"), res.get("title"),
                  (res.get("bullets") or [None])[0]))
    gate.check(len(calls) == 2, "解析失敗會先重試一次再降級（規格 §6.3）",
               "呼叫 %d 次（預期 2：原始 + 加上格式糾正提示的重試）" % len(calls))
    if len(calls) == 2:
        gate.check("只輸出 JSON" in calls[1], "重試時在 prompt 前加上格式糾正提示",
                   repr(calls[1][:30]))


async def run_stream_checks(gate, args, audio):
    ws_url = args.url + "/ws/session?course_id=" + args.course
    peak_vram = gpu.used_mb()
    async with WsClient(ws_url) as c:
        await c.send({"type": "start", "course_id": args.course,
                      "client_ts": int(time.time() * 1000)})
        started = await c.wait_for("session_started", timeout=30)
        if started is None:
            gate.check(False, "建立 session", "沒收到 session_started")
            return
        gate.info("session %s" % started["session_id"])

        pos = 0
        latencies = []

        # 1. 連續觸發 5 次摘要，量測延遲與佇列深度 -------------------
        for i in range(5):
            pos = await feed_seconds(c, audio, pos, args.section_s)
            peak_vram = max(peak_vram, gpu.used_mb())
            before = len(c.events)
            t0 = time.monotonic()
            await c.send({"type": "mark", "note": "第 %d 段重點" % (i + 1)})
            evt = await c.wait_for("summary", timeout=90, since=before)
            peak_vram = max(peak_vram, gpu.used_mb())
            if evt is None:
                gate.check(False, "按鈕到摘要回傳 P95 < 8s",
                           "第 %d 次 mark 在 90s 內沒收到 summary" % (i + 1))
                return
            latencies.append(time.monotonic() - t0)

        p95 = percentile(latencies, 0.95)
        gate.check(p95 < 8.0, "按鈕到摘要回傳 P95 < 8s",
                   "P95 %.2fs，各次 %s" % (p95, ["%.1f" % x for x in latencies]))

        depths = [s["queue_depth"] for s in c.stats]
        gate.check(bool(depths) and max(depths) <= 2,
                   "摘要生成期間 ASR 佇列深度不增長（queue_depth <= 2）",
                   "取樣 %d 次，最大 %s" % (len(depths),
                                            max(depths) if depths else "n/a"))

        # 2. 去抖動：100ms 內送 3 次 mark -------------------------
        pos = await feed_seconds(c, audio, pos, args.section_s)
        n_before = c.count("summary")
        for _ in range(3):
            await c.send({"type": "mark"})
            await asyncio.sleep(0.033)
        await asyncio.sleep(1.0)
        await c.wait_for("summary", timeout=90, since=len(c.events) - 1)
        await asyncio.sleep(2.0)
        produced = c.count("summary") - n_before
        gate.check(produced == 1, "去抖動：100ms 內送 3 次 mark 只產生 1 則 section",
                   "產生 %d 則" % produced)

        # 3. 短 buffer 合併：30 秒內二次按鈕 ----------------------
        sections_before = c.count("summary")
        last_summary = [e for e in c.events if e.get("type") == "summary"][-1]
        pos = await feed_seconds(c, audio, pos, 12)
        before = len(c.events)
        await c.send({"type": "mark", "note": "補充"})
        merged = await c.wait_for("summary", timeout=90, since=before)
        gate.check(merged is not None
                   and merged["section_id"] == last_summary["section_id"],
                   "短 buffer 合併：30 秒內二次按鈕，section 數不增加",
                   "section_id %s → %s" % (last_summary["section_id"],
                                           merged["section_id"] if merged else "無回應"))
        changed = (merged is not None
                   and (merged["bullets"] != last_summary["bullets"]
                        or merged["title"] != last_summary["title"]
                        or merged.get("end") != last_summary.get("end")))
        gate.check(changed, "短 buffer 合併：內容有更新",
                   "title/bullets/end 至少一項改變" if changed else "內容完全相同")

        session_id = c.session_id
        await c.send({"type": "end"})
        await c.wait_for("final_summary", timeout=300)

    # 4. VRAM 峰值 ------------------------------------------------
    bench = load_bench()
    gpu_info = bench.get("gpu") or {}
    available = gpu_info.get("available_mb", 0)
    baseline = gpu_info.get("baseline_used_mb", 0)
    if available:
        # peak_vram 是 nvidia-smi 的絕對用量（含桌面基準），
        # available 已經扣掉基準了，兩邊要換算到同一個基準才能比。
        limit = available * config.VRAM_BUDGET_RATIO
        net_peak = max(0.0, peak_vram - baseline)
        gate.check(net_peak <= limit, "全程 VRAM 峰值 <= 可用量的 85%",
                   "模型峰值 %.0fMB（絕對 %.0fMB − 桌面基準 %.0fMB），"
                   "上限 %.0fMB（可用 %.0fMB × 85%%）"
                   % (net_peak, peak_vram, baseline, limit, available))
    else:
        gate.skip("全程 VRAM 峰值 <= 可用量的 85%",
                  "bench.json 沒有 gpu.available_mb，請先跑 bench_vram.py")

    # 5. 資料庫中的 section 列數（規格 §14.4）--------------------
    from server import storage
    storage.init()
    n_rows = storage.count_sections(session_id) if session_id else 0
    gate.check(n_rows > 0, "sections 已寫入資料庫", "session %s 共 %d 列"
               % (session_id, n_rows))
    merge_bench({"m2": {"summary_latency_p95_s": round(percentile(latencies, 0.95), 2),
                        "vram_peak_mb": round(max(0.0, peak_vram - baseline), 1),
                        "vram_peak_abs_mb": round(peak_vram, 1),
                        "sections_rows": n_rows}})
    return session_id


async def main_async(args) -> int:
    header("verify_m2.py — 閘門 G2：按鈕摘要（規格 §11 M2）")
    gate = Gate("G2")

    base_http = args.url.replace("ws://", "http://").replace("wss://", "https://")
    proc = None
    if args.start_server:
        gate.info("啟動 uvicorn 子行程…")
        proc = subprocess.Popen([sys.executable, "-m", "server.main"], cwd=str(ROOT))
    if not await wait_server(base_http):
        gate.check(False, "服務可連線",
                   "%s/api/health 無回應。請先執行 python -m server.main" % base_http)
        if proc:
            proc.terminate()
        return gate.finish()
    gate.check(True, "服務可連線", base_http)

    fixture = (Path(args.audio) if args.audio
               else find_fixture("lecture_3h.wav", "lecture_synth.wav",
                             "bench.wav", "m0_sample.wav"))
    if fixture is None or not fixture.exists():
        gate.check(False, "找到測試音檔", "請放音檔到 tests/fixtures/")
        if proc:
            proc.terminate()
        return gate.finish()
    audio = load_wav(fixture)

    test_sid = None
    try:
        test_sid = await run_stream_checks(gate, args, audio)
        await check_json_rate(gate)
    finally:
        # 這個閘門是打真的伺服器跑的，session 會寫進正式資料庫。
        # 不清掉的話使用者的課堂列表裡會混進一堆 ml-2026 的測試紀錄。
        if test_sid:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.delete("%s/api/sessions/%s" % (base_http, test_sid))
                gate.info("已清除測試 session %s（HTTP %d）"
                          % (test_sid[:8], r.status_code))
            except Exception as e:
                gate.info("清除測試 session 失敗（請手動刪除 %s）：%s"
                          % (test_sid[:8], e))
        if proc:
            proc.terminate()
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--course", default="ml-2026")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--section-s", type=float, default=45.0,
                    help="每段餵多少秒音訊再按鈕（需 > MIN_SECTION_S 30 秒）")
    ap.add_argument("--start-server", action="store_true")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
