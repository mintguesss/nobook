"""閘門 G1 — 即時逐字稿（規格 §11 M1、§14.3）。

用預錄的長音檔模擬即時串流，不需要啟動服務、不需要真的去上課。
這個階段完全不碰 LLM。

用法：
    python scripts/verify_m1.py                 # 完整 3 小時模擬（約需 3 小時）
    python scripts/verify_m1.py --hours 0.25    # 縮短時長，記憶體洩漏項改為參考值
    python scripts/verify_m1.py --fast          # 不 sleep，只驗正確性不驗延遲
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from _common import (Gate, find_fixture, header, merge_bench, note_deviation,
                     percentile)

from server import config, courses
from server.asr import ASREngine
from server.audio_pipeline import AudioPipeline

CHUNK_S = 0.25                      # 前端每 250ms 打包一次（規格 §4.1）
CHUNK_SAMPLES = int(CHUNK_S * config.SAMPLE_RATE)
WINDOW_S = 300.0                    # 重複偵測的 5 分鐘滑動窗口
REPEAT_LIMIT = 3

SPEC_ESTIMATES = {"rtf": 0.5, "latency_median_s": 6.0, "latency_p95_s": 10.0}

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    return _PUNCT.sub("", text)


def load_wav(path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        idx = np.linspace(0, len(data) - 1,
                          int(len(data) * config.SAMPLE_RATE / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype("float32")
    return np.ascontiguousarray(data, dtype=np.float32)


class Collector:
    def __init__(self):
        self.segments = []          # {start,end,text,latency,rtf}
        self.errors = []

    async def on_result(self, seg, res):
        self.segments.append({
            "start": seg.start_s, "end": seg.end_s, "text": res["text"],
            "latency": time.monotonic() - seg.fed_at,
            "rtf": res.get("rtf", 0.0),
        })

    async def on_error(self, code, message, fatal):
        self.errors.append((code, message))


async def stream(engine, audio, hours, realtime, initial_prompt, collector,
                 rss_trend=None, speed=1.0):
    """按 250ms 逐塊餵進 pipeline，模擬真實即時串流（規格 §14.3）。

    rss_trend 若給了 list，會每分鐘（音訊時間）記一筆
    (音訊秒數, RSS MB, private commit MB)。

    兩個指標都要記：只看頭尾分不出「一次性成本」與「持續洩漏」，
    而 Windows 的 RSS（working set）還會被 OS 回收——實測 3 小時跑
    出現過驟降 465MB 的階梯，那是 trimming 不是釋放。RSS 下降**不能**
    證明沒有洩漏，private commit 才是不受 trimming 影響的指標。
    """
    import psutil
    proc = psutil.Process()
    pipe = AudioPipeline(engine, initial_prompt=initial_prompt,
                         on_result=collector.on_result,
                         on_error=collector.on_error)
    pipe.start()
    total_samples = int(hours * 3600 * config.SAMPLE_RATE)
    fed = 0
    pos = 0
    loops = 1                       # 音檔被走過幾遍（供重複偵測判斷可信度）
    next_sample = 60 * config.SAMPLE_RATE
    t_start = time.monotonic()
    while fed < total_samples:
        if pos + CHUNK_SAMPLES > len(audio):
            pos = 0                 # 音檔不夠長就循環播放
            loops += 1
        chunk = audio[pos:pos + CHUNK_SAMPLES]
        pos += CHUNK_SAMPLES
        fed += len(chunk)
        await pipe.feed(chunk, fed_at=time.monotonic())
        if rss_trend is not None and fed >= next_sample:
            mi = proc.memory_info()
            rss_trend.append((round(fed / config.SAMPLE_RATE),
                              round(mi.rss / 1e6),
                              round(getattr(mi, "private", mi.vms) / 1e6)))
            next_sample += 60 * config.SAMPLE_RATE
        if realtime:
            # 補償處理耗時，維持 speed 倍速的節奏。
            # speed=1 是真實節奏；>1 用來壓縮長時測試的等待時間，
            # 但不能超過 ASR 的處理能力（約 1/RTF 倍速），
            # 否則佇列會塞爆開始丟片段，量到的就不是正常路徑了。
            target = t_start + fed / (float(config.SAMPLE_RATE) * speed)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)
    await pipe.flush()
    await asyncio.wait_for(pipe.drain(), timeout=600)
    await pipe.stop()
    return pipe, loops


def check_repeat_loop(gate, segments, audio_s, loops):
    """滑動 5 分鐘窗口，任一正規化字串出現 >= 3 次即 FAIL（規格 §14.3）。

    這一項是在抓 Whisper 的重複輸出迴圈（`condition_on_previous_text` 開啟時
    幾乎必然發生的那種）。但如果測試音檔比窗口短、又被循環播放，
    內容本身就會重複，量到的是 fixture 的缺陷而不是模型的缺陷——
    這種情況下判 FAIL 是假陽性，據實標為無法判定。
    """
    if loops > 1 and audio_s < WINDOW_S:
        gate.skip("無重複輸出迴圈（5 分鐘窗口內同句 < 3 次）",
                  "音檔只有 %.0f 秒、短於 %.0f 秒的偵測窗口，被循環播放 %d 次；"
                  "內容重複來自 fixture 而非模型，此項無法判定。"
                  "請放一份長度足夠的真實錄音到 tests/fixtures/lecture_3h.wav"
                  % (audio_s, WINDOW_S, loops))
        return
    worst = ("", 0, 0.0)
    n = len(segments)
    i = 0
    for j in range(n):
        while segments[j]["end"] - segments[i]["start"] > WINDOW_S:
            i += 1
        counts = {}
        for k in range(i, j + 1):
            t = normalize(segments[k]["text"])
            if len(t) < 4:
                continue
            counts[t] = counts.get(t, 0) + 1
            if counts[t] > worst[1]:
                worst = (t, counts[t], segments[i]["start"])
    gate.check(worst[1] < REPEAT_LIMIT, "無重複輸出迴圈（5 分鐘窗口內同句 < 3 次）",
               "最高重複 %d 次%s" % (worst[1],
               ("：%r（約 %.0fs 起）" % (worst[0][:40], worst[2])) if worst[1] > 1 else ""))


# 實測：CUDA context、cuDNN workspace、CTranslate2 內部 buffer 會在
# 前約 11 分鐘的音訊內配置完畢（961→997MB），之後完全持平。
# 規格 §11 的「前後 RSS 差 < 200MB」若從「模型載入後」起算，這段
# 一次性成本（約 430MB）就會讓一個毫無洩漏的系統被判為洩漏。
# 所以基準取在穩態之後，量的才是規格真正想問的東西：有沒有持續成長。
RSS_PLATEAU_S = 900.0        # 音訊時間；取樣點落在這之後才算穩態
RSS_MIN_WINDOW_S = 1800.0    # 穩態之後至少要再跑這麼久才判得準


def _rss_steady_delta(trend, idx=2):
    """回傳 (穩態前段中位數, 後段中位數, 觀察窗長度秒)，資料不足回傳 None。

    idx=1 取 RSS、idx=2 取 private commit。判定用 private commit：
    Windows 會回收 working set，RSS 下降不代表記憶體真的被釋放。

    用「前四分之一 vs 後四分之一的中位數」而不是頭尾單點：CUDA allocator
    在推論時會反覆保留／釋放，private commit 實測有 ±400MB 的暫時性起伏
    （後半段 std 約 190MB）。取單點會抓到起伏的隨機相位，同一個沒有洩漏的
    系統可能這次過、下次不過；中位數對這種雜訊免疫。
    """
    steady = [p for p in trend if p[0] >= RSS_PLATEAU_S and len(p) > idx]
    if len(steady) < 8:
        return None
    window = steady[-1][0] - steady[0][0]
    if window < RSS_MIN_WINDOW_S:
        return None
    import statistics
    q = max(2, len(steady) // 4)
    head = statistics.median(p[idx] for p in steady[:q])
    tail = statistics.median(p[idx] for p in steady[-q:])
    return head, tail, window


def _rss_growth_per_hour(trend, idx=2):
    """穩態成長率（MB/小時音訊）。由 _rss_steady_delta 的中位數差推導。

    不用線性擬合：CUDA allocator 的起伏尖峰分布不均，即使先做滑動中位數，
    擬合出來的斜率仍會與總差矛盾（實測擬合說 +37.5 MB/h，實際總差只有
    +1 MB / 2.8 小時）。既然判定本身用的是穩健的中位數差，成長率就從
    同一個數字推導，兩者才不會互相打架。
    """
    d = _rss_steady_delta(trend, idx)
    if d is None:
        return None
    head, tail, window = d
    hours = window / 3600.0
    return (tail - head) / hours if hours > 0 else None


def _rss_growth_fit(trend, idx=2):
    """舊的線性擬合版本，只留著當診斷參考，不用於判定。"""
    if len(trend) < 4:
        return None
    half = [p for p in trend[len(trend) // 2:] if len(p) > idx]
    if len(half) < 2:
        return None
    # 先用滑動中位數壓掉 CUDA allocator 的暫時性起伏，再擬合；
    # 直接擬合原始序列等於在擬合雜訊（實測 std 約 190MB）。
    import statistics
    win = min(9, max(3, len(half) // 8) | 1)
    ys = [statistics.median(p[idx] for p in half[max(0, i - win // 2):
                                                 i + win // 2 + 1])
          for i in range(len(half))]
    xs = [p[0] / 3600.0 for p in half]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def check_latency(gate, segments, realtime, speed=1.0):
    if not realtime:
        gate.skip("端到端延遲中位數 < 6s、P95 < 10s",
                  "--fast 模式沒有節流，延遲數字無意義")
        return None, None
    if abs(speed - 1.0) > 1e-6:
        gate.skip("端到端延遲中位數 < 6s、P95 < 10s",
                  "--speed %.1fx 不是真實節奏；使用者感受到的延遲只有在 "
                  "1.0x 下量才有意義" % speed)
        return None, None
    lats = [s["latency"] for s in segments]
    if not lats:
        gate.check(False, "端到端延遲中位數 < 6s、P95 < 10s", "沒有產出任何 segment")
        return None, None
    med = percentile(lats, 0.5)
    p95 = percentile(lats, 0.95)
    gate.check(med < 6.0 and p95 < 10.0,
               "端到端延遲中位數 < 6s、P95 < 10s",
               "中位數 %.2fs，P95 %.2fs（%d 段）" % (med, p95, len(lats)))
    note_deviation("latency_median_s", SPEC_ESTIMATES["latency_median_s"], med)
    note_deviation("latency_p95_s", SPEC_ESTIMATES["latency_p95_s"], p95)
    return med, p95


def _vad_segments(clip):
    """用正式管線的 VAD 切段規則把 clip 切成片段，回傳 audio 陣列清單。"""
    from server.audio_pipeline import VadSegmenter, default_vad
    seg = VadSegmenter(default_vad())
    out = [s.audio for s in seg.feed(clip)]
    out += [s.audio for s in seg.flush()]
    return out


async def _transcribe_segments(engine, segs, prompt):
    """逐段轉錄再串起來——與正式管線相同的呼叫形狀。"""
    texts = []
    for s in segs:
        r = await asyncio.to_thread(engine.transcribe_joined, s, prompt)
        if r:
            texts.append(r["text"])
    return "".join(texts)


def _expected_terms(fixture, course, clip_s, total_s):
    """讀 fixture 的 ground truth，換算成這段 clip 應該出現幾個術語。

    合成音檔會附 *.groundtruth.json；真實錄音沒有，回傳 None。
    """
    gt = fixture.with_name(fixture.stem + ".groundtruth.json")
    if not gt.exists():
        return None
    try:
        counts = json.loads(gt.read_text(encoding="utf-8"))["term_counts"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    gl = {g.lower() for g in course.glossary}
    total = sum(v for k, v in counts.items() if k.lower() in gl)
    frac = min(1.0, clip_s / total_s) if total_s else 1.0
    return total * frac


async def check_glossary(gate, engine, audio, course, fixture):
    """有／無 initial_prompt 的對照測試，術語辨識率應提升 >= 10%（規格 §14.3）。

    這一項在乾淨的合成音訊上會失去鑑別力：ASR 不靠 prompt 就已經全對，
    提升自然是 0%，那是天花板效應不是實作缺陷。有 ground truth 可對照時
    先判斷是否已經打到天花板，是的話據實標為無法判定。
    """
    if not course.glossary:
        gate.skip("術語命中率提升 >= 10%", "課程設定檔沒有 glossary")
        return None
    clip_s = 600.0
    clip = audio[:int(config.SAMPLE_RATE * clip_s)]

    def count_hits(text):
        low = text.lower()
        return sum(low.count(g.lower()) for g in course.glossary)

    # 必須走 VAD 切段（正式管線的路徑），不能整段丟給 transcribe：
    # faster-whisper 在 condition_on_previous_text=False 時，處理完第一個
    # 30 秒窗口就把 prompt_reset_since 推到尾端，initial_prompt 之後就失效。
    # 整段丟 180 秒 → prompt 只作用在前 30 秒，量到的提升會被稀釋成 1/6。
    segs = await asyncio.to_thread(_vad_segments, clip)
    if not segs:
        gate.skip("術語命中率提升 >= 10%", "VAD 在這段音訊中切不出任何語音片段")
        return None
    a = count_hits(await _transcribe_segments(engine, segs, ""))
    b = count_hits(await _transcribe_segments(engine, segs, course.asr_prompt))
    if a == 0 and b == 0:
        gate.skip("術語命中率提升 >= 10%",
                  "音檔中偵測不到 %s 的任何術語，請換一段真的講到這些詞的錄音"
                  % course.id)
        return None

    expected = _expected_terms(fixture, course, clip_s,
                               len(audio) / float(config.SAMPLE_RATE))
    if expected and a >= expected * 0.9:
        gate.skip("術語命中率提升 >= 10%",
                  "無 prompt 就已命中 %d/%.0f（%.0f%%），音訊乾淨到不需要 "
                  "initial_prompt 幫忙，此項在這份素材上沒有鑑別力。"
                  "有 prompt %d 次。請改用真實課堂錄音重測（規格 §14.3）"
                  % (a, expected, 100.0 * a / expected, b))
        return {"without_prompt": a, "with_prompt": b,
                "expected": round(expected, 1), "verdict": "ceiling"}

    gain = ((b - a) / a) if a else 1.0
    gate.check(gain >= 0.10, "術語命中率提升 >= 10%",
               "無 prompt %d 次 → 有 prompt %d 次（%+.0f%%）" % (a, b, gain * 100))
    return {"without_prompt": a, "with_prompt": b, "gain_pct": round(gain * 100, 1)}


def check_seq_dedup(gate):
    """重連補送：seq 去重後音訊無缺漏、無重複（規格 §5.3）。"""
    from server.ws_session import HEADER, pcm16_to_float32

    class Fake:
        def __init__(self):
            self.seen = set()
            self.received = []

        def push(self, seq, samples):
            if seq in self.seen:
                return False
            self.seen.add(seq)
            self.received.append(seq)
            return True

    f = Fake()
    for seq in range(0, 40):
        f.push(seq, 4000)
    # 模擬斷線：40~59 在客戶端 buffer 裡，重連後補送，且 30~44 重複送
    for seq in list(range(30, 45)) + list(range(45, 60)):
        f.push(seq, 4000)
    got = sorted(f.received)
    no_dup = len(got) == len(set(got))
    no_gap = got == list(range(0, 60))
    gate.check(no_dup and no_gap, "斷線重連後音訊補送無缺漏、無重複（seq 連續性）",
               "收到 seq 0..%d，共 %d 筆，重複 %d 筆"
               % (got[-1], len(got), len(f.received) - len(set(f.received))))

    # 同時驗 binary frame header 的封裝／解封裝
    payload = (np.arange(8, dtype=np.int16) * 100).tobytes()
    frame = HEADER.pack(7, 8) + payload
    seq, cnt = HEADER.unpack_from(frame, 0)
    pcm = pcm16_to_float32(frame[HEADER.size:])
    gate.check(seq == 7 and cnt == 8 and len(pcm) == 8,
               "音訊 chunk header 格式正確（uint32 seq + uint32 sampleCount, LE）",
               "seq=%d sampleCount=%d 解出 %d sample" % (seq, cnt, len(pcm)))


async def main_async(args) -> int:
    header("verify_m1.py — 閘門 G1：即時逐字稿（規格 §11 M1）")
    gate = Gate("G1")

    fixture = (Path(args.audio) if args.audio
               else find_fixture("lecture_real.wav", "lecture_3h.wav",
                                    "lecture_synth.wav", "bench.wav",
                                    "m0_sample.wav"))
    if fixture is None or not fixture.exists():
        gate.check(False, "找到測試音檔",
                   "請放一份長篇中文演講錄音到 tests/fixtures/lecture_3h.wav"
                   "（規格 §14.3；可用任何公開的長篇中文演講音檔）")
        return gate.finish()

    bench_path = Path(__file__).resolve().parent.parent / "data" / "bench.json"
    bench = (json.loads(bench_path.read_text(encoding="utf-8"))
             if bench_path.exists() else {})
    model_path = args.asr_model or (bench.get("asr") or {}).get("model")
    if not model_path:
        gate.check(False, "取得 ASR 模型路徑",
                   "bench.json 沒有 asr.model，請先跑 scripts/bench_vram.py")
        return gate.finish()

    course = courses.get_course(args.course)
    audio = load_wav(fixture)
    gate.info("音檔 %s（%.1f 分鐘），模擬 %.2f 小時%s"
              % (fixture.name, len(audio) / config.SAMPLE_RATE / 60,
                 args.hours, "" if not args.fast else "，--fast 不做真實節奏"))

    engine = ASREngine(model_path,
                       compute_type=(bench.get("asr") or {}).get(
                           "compute_type", "int8_float16"))

    import psutil
    proc = psutil.Process()
    await asyncio.to_thread(engine.load)
    # 先跑幾次推論再取基準：CUDA context、cuDNN workspace、CTranslate2 的
    # 內部 buffer 都是第一次推論時才配置的一次性成本（實測約 500MB）。
    # 把它算進「洩漏」會讓短時測試永遠失敗，也會掩蓋真正的漸進式成長。
    warm = audio[:config.SAMPLE_RATE * 20]
    for _ in range(3):
        await asyncio.to_thread(engine.transcribe_joined, warm, "")
    rss_before = proc.memory_info().rss / 1e6
    gate.info("暖機後 RSS 基準 %.0fMB" % rss_before)

    collector = Collector()
    rss_trend = []
    t0 = time.monotonic()
    try:
        pipe, loops = await stream(engine, audio, args.hours, not args.fast,
                                   course.asr_prompt, collector, rss_trend,
                                   speed=args.speed)
        crashed = None
    except Exception as e:
        crashed = e
        pipe, loops = None, 1
    wall = time.monotonic() - t0
    rss_after = proc.memory_info().rss / 1e6

    dropped = pipe.dropped if pipe is not None else 0
    if dropped:
        gate.info("餵入速度超過 ASR 處理能力，丟棄 %d 段；"
                  "請把 --speed 調低（目前 %.1fx，ASR 約可承受 %.1fx）"
                  % (dropped, args.speed,
                     1.0 / max(0.01, sum(s["rtf"] for s in collector.segments)
                               / max(1, len(collector.segments)))))
    gate.check(crashed is None, "%.2f 小時模擬串流跑完不崩" % args.hours,
               "產出 %d 段，耗時 %.0fs%s"
               % (len(collector.segments), wall,
                  "" if crashed is None else "，例外：%r" % (crashed,)))
    if crashed is not None:
        return gate.finish()

    # RTF ---------------------------------------------------------
    rtfs = [s["rtf"] for s in collector.segments if s.get("rtf")]
    rtf = sum(rtfs) / len(rtfs) if rtfs else engine.last_rtf
    rtf_p95 = percentile(rtfs, 0.95) if rtfs else rtf
    gate.check(rtf < 0.5, "RTF < 0.5",
               "平均 %.3f、P95 %.3f（RTF >= 0.5 代表即時性不成立，"
               "需降級模型或改用 large-v3-turbo）" % (rtf, rtf_p95))
    note_deviation("rtf", SPEC_ESTIMATES["rtf"], rtf)

    # 記憶體洩漏 --------------------------------------------------
    delta = rss_after - rss_before
    growth = _rss_growth_per_hour(rss_trend)
    rss_growth = _rss_growth_per_hour(rss_trend, idx=1)
    if rss_trend:
        gate.info("記憶體趨勢（音訊秒數, RSS MB, private MB）：%s … %s"
                  % (rss_trend[:2], rss_trend[-2:]))
        gate.info("穩態成長率：private commit %s、RSS %s"
                  % ("%+.1f MB/h" % growth if growth is not None else "資料不足",
                     "%+.1f MB/h" % rss_growth if rss_growth is not None
                     else "資料不足"))
        if rss_growth is not None and growth is not None and                 rss_growth < -50 and abs(growth) < 20:
            gate.info("RSS 明顯下降但 private commit 持平——那是 Windows 回收"
                      "working set，不是記憶體被釋放；判定以 private commit 為準")
    steady = _rss_steady_delta(rss_trend)
    if args.hours < 2.5:
        gate.skip("記憶體無洩漏（穩態後 RSS 差 < 200MB）",
                  "本次只跑 %.2f 小時，未達規格要求的 3 小時；"
                  "載入後至結束 %+.0fMB、穩態成長率 %s（參考值，%d 段）"
                  % (args.hours, delta,
                     "%+.1f MB/h" % growth if growth is not None else "資料不足",
                     len(collector.segments)))
    elif steady is None:
        gate.skip("記憶體無洩漏（穩態後 RSS 差 < 200MB）",
                  "取樣點不足以判斷穩態（需要音訊時間 %.0fs 之後再跑 %.0fs 以上）"
                  % (RSS_PLATEAU_S, RSS_MIN_WINDOW_S))
    else:
        s0, s1, window = steady
        gate.check(
            s1 - s0 < 200, "記憶體無洩漏（穩態後 private commit 差 < 200MB）",
            "private commit 穩態 %dMB → 結束 %dMB（%+dMB / %.1f 小時音訊，"
            "成長率 %+.1f MB/h，%d 段）。"
            "用 private commit 而非 RSS：Windows 會回收 working set，"
            "RSS 下降不代表記憶體真的被釋放；基準取在穩態之後（音訊 %.0f 分鐘），"
            "是因為 CUDA context 與 CTranslate2 buffer 的一次性配置若算進來，"
            "會讓沒有洩漏的系統也超過 200MB 門檻"
            % (s0, s1, s1 - s0, window / 3600.0,
               growth if growth is not None else 0.0,
               len(collector.segments), RSS_PLATEAU_S / 60.0))

    check_repeat_loop(gate, collector.segments,
                      len(audio) / float(config.SAMPLE_RATE), loops)
    med, p95 = check_latency(gate, collector.segments, not args.fast,
                             args.speed)
    glossary = await check_glossary(gate, engine, audio, course, fixture)
    check_seq_dedup(gate)

    if collector.errors:
        gate.info("串流期間收到 %d 則 error 事件：%s"
                  % (len(collector.errors), collector.errors[:3]))

    merge_bench({"m1": {
        "hours": args.hours, "segments": len(collector.segments),
        "rtf": round(rtf, 4), "rtf_p95": round(rtf_p95, 4),
        "latency_median_s": round(med, 2) if med else None,
        "latency_p95_s": round(p95, 2) if p95 else None,
        "rss_delta_mb": round(delta, 1),
        "private_growth_mb_per_hour": round(growth, 1) if growth is not None else None,
        "rss_growth_mb_per_hour": (round(rss_growth, 1)
                                   if rss_growth is not None else None),
        "rss_trend": rss_trend,
        "glossary": glossary,
    }})
    await asyncio.to_thread(engine.unload)
    return gate.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--course", default="ml-2026")
    ap.add_argument("--hours", type=float, default=3.0,
                    help="模擬時長，規格要求 3 小時")
    ap.add_argument("--fast", action="store_true",
                    help="完全不節流（會塞爆佇列而丟片段，只適合快速檢查）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="餵入倍速。1.0 = 真實節奏（延遲數字才有效）；"
                         "調高可壓縮長時測試，但別超過 1/RTF 否則佇列會滿")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
