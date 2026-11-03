"""講者辨識：錄完之後整段跑一次，標出每段話是誰講的。

只分得出「講者 1、講者 2」，不知道誰是誰。上課不需要這層，所以不進即時
流程；模型或 sherpa-onnx 缺席時 available() 回 False，其他功能照常。

做法：只取 VAD 判定為語音的部分，每 0.75 秒取一個 1.5 秒窗算聲紋，再做譜分群。
先試過 sherpa-onnx 內建的 pyannote 流程，3 個人被分成 25 群；靜音和雜訊的
窗一混進來，聲紋分群就垮了。全程跑在 CPU，不佔 VRAM。
"""
from __future__ import annotations

import logging
import os

import numpy as np

from . import config

log = logging.getLogger(__name__)

EMBEDDING_MODEL = config.MODELS_DIR / "diarization" / "wespeaker_zh_cnceleb_resnet34.onnx"
EMBEDDING_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                 "speaker-recongition-models/" + EMBEDDING_MODEL.name)

SR = config.SAMPLE_RATE
WINDOW_S = 1.5
HOP_S = 0.75
MIN_REGION_S = 0.5       # 短於此的語音片段聲紋不可靠，不拿來分群
# 人數未知時先分細再合併。直接看特徵值落差猜人數，1 個老師會被分成 5 人。
# 門檻實測（已知講者的剪接音檔）：0.3 時 3 人 92~99%、5 人 86~92%；到 0.35 就開始
# 把不同人併在一起，5 人短發言掉到 43%。
OVERSEGMENT_K = 12
MERGE_DISTANCE = 0.3
SMOOTH_WINDOWS = 5       # 用前後幾個窗投票，去掉只跳一下的誤判
# 佔全場語音不到這個比例的群多半是咳嗽、雜音，併給最像的講者
MIN_SPEAKER_SHARE = 0.01
TURN_MERGE_GAP_S = 0.5   # 同一人中間停頓不到這麼久就視為同一次發言
NEAREST_GAP_S = 0.5      # 對不到講者的短句，只跟著這麼近的發言標


def available() -> bool:
    if not EMBEDDING_MODEL.exists():
        return False
    try:
        import scipy  # noqa: F401
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def speech_regions(audio: np.ndarray, vad):
    """回傳語音區間 [(start_sample, end_sample)]，已濾掉太短的片段。"""
    if hasattr(vad, "reset"):
        vad.reset()
    frame = config.VAD_FRAME_SAMPLES
    regions = []
    start = None
    for i in range(0, len(audio) - frame + 1, frame):
        is_speech = float(vad(audio[i:i + frame])) >= config.VAD_THRESHOLD
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(audio)))
    min_len = int(MIN_REGION_S * SR)
    return [r for r in regions if r[1] - r[0] >= min_len]


def _windows(regions):
    """語音區間切成聲紋窗。不足一個窗長的區間就整段當一個窗；最後補一個貼齊
    區間結尾的窗，句尾才分得到講者。"""
    win, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    out = []
    for start, end in regions:
        if end - start <= win:
            out.append((start, end))
            continue
        for i in range(start, end - win + 1, hop):
            out.append((i, i + win))
        if out[-1][1] < end:
            out.append((end - win, end))
    return out


def _embed(audio, windows, threads):
    import sherpa_onnx
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMBEDDING_MODEL), num_threads=threads))
    out = []
    for start, end in windows:
        stream = extractor.create_stream()
        stream.accept_waveform(SR, audio[start:end])
        stream.input_finished()
        out.append(np.asarray(extractor.compute(stream), dtype=np.float32))
    vectors = np.stack(out)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def _spectral_labels(vectors, k):
    """把聲紋窗分成 k 群。"""
    from scipy.cluster.vq import kmeans2
    from scipy.linalg import eigh
    n = len(vectors)
    k = min(k, n - 1)
    if k < 2:
        return np.zeros(n, dtype=int)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, 0)
    # 每個窗只留最像的幾個鄰居，離群的窗才不會把群連在一起
    neighbors = max(5, int(0.03 * n))
    top = np.argsort(-sim, axis=1)[:, :neighbors]
    rows = np.arange(n)[:, None]
    graph = np.zeros_like(sim)
    graph[rows, top] = np.clip(sim[rows, top], 0, None)
    graph = (graph + graph.T) / 2
    degree = np.maximum(graph.sum(axis=1), 1e-6)
    laplacian = np.eye(n) - graph / np.sqrt(np.outer(degree, degree))
    _, vecs = eigh(laplacian, subset_by_index=[0, k - 1])
    points = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    best_err, best = None, None
    for seed in range(5):
        centers, labels = kmeans2(points, k, minit="++", seed=seed)
        err = float(((points - centers[labels]) ** 2).sum())
        if best_err is None or err < best_err:
            best_err, best = err, labels
    return best


def _merge_similar_clusters(vectors, labels):
    """聲音夠像的群併成同一人。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    ids = np.unique(labels)
    if len(ids) < 2:
        return labels
    centers = np.stack([vectors[labels == i].mean(axis=0) for i in ids])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    merged = fcluster(linkage(centers, "average", metric="cosine"),
                      MERGE_DISTANCE, "distance")
    mapping = dict(zip(ids.tolist(), merged.tolist()))
    return np.array([mapping[i] for i in labels.tolist()])


def _merge_small_clusters(vectors, labels):
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids[counts >= MIN_SPEAKER_SHARE * len(labels)]
    if len(keep) in (0, len(ids)):
        return labels
    centers = np.stack([vectors[labels == i].mean(axis=0) for i in keep])
    out = labels.copy()
    for i in np.where(~np.isin(labels, keep))[0]:
        out[i] = keep[int(np.argmax(centers @ vectors[i]))]
    return out


def _smooth(labels, windows):
    """連續語音裡只跳一兩個窗的講者視為誤判。隔太遠的窗不互相投票。"""
    half = SMOOTH_WINDOWS // 2
    max_gap = int(2 * HOP_S * SR)
    out = labels.copy()
    for i in range(len(labels)):
        votes = [labels[i]]
        for step in (-1, 1):
            j = i
            for _ in range(half):
                nxt = j + step
                if not 0 <= nxt < len(labels):
                    break
                if abs(windows[nxt][0] - windows[j][0]) > max_gap:
                    break
                j = nxt
                votes.append(labels[j])
        values, counts = np.unique(votes, return_counts=True)
        out[i] = values[int(np.argmax(counts))]
    return out


def _turns(windows, labels):
    """窗接成發言區間，並把最先開口的人編成講者 1。

    重疊的窗以中心點為界；語音區間的頭尾歸給第一個和最後一個窗，
    不然短句和句子頭尾會落在所有區間之外，變成沒有講者。
    """
    centers = [(start + end) / 2 / SR for start, end in windows]
    merged = []
    for i, ((start, end), spk) in enumerate(zip(windows, labels)):
        t_start, t_end = start / SR, end / SR
        if i > 0 and windows[i - 1][1] > start:
            t_start = (centers[i - 1] + centers[i]) / 2
        if i + 1 < len(windows) and windows[i + 1][0] < end:
            t_end = (centers[i] + centers[i + 1]) / 2
        spk = int(spk)
        if merged and merged[-1][2] == spk and t_start - merged[-1][1] <= TURN_MERGE_GAP_S:
            merged[-1][1] = t_end
        else:
            merged.append([t_start, t_end, spk])
    return renumber([tuple(t) for t in merged])


def diarize(audio: np.ndarray, vad, num_speakers: int = 0, threads: int = 0):
    """audio: float32 16kHz 單聲道。回傳依時間排序的 [(start_s, end_s, 講者編號)]。

    num_speakers 已知就填，比自動判斷準；0 代表自動。
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    windows = _windows(speech_regions(audio, vad))
    if not windows:
        return []
    threads = threads or max(1, (os.cpu_count() or 4) // 2)
    vectors = _embed(audio, windows, threads)
    if num_speakers > 0:
        labels = _spectral_labels(vectors, num_speakers)
    else:
        labels = _merge_similar_clusters(
            vectors, _spectral_labels(vectors, OVERSEGMENT_K))
        labels = _merge_small_clusters(vectors, labels)
    labels = _smooth(labels, windows)
    return _turns(windows, labels)


def renumber(turns):
    """講者編號照第一次開口的順序重排，從 1 開始。"""
    seen = {}
    out = []
    for start, end, spk in sorted(turns):
        seen.setdefault(spk, len(seen) + 1)
        out.append((start, end, seen[spk]))
    return out


def assign_speaker(start_s: float, end_s: float, turns):
    """這段時間裡講最久的人。完全沒重疊時回傳 None。turns 須依時間排序。"""
    spoken = {}
    for t_start, t_end, spk in turns:
        if t_end <= start_s:
            continue
        if t_start >= end_s:
            break
        spoken[spk] = spoken.get(spk, 0.0) + min(end_s, t_end) - max(start_s, t_start)
    if not spoken:
        return None
    return max(spoken, key=spoken.get)


def split_words_by_speaker(words, turns):
    """把一段 ASR 的逐字時間戳依講者切開，避免一段話裡換人卻只標一個人。

    words: [(start_s, end_s, 文字)]。回傳 [(start_s, end_s, 文字, 講者或 None)]。
    對不到任何講者的字跟著前一個字走（多半是句尾收音）。
    """
    out = []
    for w_start, w_end, word in words:
        spk = assign_speaker(w_start, w_end, turns)
        if out and (spk is None or spk == out[-1][3]):
            prev = out[-1]
            out[-1] = (prev[0], w_end, prev[2] + word, prev[3])
        else:
            out.append((w_start, w_end, word, spk))
    # 開頭幾個字沒對到人時，併進下一段
    if len(out) > 1 and out[0][3] is None:
        first, second = out[0], out[1]
        out[:2] = [(first[0], second[1], first[2] + second[2], second[3])]
    # 整段都對不到（太短算不出聲紋）時，只有緊鄰別人發言才跟著標，否則留白
    if len(out) == 1 and out[0][3] is None:
        start, end, text, _ = out[0]
        out[0] = (start, end, text, nearest_speaker(start, end, turns))
    return out


def nearest_speaker(start_s: float, end_s: float, turns, max_gap_s: float = NEAREST_GAP_S):
    """離這段時間最近、而且夠近的講者。"""
    best, best_gap = None, max_gap_s
    for t_start, t_end, spk in turns:
        gap = max(t_start - end_s, start_s - t_end, 0.0)
        if gap <= best_gap:
            best, best_gap = spk, gap
    return best
