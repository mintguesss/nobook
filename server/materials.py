"""把上課用的投影片對到逐字稿的時間軸。

為什麼要做：段落的邊界現在是「使用者剛好按按鈕的時間點」，那是隨機的。
投影片本身就帶著老師整理好的章節結構，對齊之後筆記可以照著它分段，
而且能回答上課真正想知道的那件事——**這頁投影片上沒寫、但老師講了什麼**。

做法：投影片大致照順序講，所以這是「單調對齊」。每一塊逐字稿指派給一頁
投影片，頁碼只能往前不能往回；相似度太低的塊標成「沒有對應」（老師離題、
講前一份教材、或那頁只是圖）。動態規劃掃一次，O(頁數 × 塊數)，
**純 CPU、不用模型**，一堂課幾十毫秒。

實測（2026-09-21 生產與作業管理，49 頁 PDF 對 135 塊逐字稿）：
真的在講那頁時相似度 0.5–0.94，沒在講時只有 0.27——這個差距就是門檻的依據。
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

# 低於這個相似度就當成「這塊沒有對應的投影片」。
# 0.30 取自實測：真的在講那頁時含括度 0.25–0.72，雜訊落在 0.15–0.25。
MATCH_THRESHOLD = 0.30
# 每幾句逐字稿併成一塊再比對。單句太短，比出來的相似度全是雜訊。
CHUNK_SENTENCES = 6

_EN = re.compile(r"[A-Za-z][A-Za-z0-9+.#-]{1,}")
_CJK_RE = re.compile(r"[一-鿿]")
_WS = re.compile(r"\s+")
# 到處都有的字，比對它們沒有鑑別度
_STOP = set("的了是在有和與及對不就都也很我你他這那個們會要可以所以因為但是"
            "如果然後還有什麼怎麼樣之類等等他們我們你們一個這個那個" )

SUPPORTED = (".pdf", ".pptx")


def features(text: str) -> Counter:
    """抽出比對用的特徵：中文字元 bigram ＋ 英文單字。

    用 bigram 而不是斷詞，是因為 ASR 的逐字稿沒有標點也沒有詞界，
    斷詞器在上面表現不穩；bigram 對錯字也比較不敏感（錯一個字只壞掉
    兩個 bigram，不會整個詞消失）。英文專名鑑別度高，額外加權。
    """
    c = Counter()
    cjk = "".join(ch for ch in text if _CJK_RE.match(ch) and ch not in _STOP)
    for i in range(len(cjk) - 1):
        c[cjk[i:i + 2]] += 1
    for w in _EN.findall(text):
        if len(w) > 2:
            c["en:" + w.lower()] += 2
    return c


def idf_of(docs) -> dict:
    """每個特徵的 IDF。

    投影片的頁尾會重複出現課名、教授名、日期，投影片母片的標題也是；
    這些特徵每頁都有，拿來比對只會讓每一塊都「像」每一頁。用 IDF 把它們
    的權重壓掉，剩下真正有鑑別度的內容詞。
    """
    import math
    n = len(docs) or 1
    df = Counter()
    for d in docs:
        df.update(set(d))
    return {k: math.log((n + 1) / (v + 1)) + 1.0 for k, v in df.items()}


def length_damp(page: Counter, typical: float) -> float:
    """把「整篇文章貼上去」的投影片壓下來。

    含括度的分母只有逐字稿那一側，所以字越多的投影片越容易包住任何東西。
    實測這份 PPTX 的中位數是每頁 422 字，但第 55 頁有 4,381 字（10.4 倍，
    是一整篇文章貼上去的），結果它一頁吃掉 78 分鐘的逐字稿。

    只罰超過典型長度兩倍的頁，正常長度的頁完全不受影響。
    """
    n = sum(page.values())
    if not n or not typical:
        return 1.0
    return min(1.0, ((2.0 * typical) / n) ** 0.5)


def similarity(page: Counter, chunk: Counter, idf: dict = None,
               damp: float = 1.0) -> float:
    """含括度：這塊逐字稿有多少內容（IDF 加權）出現在這頁投影片上。

    刻意不用餘弦。餘弦要求兩邊「整體相像」，但這裡的關係是**包含**——
    一分鐘的逐字稿本來就只會講到那頁的一小部分，兩邊長度天差地遠，
    餘弦會把正確的配對壓得很低。只用逐字稿那邊當分母，長投影片不吃虧。

    分母不含投影片長度，所以要靠 IDF 擋住「一頁塞滿泛泛字詞就跟誰都像」。
    實測三種算法（舊式交集、餘弦、含括）在已知答案的 21 個時間點上
    單塊命中率分別是 67%／62%／62%，差距不大——真正在壓制錯誤的是
    後面的單調 DP，不是這個分數本身。選含括是因為它的尺度穩定（0–1）
    而且跟「逐字稿是投影片主題的子集」這個真實關係一致。
    """
    if not page or not chunk:
        return 0.0
    w = idf or {}
    num = sum(min(page[k], v) * (w.get(k, 1.0) ** 2)
              for k, v in chunk.items() if k in page)
    den = sum(v * (w.get(k, 1.0) ** 2) for k, v in chunk.items())
    return (num / den if den else 0.0) * damp


# ── 讀投影片 ──────────────────────────────────────────────────────────
def _pdf_pages(path: Path):
    from pypdf import PdfReader
    out = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        out.append((i, _WS.sub(" ", page.extract_text() or "").strip()))
    return out


def _pptx_pages(path: Path):
    """PPTX 連講者備忘稿一起抽——那通常比投影片上的字更接近老師講的話。"""
    from pptx import Presentation
    out = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.extend(c.text for c in row.cells)
        try:
            if slide.has_notes_slide:
                parts.append(slide.notes_slide.notes_text_frame.text)
        except Exception:
            pass
        out.append((i, _WS.sub(" ", " ".join(parts)).strip()))
    return out


def read_pages(path):
    """回傳 [(頁碼, 文字)]。抽不出文字的頁（純圖、掃描檔）文字會是空字串。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf_pages(path)
    if ext == ".pptx":
        return _pptx_pages(path)
    raise ValueError("不支援的格式：%s（只吃 %s）" % (ext, "、".join(SUPPORTED)))


def list_materials(materials_dir=None):
    """列出教材資料夾裡的檔案。"""
    d = Path(materials_dir or config.MATERIALS_DIR)
    if not d.exists():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            st = f.stat()
            out.append({"name": f.name, "path": str(f), "bytes": st.st_size,
                        "id": hashlib.sha1(str(f).encode()).hexdigest()[:12]})
    return out


# ── 對齊 ──────────────────────────────────────────────────────────────
def chunk_segments(segments, per=CHUNK_SENTENCES):
    out = []
    for i in range(0, len(segments), per):
        grp = segments[i:i + per]
        if not grp:
            continue
        text = "".join(g["text"] for g in grp)
        out.append({"start_s": grp[0]["start_s"], "end_s": grp[-1]["end_s"],
                    "text": text})
    return out


def align(pages, chunks, threshold: float = MATCH_THRESHOLD):
    """單調對齊，回傳每一塊的頁碼（對不上的是 None）與相似度。

    允許「跳過」：相似度低於門檻的塊不指派給任何頁。沒有這條的話，
    老師離題或在講別份教材的那 50 分鐘會被硬塞到某一頁上，
    看起來像對齊成功但其實全錯。
    """
    pf = [features(t) for _, t in pages]
    cf = [features(c["text"]) for c in chunks]
    N, M = len(pages), len(chunks)
    if not N or not M:
        return [], []

    # IDF 從投影片各頁算：每一頁都有的東西（頁尾、母片標題）不該有鑑別度
    idf = idf_of(pf)
    sizes = sorted(sum(f.values()) for f in pf) or [1]
    typical = sizes[len(sizes) // 2] or 1
    damps = [length_damp(f, typical) for f in pf]
    S = [[similarity(pf[j], cf[i], idf, damps[j]) for j in range(N)]
         for i in range(M)]
    gain = [[max(S[i][j] - threshold, 0.0) for j in range(N)] for i in range(M)]

    NEG = float("-inf")
    dp = [[NEG] * N for _ in range(M)]
    bk = [[0] * N for _ in range(M)]
    for j in range(N):
        dp[0][j] = gain[0][j]
    for i in range(1, M):
        best, arg = NEG, 0
        for j in range(N):
            if dp[i - 1][j] > best:
                best, arg = dp[i - 1][j], j
            dp[i][j] = best + gain[i][j]
            bk[i][j] = arg
    j = max(range(N), key=lambda x: dp[M - 1][x])
    path = [0] * M
    for i in range(M - 1, -1, -1):
        path[i] = j
        j = bk[i][j]

    assigned = [pages[j][0] if gain[i][j] > 0 else None
                for i, j in enumerate(path)]
    scores = [S[i][j] for i, j in enumerate(path)]
    return assigned, scores


def align_material(pages, segments, threshold: float = MATCH_THRESHOLD):
    """回傳這份教材跟這段逐字稿的對齊結果與整體信心。"""
    chunks = chunk_segments(segments)
    assigned, scores = align(pages, chunks, threshold)
    by_page = {}
    for ch, pg, sc in zip(chunks, assigned, scores):
        if pg is None:
            continue
        e = by_page.setdefault(pg, {"page": pg, "start_s": ch["start_s"],
                                    "end_s": ch["end_s"], "n": 0, "score": 0.0})
        e["start_s"] = min(e["start_s"], ch["start_s"])
        e["end_s"] = max(e["end_s"], ch["end_s"])
        e["n"] += 1
        e["score"] += sc
    for e in by_page.values():
        e["score"] = round(e["score"] / e["n"], 3)
    matched = sum(1 for a in assigned if a is not None)
    text_pages = sum(1 for _, t in pages if len(t) > 20)
    return {
        "pages": len(pages),
        "pages_with_text": text_pages,
        "chunks": len(chunks),
        "matched_chunks": matched,
        "coverage": round(matched / len(chunks), 3) if chunks else 0.0,
        "confidence": round(
            sum(e["score"] * e["n"] for e in by_page.values()) / matched, 3)
        if matched else 0.0,
        "ranges": sorted(by_page.values(), key=lambda e: e["page"]),
    }


def align_all(segments, materials_dir=None, threshold=MATCH_THRESHOLD):
    """把逐字稿對到「所有教材的所有頁」，回傳每一塊的歸屬。

    不要先問「這堂用哪一份教材」再對齊——一堂課同時用到兩份是常態
    （上半堂接續前一週的投影片，下半堂換新的）。實測 2026-09-21 那堂就是
    前 51 分鐘在講前一份的蘋果供應鏈，後面才換到 SDG/ESG 那份；
    硬要選一份的話，不管選哪份都有一半的課對不上。

    做法：每一份各自跑單調對齊（頁碼在自己那份內不回頭），
    再讓每一塊挑分數最高的那一份。對不上任何一頁的塊標成 None。
    """
    mats = []
    for m in list_materials(materials_dir):
        try:
            mats.append((m, read_pages(m["path"])))
        except Exception as e:
            log.warning("讀不到教材 %s：%s", m["name"], e)
    chunks = chunk_segments(segments)
    if not mats or not chunks:
        return {"chunks": chunks, "picks": [None] * len(chunks), "materials": []}

    per_mat = []
    for m, pages in mats:
        assigned, scores = align(pages, chunks, threshold)
        per_mat.append((m, assigned, scores))

    picks = []
    for i in range(len(chunks)):
        best = None
        for m, assigned, scores in per_mat:
            if assigned[i] is None:
                continue
            if best is None or scores[i] > best["score"]:
                best = {"material": m["name"], "material_id": m["id"],
                        "page": assigned[i], "score": round(scores[i], 3)}
        picks.append(best)

    used = {}
    for ch, pk in zip(chunks, picks):
        if not pk:
            continue
        u = used.setdefault(pk["material_id"],
                            {"material": pk["material"], "id": pk["material_id"],
                             "chunks": 0, "pages": set(), "score": 0.0,
                             "start_s": ch["start_s"], "end_s": ch["end_s"]})
        u["chunks"] += 1
        u["pages"].add(pk["page"])
        u["score"] += pk["score"]
        u["start_s"] = min(u["start_s"], ch["start_s"])
        u["end_s"] = max(u["end_s"], ch["end_s"])
    summary = []
    for u in used.values():
        summary.append({"material": u["material"], "id": u["id"],
                        "chunks": u["chunks"], "pages": len(u["pages"]),
                        "share": round(u["chunks"] / len(chunks), 3),
                        "score": round(u["score"] / u["chunks"], 3),
                        "start_s": u["start_s"], "end_s": u["end_s"]})
    summary.sort(key=lambda x: x["chunks"], reverse=True)
    return {"chunks": chunks, "picks": picks, "materials": summary,
            "matched": sum(1 for p in picks if p),
            "coverage": round(sum(1 for p in picks if p) / len(chunks), 3)}


def best_material(segments, materials_dir=None, threshold=MATCH_THRESHOLD):
    """這段逐字稿最可能對應哪一份教材。回傳照分數排序的候選清單。

    「挑哪一份」和「對到第幾頁」是兩個不同的問題，要用不同的 IDF：

    - 挑哪一份：IDF 要**跨所有教材**一起算。同一門課不同週的投影片用詞
      高度重疊，只有那份獨有的詞才有鑑別度。用單份內部的 IDF 去比，
      兩份的分數會幾乎一樣（實測中位數 0.095 對 0.097，完全分不開）。
    - 對到第幾頁：IDF 用**那一份內部**的，把頁尾與母片標題壓掉。

    同一堂課用到兩份教材是常態（上半堂講前一份的延續），所以回傳的是
    排序後的清單，不是單一答案。
    """
    mats = []
    for m in list_materials(materials_dir):
        try:
            mats.append((m, read_pages(m["path"])))
        except Exception as e:
            log.warning("讀不到教材 %s：%s", m["name"], e)
    if not mats:
        return []

    chunks = chunk_segments(segments)
    cf = [features(c["text"]) for c in chunks]
    pooled = [features(t) for _, pages in mats for _, t in pages]
    gidf = idf_of(pooled)

    out = []
    for m, pages in mats:
        pf = [features(t) for _, t in pages]
        # 跨教材 IDF：這份「有沒有被講到」
        sizes = sorted(sum(f.values()) for f in pf) or [1]
        typical = sizes[len(sizes) // 2] or 1
        dm = [length_damp(f, typical) for f in pf]
        hits = [max((similarity(p, c, gidf, d) for p, d in zip(pf, dm)),
                    default=0.0) for c in cf]
        strong = [h for h in hits if h >= threshold]
        # 單份內 IDF：對到第幾頁、涵蓋多少
        res = align_material(pages, segments, threshold)
        out.append(dict(m, **{
            "pages": res["pages"], "pages_with_text": res["pages_with_text"],
            "matched_pages": len(res["ranges"]),
            "coverage": res["coverage"],
            "confidence": round(sum(strong) / len(strong), 3) if strong else 0.0,
            "hit_ratio": round(len(strong) / len(hits), 3) if hits else 0.0,
        }))
    out.sort(key=lambda x: x["hit_ratio"] * x["confidence"], reverse=True)
    return out
