"""用課程術語表自動修正 ASR 的近似音錯誤。

實測看到的錯誤（生產與作業管理、企業電腦網路兩堂課）：

    變動成本 → 電動成本      biàn → diàn
    原物料   → 原木料        wù   → mù
    加值     → 加持          zhí  → chí
    直接人工 → 職業員工
    雲端服務 → 原端服務      yún  → yuán
    電算中心 → 電傳中心／店家中心／電訊中心（同一堂三種錯法）

這些詞在術語表裡都有，錯的版本讀音又很接近——靠拼音比對就能抓回來。
`initial_prompt` 幫得了一部分（實測術語命中 +18~46%），但擋不住全部，
這層是補網。

**會改錯。** 所以每一次修正都記錄下來（corrections 欄位），
並且只在拼音高度相似時才動手；寧可漏改也不要把對的改成錯的。
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_lazy = {}


def _pinyin_of(text: str):
    """回傳無聲調拼音串列。pypinyin 缺席時回傳 None（整個修正功能停用）。"""
    if "fn" not in _lazy:
        try:
            from pypinyin import lazy_pinyin
            _lazy["fn"] = lazy_pinyin
        except ImportError:
            log.warning("未安裝 pypinyin，近似音自動修正停用")
            _lazy["fn"] = None
    fn = _lazy["fn"]
    if fn is None:
        return None
    return fn(text)


# 聲母／韻母容易互相聽錯的組合。中文 ASR 的典型混淆。
_CONFUSABLE = [
    ("b", "d"), ("b", "p"), ("d", "t"), ("g", "k"), ("j", "q"), ("j", "zh"),
    ("z", "zh"), ("c", "ch"), ("s", "sh"), ("z", "j"), ("l", "n"), ("f", "h"),
    ("r", "l"), ("x", "sh"), ("ch", "q"), ("zh", "j"),
]


def _syl_similar(a: str, b: str) -> float:
    """單一音節的相似度 0~1。"""
    if a == b:
        return 1.0
    # 拆成聲母 + 韻母（粗略：取前 1~2 個字母當聲母）
    def split(s):
        for ini in ("zh", "ch", "sh"):
            if s.startswith(ini):
                return ini, s[2:]
        if s and s[0] in "bpmfdtnlgkhjqxrzcsyw":
            return s[0], s[1:]
        return "", s

    ia, fa = split(a)
    ib, fb = split(b)
    score = 0.0
    if fa == fb:                       # 韻母相同是主要依據
        score += 0.65
    elif fa and fb and (fa.endswith(fb) or fb.endswith(fa)):
        score += 0.35
    if ia == ib:
        score += 0.35
    elif {ia, ib} in [set(p) for p in _CONFUSABLE]:
        score += 0.28                  # 易混聲母：算幾乎相同
    return min(1.0, score)


def _phon_similar(a: str, b: str) -> float:
    """兩個詞的整體讀音相似度。長度不同直接回 0。"""
    pa, pb = _pinyin_of(a), _pinyin_of(b)
    if pa is None or pb is None or len(pa) != len(pb):
        return 0.0
    if not pa:
        return 0.0
    return sum(_syl_similar(x, y) for x, y in zip(pa, pb)) / len(pa)


class TermFixer:
    """針對一門課的術語表建立修正器。"""

    # 門檻訂高：只改讀音幾乎一樣的。實測 0.85 能抓到
    # 變動→電動(0.88)、原物→原木(0.88)，又不會誤傷無關的詞。
    THRESHOLD = 0.85

    def __init__(self, glossary, threshold: float = None):
        self.threshold = self.THRESHOLD if threshold is None else threshold
        # 只處理中文術語；英文術語靠 initial_prompt 與 LLM 處理
        self.terms = [g for g in (glossary or [])
                      if g and not g.isascii() and 2 <= len(g) <= 6]
        self._by_len = {}
        for t in self.terms:
            self._by_len.setdefault(len(t), []).append(t)
        self.enabled = bool(self.terms) and _pinyin_of("測") is not None

    # 出現這麼多次以上的字串視為「老師真的在講這個詞」，不修正。
    # 這條規則是實測逼出來的：供應鏈那堂課「價值」出現 12 次，
    # 讀音跟術語表裡的「加值」極像（jià zhí / jiā zhí），
    # 沒有這條保護就會把 12 個正確的「價值」全部改成意思不同的「加值」。
    # 真正的 ASR 錯誤通常是零星的，正確的詞則反覆出現。
    MAX_OCCURRENCES = 3

    def fix(self, text: str):
        """回傳 (修正後文字, [(原文, 修正後, 相似度)])。"""
        if not self.enabled or not text:
            return text, []
        out = text
        corrections = []
        term_set = set(self.terms)
        for n, terms in self._by_len.items():
            if len(out) < n:
                continue
            # 逐一比對所有長度相同的子字串
            i = 0
            buf = []
            while i <= len(out) - n:
                chunk = out[i:i + n]
                if not _is_cjk(chunk):
                    buf.append(out[i])
                    i += 1
                    continue
                best, score = None, 0.0
                if chunk in term_set:
                    # 本身就是術語表裡的詞，絕對不動
                    buf.append(out[i])
                    i += 1
                    continue
                if text.count(chunk) > self.MAX_OCCURRENCES:
                    # 反覆出現＝老師真的在講這個詞，不是 ASR 的零星失誤
                    buf.append(out[i])
                    i += 1
                    continue
                for t in terms:
                    sc = _phon_similar(chunk, t)
                    if sc > score:
                        best, score = t, sc
                if best and score >= self.threshold:
                    corrections.append((chunk, best, round(score, 2)))
                    buf.append(best)
                    i += n
                else:
                    buf.append(out[i])
                    i += 1
            buf.append(out[i:])
            out = "".join(buf)
        return out, corrections


def _is_cjk(s: str) -> bool:
    return all("一" <= c <= "鿿" for c in s)


_CACHE = {}


def for_course(course):
    """每門課一個修正器，快取起來避免重複建。"""
    key = (course.id, tuple(course.glossary))
    if key not in _CACHE:
        _CACHE[key] = TermFixer(course.glossary)
    return _CACHE[key]
