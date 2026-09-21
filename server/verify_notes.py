"""摘要的接地檢查（grounding check）。

**這不是事實查核。** 它不知道老師講的對不對，也抓不到「ASR 聽錯成另一個
合理的詞」——那種錯誤在逐字稿裡看起來完全正常。

它只回答一個問題：**摘要裡的內容，在逐字稿裡找得到依據嗎？**
找不到依據的句子有兩種可能：模型自己補了知識（幻覺），或是它做了
逐字稿沒支持的推論。兩種都值得標出來讓使用者自己判斷。

另外用 ASR 的 avg_logprob 標出辨識信心低的時間段——那些地方的逐字稿
本身就不可靠，由它產生的摘要自然也要打折。
"""
from __future__ import annotations

import re

# avg_logprob 低於此值的 segment 視為辨識信心不足。
# Whisper 系模型的 avg_logprob 通常落在 -0.1 ~ -1.0，越接近 0 越有把握。
LOW_CONFIDENCE = -0.85

# 停用詞不列入接地比對：它們到處都有，比對它們沒有鑑別力
_STOP = set("的了是在有和與及對不就都也很我你他這那個們會要可以所以因為"
            "但是如果然後還有一個什麼怎麼樣之類等等他們我們你們")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[一-鿿]{2,}")


def _tokens(text: str):
    """抽出可比對的詞：英文字串，以及 2 字以上的中文片段。"""
    out = []
    for m in _TOKEN_RE.finditer(text or ""):
        t = m.group(0)
        if t.isascii():
            out.append(t.lower())
        else:
            # 中文用 2-gram 切，避免斷詞器依賴
            for i in range(len(t) - 1):
                g = t[i:i + 2]
                if g not in _STOP:
                    out.append(g)
    return out


def _grounded_ratio(claim: str, haystack: str) -> float:
    """claim 裡有多少比例的詞能在逐字稿中找到。"""
    toks = _tokens(claim)
    if not toks:
        return 1.0
    hit = sum(1 for t in toks if t in haystack)
    return hit / len(toks)


# 字面重疊比對在實測上不堪用：門檻 0.35 會誤報 33% 的正常要點，
# 降到 0.20 才沒有誤報，但那時只抓得到 1/3 的幻覺。原因是摘要被要求
# 「用自己的話重新組織」，改寫後字面必然有差距；而捏造的句子用的詞
# （公司名、「通常」「成長」）逐字稿裡又都有，重疊率反而很高。
#
# 改用精確得多的訊號：**數字與專有名詞**。
# 幻覺幾乎都會冒出逐字稿沒出現過的具體數字（「70%」「2024 年」）
# 或英文專名（「IEEE」「Scrum」）；合理的改寫則很少憑空生出這些。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|年|月|日|倍|萬|億|元|人|次|個月)?")
_PROPER_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#-]{1,}")


def _novel_facts(claim: str, hay: str):
    """回傳 claim 裡出現、但逐字稿中找不到的數字與英文專名。"""
    novel = []
    for m in _NUM_RE.finditer(claim or ""):
        tok = m.group(0).strip()
        digits = re.sub(r"\D", "", tok)
        if len(digits) < 2:          # 個位數太容易巧合，不算
            continue
        if digits not in hay:
            novel.append(tok)
    for m in _PROPER_RE.finditer(claim or ""):
        t = m.group(0)
        if len(t) < 3:
            continue
        if t.lower() not in hay:
            novel.append(t)
    return novel


def check_claims(claims, transcript: str, threshold: float = 0.20):
    """回傳 [(claim, ratio, ok, novel)]。

    ok=False 的判定條件是「字面重疊極低」**或**「冒出逐字稿沒有的
    數字／專有名詞」。後者才是主力訊號；前者門檻壓得很低，只當保險。
    """
    hay = (transcript or "").lower()
    out = []
    for c in claims:
        r = _grounded_ratio(c, hay)
        novel = _novel_facts(c, hay)
        ok = (r >= threshold) and not novel
        out.append((c, round(r, 2), ok, novel))
    return out


def low_confidence_spans(segments, limit: int = 5):
    """挑出辨識信心最低的幾段，附時間戳供使用者回去聽錄音確認。"""
    scored = [s for s in segments
              if s.get("avg_logprob") is not None
              and s["avg_logprob"] < LOW_CONFIDENCE]
    scored.sort(key=lambda s: s["avg_logprob"])
    return [{"start_s": s.get("start_s", s.get("start", 0.0)),
             "text": s.get("text", "")[:60],
             "avg_logprob": round(s["avg_logprob"], 2)}
            for s in scored[:limit]]


def summarize_check(notes_claims, transcript, segments):
    """把兩種檢查合成一份報告。"""
    checked = check_claims(notes_claims, transcript)
    ungrounded = [{"claim": c, "grounded": r, "novel": nv}
                  for c, r, ok, nv in checked if not ok][:4]
    low = low_confidence_spans(segments, limit=4)
    return {
        "checked": len(checked),
        "ungrounded": ungrounded,
        "low_confidence": low,
        "note": ("主要檢查摘要有沒有冒出逐字稿裡沒有的數字或專有名詞。"
                 "抓不到「老師講錯」與「ASR 聽成另一個合理的詞」——"
                 "那兩種從逐字稿本身看不出來，只能回去聽錄音。"),
    }
