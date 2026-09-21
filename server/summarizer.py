"""llama-server 客戶端與 prompt 組裝（規格 §6.3、§6.4）。"""
from __future__ import annotations

import json
import logging
import re

import httpx

from . import config
from .llm_manager import LLMUnavailable

log = logging.getLogger(__name__)

# 術語與字體的規則三個 prompt 共用，改一處就好
TERM_RULE = """- 一律使用繁體中文（台灣用語），絕對不要出現簡體字
- 英文專業術語**保留英文原文**，第一次出現時在後面用全形括號附上中文，
  例如：packet（封包）、firewall（防火牆）、virtualization（虛擬化）。
  同一個術語第二次以後只寫英文，不要再附中文
- 不要把英文術語整個換成中文——使用者要的是英文，中文只是輔助"""

SECTION_SYSTEM_TMPL = """你是課堂筆記助理。使用者會給你一段課程逐字稿，請整理成**有結構的**重點筆記。

# 結構（最重要的一件事）

不要輸出一長串平鋪的要點。二十幾個同一層級的句子不是筆記，那只是把
逐字稿剁碎。要寫成人看得懂的層次：

1. `summary`：一到兩句話說**這一段在講什麼、主軸是什麼**。
   讀的人只看這兩句就要知道這段的脈絡。
2. `groups`：把內容分成幾個子題，每個子題一個 `heading` 加幾個 `points`。
   子題要照著內容本身的結構分（流程的先後、概念的分類、制度的各個環節），
   不要照逐字稿出現的順序硬切。

例如一段在講「品牌商如何管理供應鏈」的內容，應該分成
「移工的保障機制」「行為準則與評分」「稽核與觸發調查的方式」
「採購風險分散」這幾個子題，而不是把二十幾件事並排。

**數字和例子要掛在它支撐的論點裡**，不要自己獨立成一點。
寫「認證 470 家仲介，避免落入人蛇集團」，
不要分成「認證 470 家仲介」和「避免人蛇集團」兩點。

內容很短、真的只有一個子題時，就給一個 `heading` 是空字串的 group，
不要為了湊數硬分。

# 內容要求

- 逐字稿由語音辨識產生，**一定會有同音或近似音的錯字**。
  請依上下文與下方的課程術語表主動判斷並改正，例如：
  「電動成本」應是「變動成本」、「原木料」應是「原物料」、
  「職業員工」應是「直接人工」、「原端服務」應是「雲端服務」。
  不確定的地方以術語表的正式寫法為準，不要照抄明顯錯誤的字
- 口語贅詞（對不對、是不是、這個那個）一律略去
- **用你自己的話重新組織，不要照抄逐字稿的句子**
- **同一件事只寫一次**。兩個要點在講同一個概念就合併
- 出現專有名詞、英文縮寫時，第一次用括號簡短說明它是什麼
""" + TERM_RULE + """
- 只輸出這段內容真正講到的資訊，不要補充你自己的知識
- 標題要具體，寫出這段的**重點是什麼**，不要只寫主題名詞。
  例如寫「IaaS 與 SaaS 的責任分界」而不是「雲端服務」
- 若這段只是前一段的延續、沒有新東西，標題寫「（延續前段）」，
  只寫真正新增的部分

僅輸出 JSON，不要有任何前後文字或 markdown 圍欄：
{"title": "不超過 20 字的段落標題",
 "summary": "一到兩句話：這段在講什麼",
 "groups": [{"heading": "子題", "points": ["要點一", "要點二"]}]}

這段逐字稿有 %d 字，全部的 points 加起來請寫 %d 則左右，每則不超過 60 字。
**不要只挑幾個重點草草帶過**——使用者要拿這份當上課筆記，漏掉的內容
他就永遠不知道老師講過。寧可多寫幾點，也不要把十五分鐘的課濃縮成三句話。"""

FINAL_SYSTEM = """你是課堂筆記助理。使用者會給你一堂課所有段落的重點筆記，請合成一份完整課堂筆記。

要求：
- 輸入已是整理過的段落摘要，不是逐字稿；不要再重複逐條抄寫，要做整體歸納
""" + TERM_RULE + """
- 不要補充輸入中沒有的知識

僅輸出 JSON，不要有任何前後文字或 markdown 圍欄：
{"overview": ["整體概述句一", "整體概述句二"], "open_questions": ["待釐清一"]}

overview 為 6 到 10 句整體概述，要涵蓋各段落的重點，不要只講開頭兩段。
open_questions 從各段落中挑出講述不完整或需要複習的點，最多 5 則；沒有就給空陣列。"""

HANDCOPY_SYSTEM_TMPL = """你是課堂筆記助理。使用者正在課堂上邊聽邊手寫筆記，下課要立刻交出去。
他會把你產出的內容**用手抄到紙上**，所以必須夠短、夠有結構。

要求：
- 總量約 %d 字。太短的筆記沒有用——漏掉的內容使用者永遠不會知道老師講過
- 分成 4 到 7 個主題，每個主題底下 2 到 5 個要點
- 每個要點一句話講完，不要寫成段落
- **輸入裡提到的重點都要涵蓋到**，不要只挑幾個講
""" + TERM_RULE + """
- 只寫這堂課真正講到的，不要補充你自己的知識
- 逐字稿有同音錯字，請依上下文與術語表改正後再寫進筆記
- 老師特別強調或說會考的，放在最前面

僅輸出 JSON，不要有任何前後文字或 markdown 圍欄：
{"topics": [{"title": "主題", "points": ["要點一", "要點二"]}], "exam": ["會考的重點"]}

exam 放老師明示會考或特別強調的；沒有就給空陣列。"""

WORKSHEET_RECORD_TMPL = """你是課堂筆記助理。使用者正在課堂上邊聽邊手寫一份學習單，下課要立刻交出去。
他會把你產出的內容**用手抄到紙上**，並且自己挑要抄哪幾則。

這次只寫下面這幾欄（記錄整堂課實際講了什麼）：
%s

最重要的兩件事：

一、**涵蓋整堂課**。輸入裡有 %d 個段落，每一個段落都要有對應的內容出現在
筆記裡。不可以只寫其中一個案例、一個章節就結束——使用者看不到逐字稿，
你漏掉的部分他永遠不會知道老師講過。寫之前先掃過所有段落，照時間順序寫。

二、**要有細節，也要有名詞解釋**。
- 出現專有名詞、英文縮寫、協定名稱、數字規格時，用一則專門解釋它是什麼、
  用來做什麼，例如「ARP（位址解析協定）：用 IP 查對應的 MAC 位址」
- 不要只寫結論式的標題（「講了路由」），要寫出老師實際講的內容
  （「路由表比對時採最長前綴符合，相同目的地有多筆時選遮罩最長的那筆」）
- 老師舉的例子、數字、比較、步驟順序都要寫進去

其他要求：
- 總量約 %d 字
- 每一則一到兩句，不要寫成段落；使用者是用手抄的
- **只能寫這堂課真正講到的內容**，不要補充你自己的知識
- 逐字稿有同音錯字，請依上下文與術語表改正後再寫進筆記
""" + TERM_RULE + """

僅輸出 JSON，不要有任何前後文字或 markdown 圍欄：
{"fields": {"欄位id": ["<這裡放你寫的第一則，不要照抄這行>", "<第二則>"]},
 "exam": ["<老師說會考的重點>"]}

（角括號裡是說明，不是內容。陣列裡放的必須是你真的寫出來的句子。）

fields 必須剛好包含上面列出的每一個欄位 id；整份寫完就停，不要重寫一遍。
exam 放老師明示會考或特別強調的；沒有就給空陣列。"""


WORKSHEET_REFLECT_TMPL = """你在幫一位**研究所學生**寫課堂學習單的其中一欄。他會挑幾則手抄到紙上交出去。

# 這一欄是「%s」

這一欄要做的事（**這段是這一欄的定義，優先於下面所有通則**）：
%s

要 %d 到 %d 則。上面那段說要做什麼就做什麼——如果它說不要分析技術，
那整欄就一則技術分析都不能有；如果它說要寫你自己，那每一則都要有你自己。

# 通則

- **每一則錨定一個具體的東西**：課堂上真的出現過的名詞、數字、例子、對比或步驟。
  沒有錨點的句子（「這讓我覺得很有收穫」「原來技術一直在進步」）一律不要寫。
- **不要只是把老師講的換句話說。** 他自己抄逐字稿就好了。
- **研究所的水準。** 不要寫「真的很驚人」「我覺得好方便」「原來如此」。
  不確定就寫成問題（「如果…那…是不是就…」），不要假裝有結論。
- **每一則的開頭都要不一樣。** 同一種句型整欄最多出現一次。特別不要用
  「我原本以為X，現在知道Y」這個模板，那是小學生的寫法。
- **每一則都是完整的句子，要有句號。** 不要寫成「X 的 Y，取決於 Z」這種
  沒講完的名詞片語——抄到紙上看起來像半句話。
- **句式要混著用**，整欄至少出現三種不同的形狀。可用的寫法包括：
  直接下一個判斷、指出成立的前提或條件、拿兩件事做對比、
  指出課堂漏掉的一步、提出一個追問。
  **問句最多佔三分之一**——整欄都在發問讀起來像在出考題，不像在想事情。
  也不要每一則都用「如果…」「若…」「當…」開場，或每一則都是
  「某某的某某，依賴／取決於／反映某某」這同一個骨架。
- 每一則扣不同的內容，不要整欄都在講同一組概念。
- 每一則 %d 到 %d 字，寫成完整的句子，他是用手抄的。

%s

僅輸出 JSON，不要有任何前後文字或 markdown 圍欄：
{"fields": {"%s": ["<這裡放你寫的第一則，不要照抄這行>", "<第二則>"]}}

（角括號裡是說明，不是內容。陣列裡放的必須是你真的寫出來的句子。）"""


WORKSHEET_AVOID_TMPL = """# 不可以重複

下面這些已經寫在學習單的別欄了。你這一欄的每一則都必須**講不同的事**——
不是換句話說，是換一件事講。只要主題跟下面任何一則重疊就重挑一個：
%s"""


WORKSHEET_RETRY_TMPL = """剛才那次有下面的問題：
%s

重寫這一欄。每一則換一種說法開場、換一個句型，長度控制在 %d 字以內。"""


WORKSHEET_CHARS_DEFAULT = (40, 70)


def worksheet_chars(sec) -> tuple:
    """這一欄每則的字數範圍。課程設定沒寫就用預設。"""
    chars = sec.get("chars")
    if chars:
        lo, hi = int(chars[0]), int(chars[1])
        return max(10, lo), max(lo + 10, hi)
    return WORKSHEET_CHARS_DEFAULT


def worksheet_counts(sec, n_source: int = 0) -> tuple:
    """這一欄要產幾則：(最少, 最多)。

    pick 是「使用者實際只會抄幾則」，其餘是給他挑的備選，所以上限要比
    pick 大一截。pick=0 的欄位（例如「今天上課內容」）是整堂的紀錄，
    給得越多他越有得挑。

    n_source 是這次能用的素材則數。**下限一定要跟著素材縮**：設定檔寫
    最少 18 則、素材只有 11 則時，模型會照著湊數量，湊出來的是編的——
    實測 8 分鐘的錄音硬生出 34 則，裡面的「4 vCPU 8GB RAM」「100 GB 磁碟」
    「Google Meet 最多 100 人」全部是逐字稿裡沒有的。寧可短，不可以編。
    """
    count = sec.get("count")
    if count:
        lo, hi = int(count[0]), int(count[1])
        lo, hi = max(1, lo), max(lo + 1, hi)
    else:
        pick = int(sec.get("pick") or 0)
        lo, hi = (6, 12) if not pick else (pick + 1, pick + 4)
    if n_source:
        if sec.get("pick") or 0:
            # 感受型：素材少的時候幾則就會開始互相重複或開始編
            avail = max(3, n_source // 2)
        else:
            # 記錄型：一則素材最多撐出 1.2 則筆記（拆出名詞解釋算合理，
            # 再多就是掰的）
            avail = max(3, int(n_source * 1.2))
        hi = min(hi, avail)
        lo = min(lo, max(2, int(avail * 0.6)))
        lo = min(lo, hi - 1 if hi > 1 else 1)
    return lo, hi


_ARTIFACT_RE = re.compile(r"^[\s\"'`\]\}\[\{,]+|[\s\"'`\]\}\[\{,]+$")


def strip_json_artifacts(text: str) -> str:
    """把漏進文字裡的 JSON 標點清掉。

    模型的輸出被截斷再救回來時，尾巴偶爾會黏著 `',` 或 `"]}`；那些字元
    會原封不動抄到紙上。只清頭尾，不動句子中間，以免弄壞內容本身。
    """
    t = _ARTIFACT_RE.sub("", (text or "").strip())
    # 清掉標點後可能露出被截斷的句尾逗號
    return t.rstrip("，、").strip()


def _strip_title_echo(point: str, title: str) -> str:
    """去掉要點開頭重複的欄位名，例如「今天上課內容：軟體工程的核心…」。"""
    t = strip_json_artifacts(point)
    for sep in ("：", ":", "、", "-", "—"):
        pre = title + sep
        if t.startswith(pre):
            return t[len(pre):].strip()
    return t


def worksheet_groups(course):
    """把欄位分成「記錄型」與「感受型」兩組。

    分兩次呼叫模型，記錄型那一欄才拿得到完整的輸出預算。擠在同一次時
    max_tokens 要同時養活四欄，上課內容永遠是被犧牲的那一欄——實測整堂
    2.5 小時只寫出 8 則，而且全部集中在同一個案例。
    """
    record = [x for x in course.handcopy_sections if not (x.get("pick") or 0)]
    reflect = [x for x in course.handcopy_sections if x.get("pick") or 0]
    return record, reflect


_PAREN_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*|[一-鿿]{2,8})\s*[（(]([^）)]{3,60})[）)]")
# 譯名只會是單純的英文片語；帶斜線、數字或版本號的多半是模型自己補的產品名
_GLOSS_RE = re.compile(r"^[A-Za-z][A-Za-z \-]*$")


def _gloss_expansions(claim: str, haystack: str, ok_words) -> set:
    """括號裡的英文譯名／全稱不算「憑空冒出來的專名」。

    使用者要的就是名詞解釋。「IaaS（Infrastructure as a Service）」的
    Infrastructure、「虛擬機器（Virtual Machine）」的 Machine 本來就不會
    出現在逐字稿裡——老師講的是縮寫或中文。

    放行的條件有兩個，兩個都要成立：括號前面那個詞在素材裡查得到，
    而且括號內容是單純的英文片語。所以「前端網頁（React/Vue.js）」這種
    掰出來的產品名還是會被擋下來——它有斜線和版本點。
    """
    out = set()
    hay = (haystack or "").lower()
    for m in _PAREN_RE.finditer(claim or ""):
        term, expan = m.group(1), m.group(2).strip()
        if term.lower() not in hay and term.lower() not in ok_words:
            continue
        if not _GLOSS_RE.match(expan):
            continue
        words = [w for w in re.split(r"[\s-]+", expan) if w]
        # 注意：中文字的 isalpha() 也是 True，要先確認是 ASCII 才算縮寫
        if term[0].isascii() and term[0].isalpha() and len(words) >= 2:
            # 英文縮寫：字首要拼得回去
            initials = "".join(w[0] for w in words).lower()
            if not (initials.startswith(term.lower())
                    or term.lower().startswith(initials)):
                continue
        out.update(w.lower() for w in words)
    return out


_PLACEHOLDER_RE = re.compile(
    r"^(第[一二三四五六七八九十\d]+則|一則|另一則|要點[一二三四五六七八九十\d]*|"
    r"[<＜].*[>＞]|\.{2,}|…+)$")


_END_PUNCT = ("。", "！", "？", "；", ".", "!", "?", ":", "：")


def ensure_sentence(text: str) -> str:
    """句子沒收尾就補句號。

    這種事不值得多跑一次推論——模型偶爾整批都不寫句號（實測 40 則
    一個都沒有），抄到紙上每一行都像沒寫完。直接補掉。
    """
    t = (text or "").strip()
    if t and not t.endswith(_END_PUNCT):
        t += "。"
    return t


def is_placeholder(text: str) -> bool:
    """模型有時會把 prompt 裡的示意字串當成內容照抄。

    實測「今日反思」整欄變成「一則／另一則／第三則／第四則／第五則」——
    平均 3 個字。這種東西抄到紙上交出去等於空白，寧可整欄空著讓使用者
    知道要重按一次，也不要給他五行廢話。
    """
    t = (text or "").strip()
    return (not t) or len(t) < 10 or bool(_PLACEHOLDER_RE.match(t))


_COND_OPEN_RE = re.compile(r"^(如果|若|當|假如|倘若)")
# 句型指紋：只看「這句話用了哪幾類連接詞」，不看內容字。
# 「A 若未 B，可能導致 C」和「D 若未 E，可能造成 F」內容完全不同、開頭
# 四個字也不同，但骨架一樣——實測「想法」八則有七則是這同一個骨架。
# 用類別而不是精確的詞，才不會因為「導致」換成「造成」就算不同。
_SHAPE_GROUPS = (
    ("條件", re.compile(r"若|如果|當|假如|倘若|未能|未設|未明確")),
    ("推測", re.compile(r"可能|或許|將會|恐|也許")),
    ("因果", re.compile(r"導致|使得|造成|引發|以致")),
    ("轉折", re.compile(r"但|卻|而非|而是|反而|然而")),
    ("依存", re.compile(r"取決於|依賴|建立在|前提|隱含|反映|意味|來自")),
)


def shape_signature(text: str):
    """回傳這句話用到的句型類別，當作骨架的指紋。"""
    t = text or ""
    return tuple(name for name, rx in _SHAPE_GROUPS if rx.search(t))


_FRAME_RE = re.compile(
    r"我原本(以為|認為|覺得|想)|我本來以為|過去我以為|我一直以為|"
    r"這讓我(覺得|發現|意識|想到)|讓我(覺得|發現)|我才(發現|知道)")
# 模型會替使用者編出他沒說過的個人經歷，那是要交出去的東西
_FABRICATED_PAST_RE = re.compile(
    r"我(去年|前年|上學期|上一學期|大[一二三四]|以前|之前|曾經)[^，。]{0,12}"
    r"(做過|做了|寫過|修過|參加過|實習|專案|經驗)")


def worksheet_quality_issues(points, max_chars: int = 95,
                             repeat_limit: float = 0.34):
    """回傳這一欄要重寫的理由；沒問題就回空陣列。

    prompt 說「不要用同一個句型」模型照樣會用，所以要量出來。三件事：
    開頭重複、同一個心得句型反覆出現、以及句子長到手抄不完。
    """
    issues = []
    if len(points) < 2:
        return issues
    head, n, ratio = repeated_openings(points)
    if ratio > repeat_limit:
        issues.append("有 %d/%d 則以「%s」開頭" % (n, len(points), head))
    frames = [x for x in points if _FRAME_RE.search(x or "")]
    if len(frames) / len(points) > repeat_limit:
        issues.append("有 %d/%d 則用了「我原本以為…這讓我…」這類心得句型，"
                      "整欄讀起來像同一句話換字" % (len(frames), len(points)))
    long_ones = [x for x in points if len(x or "") > max_chars]
    if len(long_ones) > len(points) / 2:
        issues.append("有 %d/%d 則超過 %d 字，使用者是用手抄的，抄不完"
                      % (len(long_ones), len(points), max_chars))
    # 句式也要有變化。實測「想法」那一欄五則全是問句、其中三則還都用
    # 「如果…是不是…」開場，整欄讀起來像在考試出題而不是在想事情。
    qs = [x for x in points if "？" in (x or "") or (x or "").rstrip().endswith("?")]
    if len(qs) / len(points) > 0.4:
        issues.append("有 %d/%d 則是問句，整欄都在發問；要混一些直接下判斷、"
                      "指出條件、做對比的寫法" % (len(qs), len(points)))
    opens = [x for x in points if _COND_OPEN_RE.match((x or "").strip())]
    if len(opens) / len(points) > 0.5:
        issues.append("有 %d/%d 則用「如果／若／當」這種條件句開場，句式太單一"
                      % (len(opens), len(points)))
    # 沒有句號的名詞片語（「其資源分配機制依賴於動態負載平衡系統」）讀起來
    # 像句子沒寫完，抄到紙上更明顯
    # 沒有任何連接詞的句子不算一種骨架（那只是單純的陳述句），
    # 否則整欄都寫得很乾淨反而會被判成單調
    sigs = {}
    for x in points:
        sig = shape_signature(x)
        if sig:
            sigs[sig] = sigs.get(sig, 0) + 1
    if sigs and len(points) >= 4:
        top_sig, top_n = max(sigs.items(), key=lambda kv: kv[1])
        if top_n / len(points) > 0.5:
            issues.append("有 %d/%d 則是「%s」這同一個句型骨架，只有名詞換掉；"
                          "換幾種不同的講法" % (top_n, len(points),
                                              "＋".join(top_sig)))
    unfinished = [x for x in points
                  if not (x or "").rstrip().endswith(("。", "！", "？", ".", "!", "?"))]
    if len(unfinished) / len(points) > 0.4:
        issues.append("有 %d/%d 則沒有句號，寫成沒講完的名詞片語；每一則都要是"
                      "完整的句子" % (len(unfinished), len(points)))
    return issues


def fabricated_past(points):
    """挑出替使用者編造個人經歷的要點。

    實測模型寫出「我去年做學生專案時因資源不足導致模型訓練中斷」——
    它不可能知道這件事。這份是要交給老師的，不能編。
    """
    return [x for x in points if _FABRICATED_PAST_RE.search(x or "")]


def dedupe_against(points, seen, threshold: float = 0.55, ignore=(), keep_min=0):
    """丟掉跟前面欄位講同一件事的要點，回傳 (保留的, 丟掉的)。

    「想法」和「反思」很容易寫成同一批內容——實測七則想法裡有四則
    在反思欄又出現一次，只是換了幾個字。光在 prompt 裡說「不要重複」
    擋不住，要真的比對。用詞彙重疊而不是字串比對，才抓得到換句話說。
    """
    from .verify_notes import _tokens
    # 術語不列入比對：同一堂課的每一欄本來就都在講 IaaS、虛擬機器，
    # 拿術語算重疊會把「講不同事情但用同一批名詞」的要點全部誤殺。
    skip = {str(g).lower() for g in (ignore or ())}

    def toks_of(t):
        return {x for x in _tokens(t) if x.lower() not in skip}

    base = [toks_of(x) for x in seen]
    keep, dropped = [], []
    for pt in points:
        toks = toks_of(pt)
        if not toks:
            continue
        score = max([len(toks & b) / max(1, min(len(toks), len(b)))
                     for b in base if b] or [0.0])
        if score >= threshold:
            dropped.append((pt, score))
        else:
            keep.append(pt)
            base.append(toks)
    # 砍到低於下限時，把最不重複的幾則放回來——整欄只剩一則，
    # 使用者在紙上就沒得挑了
    if keep_min and len(keep) < keep_min and dropped:
        # 只放回「有點像」的，不放回「幾乎一樣」的。整欄變短，好過在
        # 第四欄看到第三欄那句話原封不動再出現一次。
        dropped.sort(key=lambda x: x[1])
        restorable = [d for d in dropped if d[1] < 0.75]
        take = restorable[:keep_min - len(keep)]
        keep.extend(pt for pt, _ in take)
        dropped = [d for d in dropped if d not in take]
    return keep, [d[0] if isinstance(d, tuple) else d for d in dropped]


def repeated_openings(points, head: int = 4):
    """回傳 (最常見的開頭, 出現次數, 佔比)。

    使用者的原話：「全部格式都一樣 而且在寫啥阿 小學生才這樣寫吧」。
    實測「今日反思」六則有三則以「我原本以」開頭、六則全部是
    「我原本以為X，但老師說Y，這讓我Z」這個模板。光靠 prompt 說
    「不要重複」不夠，要真的量出來、超標就重寫。
    """
    if len(points) < 3:
        return "", 0, 0.0
    heads = {}
    for x in points:
        k = (x or "").strip()[:head]
        if k:
            heads[k] = heads.get(k, 0) + 1
    top = max(heads.items(), key=lambda kv: kv[1])
    return top[0], top[1], top[1] / len(points)


def drop_ungrounded(points, haystack: str, glossary=()):
    """丟掉素材裡找不到依據的要點，回傳 (保留的, 丟掉的)。

    使用者是把這份抄到紙上交出去的，編出來的規格數字比漏寫嚴重得多。
    術語表裡的英文詞不算新造，那是這門課本來就在講的東西。
    """
    from . import verify_notes
    ok_words = {str(g).lower() for g in (glossary or ())}
    keep, dropped = [], []
    for claim, _ratio, _ok, novel in verify_notes.check_claims(points, haystack):
        allowed = _gloss_expansions(claim, haystack, ok_words)
        bad = [n for n in novel
               if n.lower() not in ok_words and n.lower() not in allowed]
        if bad:
            dropped.append((claim, bad))
        else:
            keep.append(claim)
    return keep, dropped


def build_worksheet_schema(secs, with_exam=True, n_source: int = 0):
    """學習單的 JSON schema。

    上下限不是裝飾用的：llama-server 會把 schema 編成 grammar。沒有 maxItems
    的話模型會整份學習單寫完再從頭寫一次——實測一次輸出裡出現兩份 content，
    3,045 字全部撞上 max_tokens，真正有用的只有前面一半。
    """
    props = {}
    for sec in secs:
        lo, hi = worksheet_counts(sec, n_source)
        props[sec["id"]] = {"type": "array", "items": {"type": "string"},
                            "minItems": lo, "maxItems": hi}
    schema = {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": props,
                "required": [x["id"] for x in secs],
            },
        },
        "required": ["fields"],
    }
    if with_exam:
        schema["properties"]["exam"] = {
            "type": "array", "items": {"type": "string"}, "maxItems": 6}
    return schema


def build_worksheet_fields(secs, n_source: int = 0) -> str:
    """把欄位設定寫成 prompt 裡的清單。"""
    lines = []
    for i, sec in enumerate(secs, 1):
        lo, hi = worksheet_counts(sec, n_source)
        line = "%d. id=%s　標題「%s」　要 %d 到 %d 則" % (
            i, sec["id"], sec["title"], lo, hi)
        if sec.get("style") == "detail":
            line += "（要寫細節與名詞解釋，不要只寫標題）"
        if sec.get("hint"):
            line += "\n   —— " + sec["hint"]
        lines.append(line)
    return "\n".join(lines)


HANDCOPY_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "points"],
            },
        },
        "exam": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topics"],
}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "groups": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"},
                               "minItems": 1, "maxItems": 10},
                },
                "required": ["heading", "points"],
            },
        },
    },
    "required": ["title", "summary", "groups"],
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overview"],
}

MAX_BULLETS = 24

# 實測值（向 llama-server 的 /tokenize 問來的，不是估的）：
#   真實逐字稿 0.55 token/字、乾淨繁體句子 0.79、簡體 0.67
# 一則要點約 60 字 ≈ 33 token，加上 JSON 結構的開銷抓 80。
TOKENS_PER_CHAR = 0.8          # estimate_tokens 用，往高估比較安全
TOKENS_PER_BULLET = 80


def _split_text(text: str, limit: int):
    """把長文切成不超過 limit 字的塊，盡量在句號處斷開。"""
    if len(text) <= limit:
        return [text]
    out = []
    i = 0
    while i < len(text):
        end = min(len(text), i + limit)
        if end < len(text):
            # 往回找最近的句尾，避免把一句話切兩半
            cut = max(text.rfind(c, i + limit // 2, end)
                      for c in "。！？.!?\n")
            if cut > i:
                end = cut + 1
        out.append(text[i:end])
        i = end
    return out


def estimate_tokens(text: str) -> int:
    """粗估一段文字的 token 數。寧可高估。"""
    return int(len(text or "") * TOKENS_PER_CHAR) + 16


def _tokens_for(n_bullets: int, prompt_chars: int = 0, ctx: int = 0) -> int:
    """依要點數決定 max_tokens，並確保 prompt + output 不超過 ctx。

    max_tokens 不是免費的——它和 prompt 共用同一個 context window。
    先前一度把每則要點的預算調到 300 token，24 則就是 7600，
    配上 8723 字（約 4800 token）的逐字稿會直接撐爆 ctx 8192。
    """
    want = max(config.SUMMARY_MAX_TOKENS, 300 + n_bullets * TOKENS_PER_BULLET)
    if ctx:
        room = ctx - estimate_tokens(" " * prompt_chars) - 256   # 256 留給系統訊息
        if room > 256:
            want = min(want, room)
    return max(256, want)


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
# Qwen3 等 hybrid 模型的思考鏈。llama_manager 會盡量用旗標關掉，
# 但旗標名稱在 llama.cpp 版本間換過，這裡再擋一層。
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """剝除 <think>…</think>；若思考鏈被截斷而沒有收尾標籤，整段丟棄。"""
    t = _THINK_RE.sub("", text or "")
    if "</think>" not in t:
        t = _OPEN_THINK_RE.sub("", t)
    return t


def strip_fence(text: str) -> str:
    """剝除 LLM 常加的 ```json 圍欄與思考鏈（規格 §6.3：防禦性處理）。"""
    t = strip_think(text or "").strip()
    t = _FENCE_RE.sub("", t)
    t = _FENCE_RE.sub("", t)
    return t.strip()


def extract_json(text: str):
    """盡力從模型輸出裡撈出一個 JSON 物件；失敗回傳 None。"""
    t = strip_fence(text)
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # 退而求其次：抓第一個 { 到最後一個 } 的括號平衡子字串
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def repair_truncated_json(text: str):
    """輸出被 max_tokens 砍斷時，把最後一個不完整的元素切掉再補回括號。

    截斷的輸出仍然帶著前面所有已經生成完的要點；直接判定失敗然後重試，
    等於把那些內容丟掉再花一次推論。實測 mmdb-2026 那堂 31,449 字的逐字稿
    切成 6 塊，有 3 塊撞到這個情況。

    只處理「結構沒收尾」這一種壞法；其他壞法回傳 None，交給原本的重試。
    """
    t = strip_fence(text)
    start = t.find("{")
    if start < 0:
        return None
    stack = []
    in_str = esc = False
    cut = -1            # 最後一個「完整的值」結束後的位置
    cut_stack = None    # 切在那個位置時還有哪些括號沒關
    i = start
    while i < len(t):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                # 後面接冒號的話這是鍵不是值，切在這裡會留下孤兒鍵
                j = i + 1
                while j < len(t) and t[j].isspace():
                    j += 1
                if j >= len(t) or t[j] != ":":
                    cut = i + 1
                    cut_stack = list(stack)
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            cut = i + 1
            cut_stack = list(stack)
            if not stack:
                return None     # 結構本來就完整，不是截斷問題
        i += 1
    if not stack or cut <= start or not cut_stack:
        return None
    # 補括號要用「切點當下」的堆疊，不是掃到結尾的堆疊。截斷點通常比切點
    # 深好幾層（例：切在 "id" 的值之後，結尾卻已經進到 points 陣列裡），
    # 用結尾的堆疊補會多關掉根本還沒開的括號。
    body = t[start:cut].rstrip().rstrip(",")
    try:
        obj = json.loads(body + "".join(reversed(cut_stack)))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class _Traditional:
    """把 LLM 輸出強制轉成繁體（台灣用語）。

    ASR 那邊早就有這層（asr.py 的 OpenCC s2twp），但摘要輸出一直沒有——
    Qwen3 是中國訓練的模型，即使 prompt 要求繁體仍會吐簡體字。
    prompt 只是請求，這裡才是保證。
    """

    # 兩類要修的字：
    # (1) 模型偶爾吐出「適閤」「結閤」這種罕用異體字，OpenCC 不會動它
    #     （閤在「內閣」「閤家」裡是對的）。
    # (2) OpenCC 的台灣用語層有「导出→匯出」這條規則（檔案匯出的意思），
    #     它會把正確的「推導出」一起改成「推匯出」。這條連純繁體輸入都會
    #     被改壞，而且影響的是全部筆記，不只學習單。
    # 用詞表而不是直接換字，才不會誤傷真的要講「匯出」「內閣」的地方。
    VARIANTS = {
        "適閤": "適合", "結閤": "結合", "符閤": "符合", "閤作": "合作",
        "綜閤": "綜合", "配閤": "配合", "組閤": "組合", "閤適": "合適",
        "整閤": "整合", "閤理": "合理", "閤約": "合約", "閤併": "合併",
        "閤格": "合格", "閤法": "合法", "閤計": "合計", "閤同": "合同",
        "推匯出": "推導出", "引匯出": "引導出", "誘匯出": "誘導出",
        "匯出結論": "導出結論", "匯出公式": "導出公式", "匯出定理": "導出定理",
        # 同一類誤傷：台灣用語層有「程序→程式」（軟體的意思），
        # 會把「調查程序」「行政程序」也一起改掉。
        "調查程式": "調查程序", "行政程式": "行政程序", "司法程式": "司法程序",
        "訴訟程式": "訴訟程序", "法律程式": "法律程序",
        "標準作業程式": "標準作業程序", "作業程式書": "作業程序書",
        "程式正義": "程序正義",
        # 第三條：「权限→許可權」。許可權是對岸／微軟的譯法，台灣講權限。
        # 這條還會連帶把「人權限制」變成「人許可權制」、「授權限制」變成
        # 「授許可權制」——換回去之後那些複合詞會一起修好。
        "許可權": "權限",
        # 「信息安全→資訊保安」是港式，台灣講資訊安全
        "資訊保安": "資訊安全",
    }

    def __init__(self):
        self._c = None
        try:
            from opencc import OpenCC
            self._c = OpenCC("s2twp")
        except Exception as e:      # pragma: no cover
            log.warning("OpenCC 不可用（%s），摘要可能出現簡體字", e)

    def _fix_variants(self, t: str) -> str:
        if any(k in t for k in ("閤", "匯", "程式", "許可權", "保安")):
            for bad, good in self.VARIANTS.items():
                t = t.replace(bad, good)
        return t

    def __call__(self, x):
        if isinstance(x, str) and self._c is None:
            return self._fix_variants(x)
        if self._c is None:
            return x
        if isinstance(x, str):
            try:
                return self._fix_variants(self._c.convert(x))
            except Exception:
                return self._fix_variants(x)
        if isinstance(x, list):
            return [self(i) for i in x]
        if isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        return x


to_traditional = _Traditional()


def fmt_ts(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


class Summarizer:
    """llama-server 的 OpenAI 相容 /v1/chat/completions 客戶端。"""

    def __init__(self, manager, base_url=None, timeout_s: float = 180.0):
        self.manager = manager
        self.base_url = base_url or manager.base_url
        self.timeout_s = timeout_s
        self._supports_schema = True
        self.parse_attempts = 0
        self.parse_failures = 0
        self.salvaged = 0
        self._last_finish = None

    # 低階 -------------------------------------------------------------
    async def _chat(self, system: str, user: str, max_tokens: int,
                    temperature: float, schema=None) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if schema is not None and self._supports_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "note", "schema": schema, "strict": True},
            }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.post(self.base_url + "/v1/chat/completions", json=payload)
                if r.status_code >= 400 and "response_format" in payload:
                    # 這版 llama-server 不吃 json_schema，降級為純 prompt 約束
                    log.warning("llama-server 不支援 json_schema（HTTP %d），改用純 prompt 約束",
                                r.status_code)
                    self._supports_schema = False
                    payload.pop("response_format")
                    r = await c.post(self.base_url + "/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            raise LLMUnavailable("呼叫 llama-server 失敗：%s" % e)
        try:
            self._last_finish = data["choices"][0].get("finish_reason")
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMUnavailable("llama-server 回應格式異常：%r" % (data,))

    async def _chat_json(self, system: str, user: str, max_tokens: int, schema):
        """呼叫並解析 JSON；失敗重試一次並加上格式糾正提示（規格 §6.3）。

        回傳 (obj_or_None, raw_text)。
        """
        self.parse_attempts += 1
        raw = await self._chat(system, user, max_tokens,
                               config.SUMMARY_TEMPERATURE, schema)
        obj = extract_json(raw)
        if obj is not None:
            return to_traditional(obj), raw
        salvaged = repair_truncated_json(raw)
        # 只救得回標題、一則要點都沒有的話，救回來也沒用，照樣重試
        if salvaged and not any(
                isinstance(salvaged.get(k), list) and salvaged.get(k)
                for k in ("bullets", "topics", "overview", "exam")) and not (
                    isinstance(salvaged.get("fields"), dict)
                    and any(salvaged["fields"].values())):
            salvaged = None
        if salvaged:
            self.salvaged += 1
            log.warning("輸出被截斷（finish_reason=%s，%d 字），已救回 %d 則要點，"
                        "不重試", getattr(self, "_last_finish", None), len(raw),
                        len(salvaged.get("bullets") or salvaged.get("topics") or
                            salvaged.get("overview") or
                            salvaged.get("fields") or []))
            return to_traditional(salvaged), raw
        log.warning("摘要 JSON 解析失敗（finish_reason=%s），重試一次。原始輸出前 200 字：%r",
                    getattr(self, "_last_finish", None), raw[:200])
        retry_user = "上次輸出格式錯誤，請只輸出 JSON，不要有任何其他文字。\n\n" + user
        raw2 = await self._chat(system, retry_user, max_tokens, 0.1, schema)
        obj2 = extract_json(raw2)
        if obj2 is None:
            self.parse_failures += 1
            return None, raw2 or raw
        return to_traditional(obj2), raw2

    # 段落摘要（規格 §6.3）--------------------------------------------
    @staticmethod
    def build_section_user(course, context_sections, transcript: str) -> str:
        if context_sections:
            ctx_lines = []
            for s in context_sections:
                ctx_lines.append("- 標題：" + str(s.get("title", "")))
                for b in (s.get("bullets") or []):
                    ctx_lines.append("  - " + str(b))
            ctx = "\n".join(ctx_lines)
        else:
            ctx = "（本段為課程開頭）"
        parts = ["【課程】" + course.name]
        if course.glossary:
            # glossary 與 asr_prompt 共用同一份術語來源（規格 §9.3）
            parts.append("【本課常見術語】" + course.glossary_line)
        parts.append("【前文脈絡】\n" + ctx)
        parts.append("【本段逐字稿】\n" + transcript)
        return "\n\n".join(parts)

    def _max_prompt_chars(self) -> int:
        """一次呼叫能塞多少字的逐字稿。

        ctx 要同時容納 system prompt、逐字稿、前文脈絡與輸出。
        超過就會被 llama-server 截斷——而且是**無聲**的：
        實測一堂 32,368 字的課（25,910 token，ctx 的 3.2 倍）整段被丟掉，
        只留下最前面 4 分鐘的內容，畫面上完全看不出來。
        """
        ctx = getattr(self.manager, "current_ctx", 0) or 4096
        # 預留：system prompt 與前文脈絡約 900 token、輸出約 2200 token
        room = max(1000, ctx - 900 - 2200)
        return int(room / TOKENS_PER_CHAR)

    async def summarize_span(self, course, context_sections, transcript: str,
                             start_s: float, end_s: float):
        """把一段逐字稿整理成一到多則 section。

        逐字稿太長就依 context 上限切塊，每塊各自成為一則 section，
        時間依字數比例分配。使用者整堂課只按一次按鈕時，這是唯一
        還能產出有用筆記的路徑——否則 2.5 小時的內容會整個消失。
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return []
        limit = self._max_prompt_chars()
        chunks = _split_text(transcript, limit)
        out = []
        pos = start_s
        total = len(transcript)
        for ch in chunks:
            res = await self.summarize_section(course, (context_sections + out)[-2:], ch)
            share = (end_s - start_s) * (len(ch) / total) if total else 0.0
            out.append({
                "start_s": pos, "end_s": min(end_s, pos + share),
                "title": res["title"], "bullets": res["bullets"],
                "summary": res.get("summary", ""),
                "groups": res.get("groups") or [],
                "degraded": res.get("degraded", False),
            })
            pos += share
        if out:
            out[-1]["end_s"] = end_s
        return out

    async def summarize_section(self, course, context_sections, transcript: str):
        """回傳 {"title":..., "bullets":[...], "degraded": bool}。

        要點數量隨逐字稿長度調整。寫死上限（原本是 2–6 則）會讓十五分鐘的
        段落被壓成三句話——實測 6709 字的逐字稿只產出 12 個要點，
        壓縮比 3.7%，使用者拿不到能用的筆記。
        """
        await self.manager.ensure("inclass")
        n = len(transcript)
        target = max(3, min(MAX_BULLETS, n // config.CHARS_PER_BULLET))
        system = SECTION_SYSTEM_TMPL % (n, target)
        user = self.build_section_user(course, context_sections, transcript)
        ctx = getattr(self.manager, "current_ctx", 0)
        obj, raw = await self._chat_json(
            system, user, _tokens_for(target, len(system) + len(user), ctx),
            SECTION_SCHEMA)
        if obj is None:
            # 降級：把原始輸出當成單一 bullet（規格 §6.3）
            text = to_traditional(strip_fence(raw))[:400] or "（模型未輸出內容）"
            return {
                "title": "（自動摘要格式異常）", "summary": "",
                "groups": [{"heading": "", "points": [text]}],
                "bullets": [text], "degraded": True,
            }
        title = str(obj.get("title") or "").strip()[:40] or "未命名段落"
        summary = ensure_sentence(strip_json_artifacts(obj.get("summary") or ""))
        groups, bullets = [], []
        for g in (obj.get("groups") or []):
            if not isinstance(g, dict):
                continue
            pts = [ensure_sentence(strip_json_artifacts(x))
                   for x in (g.get("points") or [])]
            pts = [x for x in pts if x and not is_placeholder(x)]
            if not pts:
                continue
            head = strip_json_artifacts(g.get("heading") or "")[:30]
            groups.append({"heading": head, "points": pts})
            bullets.extend(pts)
        if not bullets:
            # 舊格式的模型輸出（只有 bullets）也接住，不要整段掉光
            bullets = [ensure_sentence(strip_json_artifacts(b))
                       for b in (obj.get("bullets") or [])]
            bullets = [b for b in bullets if b]
            groups = [{"heading": "", "points": bullets}] if bullets else []
        if not bullets:
            bullets = ["（模型未產出要點）"]
            groups = [{"heading": "", "points": bullets}]
        # bullets 是攤平後的版本，給手抄版、期末總結與舊的顯示路徑用
        return {"title": title, "summary": summary, "groups": groups,
                "bullets": bullets[:MAX_BULLETS], "degraded": False}

    # 期末總結（規格 §6.4）--------------------------------------------
    @staticmethod
    def build_final_user(course, sections) -> str:
        lines = ["【課程】" + course.name]
        if course.instructor:
            lines.append("【講者】" + course.instructor)
        lines.append("【各段落筆記】")
        for s in sections:
            lines.append("")
            lines.append("### [%s] %s" % (fmt_ts(s.get("start_s", 0.0)),
                                          s.get("title", "")))
            for b in (s.get("bullets") or []):
                lines.append("- " + str(b))
            if s.get("user_note"):
                lines.append("（使用者標註：%s）" % s["user_note"])
        return "\n".join(lines)

    async def summarize_handcopy(self, course, sections, model_key="inclass"):
        """產出手抄版筆記：短、有層次、能直接照著寫在紙上。

        model_key 預設用課中模型：這個功能會在上課中途被呼叫，
        切到 8B 要卸載 ASR 又要 10–20 秒冷啟動，會中斷逐字稿。
        下課後呼叫時才值得用 final 換品質。
        """
        await self.manager.ensure(model_key)
        user = self.build_final_user(course, sections)
        # 目標長度隨輸入規模走：段落多的課本來就該產出更多內容。
        # 原本寫死 250 字，30 分鐘的課只剩 251 字，等於沒整理到。
        total = sum(len(t) for sc in sections for t in (sc.get("bullets") or []))
        # 課程設定檔可以把這堂的下限拉高（學習單要抄的量比一般手抄版大）
        floor = (getattr(course, "handcopy_target_chars", 0)
                 or config.HANDCOPY_TARGET_CHARS)
        target = max(floor, min(max(1600, floor * 2), int(total * 0.55)))
        if getattr(course, "is_worksheet", False):
            return await self._handcopy_worksheet(course, sections, target)
        system = HANDCOPY_SYSTEM_TMPL % target
        ctx = getattr(self.manager, "current_ctx", 0)
        want = max(1200, int(target * TOKENS_PER_CHAR * 1.6))
        if ctx:
            room = ctx - estimate_tokens(system + user) - 256
            if room > 256:
                want = min(want, room)
        obj, raw = await self._chat_json(system, user, max(256, want),
                                         HANDCOPY_SCHEMA)
        if obj is None:
            return {"topics": [{"title": "（自動整理格式異常）",
                                "points": [to_traditional(strip_fence(raw))[:300] or "（模型未輸出內容）"]}],
                    "exam": [], "degraded": True}
        topics = []
        for t in (obj.get("topics") or []):
            title = str(t.get("title") or "").strip()
            pts = [ensure_sentence(strip_json_artifacts(p))
                   for p in (t.get("points") or [])]
            pts = [p for p in pts if p]
            if title or pts:
                topics.append({"title": title or "（未命名）", "points": pts[:6]})
        if not topics:
            topics = [{"title": "（模型未產出內容）", "points": []}]
        exam = [str(x).strip() for x in (obj.get("exam") or []) if str(x).strip()]
        return {"topics": topics[:8], "exam": exam, "degraded": False}

    async def _worksheet_call(self, system, user, secs, cap_chars, with_exam,
                              n_source=0):
        """跑一次模型，回傳 ({欄位id: [要點]}, exam, degraded, raw)。"""
        schema = build_worksheet_schema(secs, with_exam=with_exam,
                                        n_source=n_source)
        ctx = getattr(self.manager, "current_ctx", 0)
        want = max(1200, int(cap_chars * TOKENS_PER_CHAR * 2.0))
        if ctx:
            room = ctx - estimate_tokens(system + user) - 256
            if room > 256:
                want = min(want, room)
        obj, raw = await self._chat_json(system, user, max(256, want), schema)
        if obj is None:
            return {}, [], True, raw
        got = {}
        fields = obj.get("fields")
        if isinstance(fields, dict):
            for key, val in fields.items():
                if isinstance(val, list):
                    got[str(key).strip()] = [str(x).strip() for x in val
                                             if str(x).strip()]
        exam = [str(x).strip() for x in (obj.get("exam") or []) if str(x).strip()]
        return got, exam, False, raw

    def _worksheet_chunks(self, course, sections, secs):
        """把段落切成幾塊，讓每一塊的 prompt 都留得下輸出空間。

        一次把 2.5 小時的段落全塞進去，prompt 會吃掉大半個 context，
        剩下的 max_tokens 寫不完所有欄位，模型就會挑一個案例寫完了事——
        使用者看到的就是「太少而且局限於一個範圍」。切塊之後每一塊都被
        要求交出內容，涵蓋率才是被結構保證的，不是靠 prompt 拜託模型。
        """
        ctx = getattr(self.manager, "current_ctx", 0) or 4096
        cap = sum(worksheet_counts(x)[1] for x in secs)
        # 預留：system prompt + 這一塊要寫出來的輸出
        budget = ctx - 700 - int(cap * 60 * TOKENS_PER_CHAR * 2.0) - 256
        limit = max(800, int(max(budget, 600) / TOKENS_PER_CHAR))
        chunks, cur, cur_n = [], [], 0
        for sec in sections:
            n = sum(len(b) for b in (sec.get("bullets") or [])) + 40
            if cur and cur_n + n > limit:
                chunks.append(cur)
                cur, cur_n = [], 0
            cur.append(sec)
            cur_n += n
        if cur:
            chunks.append(cur)
        return chunks or [list(sections)]

    # 同一欄裡超過這個比例的要點用同樣的開頭，就判定是模板化，重寫一次
    REPEAT_LIMIT = 0.34

    async def _reflect_field(self, course, sec, base_user: str, avoid, haystack,
                             n_source: int = 0):
        """產出一個感受型欄位，回傳 (要點列表, degraded)。"""
        lo, hi = worksheet_counts(sec, n_source)
        clo, chi = worksheet_chars(sec)
        avoid_block = ""
        if avoid:
            avoid_block = WORKSHEET_AVOID_TMPL % "\n".join(
                "- " + a for a in avoid[-14:])
        system = WORKSHEET_REFLECT_TMPL % (
            sec["title"], sec.get("hint") or sec["title"], lo, hi,
            clo, chi, avoid_block, sec["id"])
        schema = build_worksheet_schema([sec], with_exam=False, n_source=n_source)

        async def run(user):
            ctx = getattr(self.manager, "current_ctx", 0)
            want = max(900, int(hi * chi * TOKENS_PER_CHAR * 2.0))
            if ctx:
                room = ctx - estimate_tokens(system + user) - 256
                if room > 256:
                    want = min(want, room)
            obj, _raw = await self._chat_json(system, user, max(256, want), schema)
            fields = (obj or {}).get("fields")
            vals = fields.get(sec["id"]) if isinstance(fields, dict) else None
            return [str(x).strip() for x in (vals or [])
                    if not is_placeholder(str(x))]

        # 字數上限比要求的再寬一點，不然剛好寫到上限的那幾則會被判成抄不完
        cap = chi + 25
        pts = await run(base_user)
        issues = worksheet_quality_issues(pts, max_chars=cap)
        if issues:
            log.warning("「%s」品質不合格，重寫一次：%s",
                        sec["title"], "；".join(issues))
            retry = await run(base_user + "\n\n" + WORKSHEET_RETRY_TMPL
                              % ("\n".join("- " + x for x in issues), chi))
            if retry and len(worksheet_quality_issues(
                    retry, max_chars=cap)) < len(issues):
                pts = retry
        fake = fabricated_past(pts)
        if fake:
            # 編造的個人經歷一律不留：這份是要交給老師的
            log.warning("「%s」丟掉 %d 則編造的個人經歷：%s",
                        sec["title"], len(fake), fake[0][:30])
            pts = [x for x in pts if x not in fake]
        keep, dropped = drop_ungrounded(pts, haystack, course.glossary)
        if dropped:
            log.warning("「%s」丟掉 %d 則查無依據的內容", sec["title"], len(dropped))
        keep, dup = dedupe_against(keep, avoid or [], ignore=course.glossary,
                                   keep_min=max(2, lo - 1))
        if dup:
            log.warning("「%s」丟掉 %d 則跟前面欄位重複的內容：%s",
                        sec["title"], len(dup), dup[0][:30])
        return keep, not keep

    async def _handcopy_worksheet(self, course, sections, target: int):
        """依課程設定的固定欄位產出學習單（回傳格式與一般手抄版相容）。

        分兩階段：先寫「這堂課講了什麼」（段落多就分塊，每塊各寫一批），
        再寫需要個人感受的欄位。擠在同一次呼叫時 max_tokens 要同時養活所有
        欄位，上課內容永遠是被犧牲的那一欄——實測 2.5 小時的課只寫出 8 則，
        而且全部集中在同一個案例。
        """
        record, reflect = worksheet_groups(course)
        full_user = self.build_final_user(course, sections)
        got, exam, degraded, raw = {}, [], False, ""

        if record:
            chunks = self._worksheet_chunks(course, sections, record)
            per = [dict(x) for x in record]
            if len(chunks) > 1:
                # 則數按塊數分攤，總量才不會因為切塊而暴增
                for x in per:
                    lo, hi = worksheet_counts(x)
                    x["count"] = (max(2, -(-lo // len(chunks))),
                                  max(3, -(-hi // len(chunks))))
            for i, chunk in enumerate(chunks, 1):
                user = self.build_final_user(course, chunk)
                if len(chunks) > 1:
                    user = ("（這是整堂課的第 %d 塊，共 %d 塊；只寫這一塊的內容）\n"
                            % (i, len(chunks))) + user
                # 這一塊手上有多少素材，決定可以要求幾則——素材不夠還硬要
                # 湊數量，模型就會開始編規格數字
                n_src = sum(len(x.get("bullets") or []) for x in chunk)
                system = WORKSHEET_RECORD_TMPL % (
                    build_worksheet_fields(per, n_src), max(1, len(chunk)),
                    int(target * 0.7 / len(chunks)))
                cap = sum(worksheet_counts(x, n_src)[1] for x in per) * 60
                g, e, d, r = await self._worksheet_call(
                    system, user, per, cap, with_exam=True, n_source=n_src)
                hay = "\n".join(b for x in chunk for b in (x.get("bullets") or []))
                for k, v in g.items():
                    v = [x for x in v if not is_placeholder(x)]
                    keep, dropped = drop_ungrounded(v, hay, course.glossary)
                    if dropped:
                        log.warning("學習單第 %d 塊丟掉 %d 則查無依據的內容：%s",
                                    i, len(dropped),
                                    "；".join("%s ← %s" % (c[:24], "/".join(b))
                                              for c, b in dropped[:3]))
                    got.setdefault(k, []).extend(keep)
                exam.extend(e)
                degraded = degraded or d
                raw = raw or r

        if reflect:
            written = [pt for sec in record for pt in (got.get(sec["id"]) or [])]
            if written:
                # 只帶已經寫好的上課內容，不要把整堂段落再塞一次。塞兩份的
                # 話 prompt 會吃掉大半個 context，輸出被 max_tokens 砍斷，
                # 「今日反思」那一欄整個空掉。
                base = "\n".join(
                    ["【課程】" + course.name,
                     "【這堂課的內容（請扣住這些不同的點）】"]
                    + ["- " + pt for pt in written])
            else:
                base = full_user
            hay = "\n".join(written) or "\n".join(
                b for x in sections for b in (x.get("bullets") or []))
            done = []
            for sec in reflect:
                # 一欄一次呼叫。三欄擠在同一次時輸出會互相同化——實測
                # 「想法」和「反思」講的是同一批事、連句型都一樣。
                pts, deg = await self._reflect_field(
                    course, sec, base, done, hay,
                    sum(len(x.get("bullets") or []) for x in sections))
                got[sec["id"]] = pts
                done.extend(pts)
                degraded = degraded or deg

        topics, missing = [], []
        n_all = sum(len(x.get("bullets") or []) for x in sections)
        for sec in course.handcopy_sections:
            hi = worksheet_counts(sec, n_all)[1]
            pts = got.get(sec["id"]) or []
            # 模型有時會在第一則前面把欄位名再抄一次（「今天上課內容：……」），
            # 抄到紙上是多餘的字，先拿掉
            pts = [ensure_sentence(_strip_title_echo(x, sec["title"])) for x in pts]
            pts = [x for x in pts if x]
            if not pts:
                missing.append(sec["title"])
            # 上限用這一欄自己的設定，不要再寫死一個數字——寫死的上限
            # 正是之前整堂課只剩幾則的原因
            topics.append({"title": sec["title"], "points": pts[:hi],
                           "pick": sec.get("pick") or 0})
        if missing:
            log.warning("學習單欄位沒產出內容：%s", "、".join(missing))
        if not any(t["points"] for t in topics):
            return {"topics": [{"title": course.handcopy_sections[0]["title"],
                                "points": [to_traditional(strip_fence(raw))[:300]
                                           or "（模型未輸出內容）"],
                                "pick": 0}],
                    "exam": [], "worksheet": True, "degraded": True}
        seen, uniq_exam = set(), []
        for e in exam:
            if e not in seen:
                seen.add(e)
                uniq_exam.append(e)
        return {"topics": topics, "exam": uniq_exam[:8], "worksheet": True,
                "degraded": degraded}

    async def summarize_final(self, course, sections):
        """回傳 {"overview":[...], "open_questions":[...], "degraded": bool}。

        輸入是 sections 的結構化內容，不是逐字稿（規格 §6.4）。
        """
        await self.manager.ensure("final")
        user = self.build_final_user(course, sections)
        obj, raw = await self._chat_json(
            FINAL_SYSTEM, user, config.FINAL_MAX_TOKENS, FINAL_SCHEMA)
        if obj is None:
            return {
                "overview": [to_traditional(strip_fence(raw))[:600] or "（模型未輸出內容）"],
                "open_questions": [],
                "degraded": True,
            }
        overview = [str(x).strip() for x in (obj.get("overview") or []) if str(x).strip()]
        oq = [str(x).strip() for x in (obj.get("open_questions") or []) if str(x).strip()]
        if not overview:
            overview = ["（模型未產出概述）"]
        return {"overview": overview, "open_questions": oq[:5], "degraded": False}
