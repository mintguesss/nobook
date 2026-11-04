"""會議記錄：逐字稿 → 固定五欄的整理表，可以下載成 Excel。

一小時的會議有四、五萬字，模型一次塞不下，所以分塊抽出候選，再一欄一欄合併。
合併時要模型回報每一則來自哪幾個候選，時間點才接得回原音，整理的人能點回去對。
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import re

from . import config
from .summarizer import fmt_ts

log = logging.getLogger(__name__)

# (key, 欄名, 這一欄收什麼, 最多幾則)
# 欄位說明只列主題、不寫成句子：寫成「怎麼評估期刊」這種句子，模型會照抄成一則筆記，
# 實測逐字稿完全沒提期刊也會寫出「文獻需評估期刊影響力」。
COLUMNS = (
    ("literature", "文獻整理重點", "找文獻和讀文獻的方法、研究方法、理論", 14),
    ("topic", "題目整理建議", "對論文題目、變數、架構、實驗或問卷設計的回饋", 12),
    ("next", "下次目標", "明確交代或答應要做的工作", 6),
    ("ideas", "可用觀點記錄", "可以用在研究裡的觀點、案例、現象", 6),
    ("misc", "小事記錄", "研討會、補助、截止日期、行政與報告上的提醒", 6),
)
COLUMN_KEYS = tuple(c[0] for c in COLUMNS)

# 一塊太大的話模型只挑得出少數幾則，實測 12000 字一塊只抽得出手寫表的三分之一
MAP_CHUNK_CHARS = 3000
MAP_MAX_TOKENS = 1500
REDUCE_MAX_TOKENS = 2500
# 字數太短的幾乎都是語助詞，或沒人講話時辨識出來的幻覺
MIN_LINE_CHARS = 6
ITEM_MAX_CHARS = 90

CONTEXT = """這是一場論文指導會議，通常是一位教授和兩位研究生。其中一位學生負責錄音和整理這份紀錄，\
整理是給她自己之後做研究用的。她最需要的是教授的回饋，不是學生自己報告過的內容。
逐字稿的講者是用聲音自動分的，可能認錯。標了（教授）的是講最多話的人，通常就是教授，但還是要看內容判斷。"""

RULES = """- 主要收教授講的話：他的建議、糾正、提問、舉的例子、交代的事。
- 學生報告的文獻或題目本身不要收。只有教授對它有回應時才收，而且寫的是教授的回應。
- 教授說「不用」「不要」「先不要」的事，要照原意寫成不用做，絕對不可以寫反。
- 教授舉例說明一個道理時，寫下那個道理；例子有趣或可以用的才放進可用觀點。
- 另一位學生的題目如果跟主要討論的題目無關，只收可以通用的方法或提醒。
- 只能根據逐字稿真的講到的內容，不可以補充延伸、不可以用你自己的知識補完。
- 閒聊、吃飯、八卦、調整設備、私人話題一律不收。
- 寫成自己看得懂的短筆記，一則一件事，具體寫出內容，不要寫「老師說」「教授建議」，也不要寫「需進一步探討」這種空話。
- 逐字稿是語音辨識的，有同音錯字：看得出原意就改正；專有名詞聽不出來就照寫並加上 (?)，不要猜。
- 一律使用繁體中文（台灣用語）。沒有內容的欄位給空陣列，寧可少也不要湊數。"""


def _column_lines() -> str:
    return "\n".join("- %s（%s）：%s" % (k, title, desc) for k, title, desc, _ in COLUMNS)


MAP_SYSTEM = """你在幫研究生整理論文指導會議的紀錄。使用者會給你一段逐字稿，每行開頭是時間和講者。

%s

請把這段裡值得記下來的內容，放進下面五欄：
%s

# 規則
%s
- 每一則都要先找依據：quote 從教授講的那一行一字不改地抄 10 到 40 字，再根據它寫 text。抄不出來就是沒講，不要寫。
- text 不超過 70 字，只能改寫 quote 和它前後文實際的意思，要具體到讀的人不用回去聽也知道該怎麼做。
- 「下次目標」只收教授明確要學生回去做的事，或交代接下來階段該怎麼做；「可以進一步探討」這類不是交代的不算。
- t 填 quote 那一行方括號裡的時間。
- 這段如果沒有值得記的內容，五欄都給空陣列。

只輸出 JSON：{"fields":{"literature":[{"quote":"…","t":"00:12:34","text":"…"}],"topic":[],"next":[],"ideas":[],"misc":[]}}
""" % (CONTEXT, _column_lines(), RULES)

REDUCE_SYSTEM = """你在幫研究生整理論文指導會議的紀錄。使用者會給你「{title}」這一欄的候選筆記，\
它們是從同一場會議的不同段落分別抽出來的，所以會有重複。

這一欄收的是：{desc}

# 規則
- 講同一件事的合併成一則，ids 列出合併了哪幾個候選的編號。
- 只能用候選裡有的內容，不可以加入候選沒有的資訊。
- 不屬於這一欄的、空泛沒有內容的（例如「要多讀論文」這種沒有具體做法的）直接丟掉。
- 最多 {limit} 則，重要的優先，照候選原本的先後順序排列。
- 每則不超過 70 字，保持短筆記的語氣，不要寫「老師說」「教授建議」。

只輸出 JSON：{{"fields":{{"items":[{{"text":"…","ids":[1,4]}}]}}}}
"""

_ITEM_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        # 鍵的順序就是模型生成的順序：先抄依據再寫筆記，比較不會自己編
        "properties": {"quote": {"type": "string"}, "t": {"type": "string"},
                       "text": {"type": "string"}},
        "required": ["quote", "t", "text"],
    },
}
MAP_SCHEMA = {
    "type": "object",
    "properties": {"fields": {
        "type": "object",
        "properties": {k: _ITEM_SCHEMA for k in COLUMN_KEYS},
        "required": list(COLUMN_KEYS),
    }},
    "required": ["fields"],
}
REDUCE_SCHEMA = {
    "type": "object",
    "properties": {"fields": {
        "type": "object",
        "properties": {"items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["text", "ids"],
            },
        }},
        "required": ["items"],
    }},
    "required": ["fields"],
}


class NoContent(ValueError):
    """逐字稿太短，沒有東西可以整理。"""


# ── 逐字稿前處理 ──────────────────────────────────────────────────────
def main_speaker(segments):
    """講最多話的人。指導會議裡幾乎都是教授，五場實際錄音都是。"""
    chars = {}
    for g in segments:
        if g.get("speaker"):
            chars[g["speaker"]] = chars.get(g["speaker"], 0) + len(g.get("text") or "")
    return max(chars, key=chars.get) if chars else None


def transcript_lines(segments, names=None) -> list:
    """逐字稿列 → [(秒數, 給模型看的那一行)]。"""
    names = names or {}
    top = main_speaker(segments)
    out = []
    for g in segments:
        text = (g.get("text") or "").strip()
        if len(text) < MIN_LINE_CHARS:
            continue
        spk = g.get("speaker")
        who = (names.get(str(spk)) or "講者%s" % spk) if spk else "講者?"
        if spk and spk == top:
            who += "（教授）"
        out.append((float(g.get("start_s") or 0.0),
                    "[%s] %s：%s" % (fmt_ts(g.get("start_s") or 0.0), who, text)))
    return out


def chunk_lines(lines, limit: int) -> list:
    chunks, cur, size = [], [], 0
    for start, line in lines:
        if cur and size + len(line) + 1 > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((start, line[:limit]))
        size += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


_TS_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")


def parse_ts(text, lo: float, hi: float) -> float:
    """模型抄回來的時間。對不上這一塊的範圍就退回這一塊的開頭，不要跳去奇怪的地方。"""
    m = _TS_RE.search(str(text or ""))
    if not m:
        return lo
    a, b, c = m.groups()
    sec = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    return float(sec) if lo <= sec <= hi else lo


_NOISE_RE = re.compile(r"[\s，,。.、：:；;！!？?「」『』（）()\[\]\-—…\"'/]+")
# 引文要有這麼多對得上教授講的話。不是 100%，因為模型抄的時候會順手改掉同音錯字。
# 實測抄教授的話幾乎都是 1.0，抄學生報告的只有 0.1 到 0.4，所以這條同時擋掉照抄報告。
QUOTE_MIN_MATCH = 0.6
# 筆記是改寫過的，改寫得好的跟原文重複的字本來就少，這條只擋完全無關的
TEXT_MIN_MATCH = 0.08


def _bigrams(text: str) -> set:
    t = _NOISE_RE.sub("", str(text or "")).lower()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def grounded(item: dict, source: set) -> bool:
    """這則筆記找得到依據嗎。模型有時候會寫出逐字稿根本沒提的內容，交出去就是錯的。"""
    quote, text = _bigrams(item.get("quote")), _bigrams(item.get("text"))
    if len(quote) < 4 or not text:
        return False
    return (len(quote & source) / len(quote) >= QUOTE_MIN_MATCH
            and len(text & source) / len(text) >= TEXT_MIN_MATCH)


def clean_item(text) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    # 編號在匯出時統一加，模型自己編的要拿掉
    text = re.sub(r"^\s*(?:\d+[.、．)]|[-•‧])\s*", "", text)
    text = re.sub(r"^(?:老師|教授)(?:要求|建議|說|提到|認為|表示|指出|強調)[，,：:]?", "", text)
    return text[:ITEM_MAX_CHARS].rstrip("，,、；;：: ")


# ── 產生 ──────────────────────────────────────────────────────────────
async def generate(summarizer, segments, names=None, terms=(),
                   on_progress=None, before_call=None) -> dict:
    """回傳 {"columns": {key: [{"text", "t"}]}, "chunks": n, "degraded": bool}。

    before_call 在每次呼叫模型前執行，用來讓路給現場錄音、檢查是否取消。
    """
    lines = transcript_lines(segments, names)
    if not lines:
        raise NoContent("逐字稿太短，沒有內容可以整理")
    # 先載入模型，才拿得到真正的 context 大小；沒載入時會切得太碎，模型看不到前後文
    if before_call:
        await before_call()
    await summarizer.manager.ensure("final")
    chunks = chunk_lines(lines, min(MAP_CHUNK_CHARS, summarizer._max_prompt_chars()))
    extra = ""
    if terms:
        extra = "這場會議會出現的詞（寫法以這裡為準）：%s\n\n" % "、".join(terms)
    steps = len(chunks) + len(COLUMNS)
    done = 0

    async def tick():
        nonlocal done
        done += 1
        if on_progress:
            await on_progress(done / steps)

    raw = {k: [] for k in COLUMN_KEYS}
    failed = dropped = 0
    for ch in chunks:
        if before_call:
            await before_call()
        await summarizer.manager.ensure("final")
        hi = ch[-1][0] + 60
        # 學生報告的內容不算依據，只比對教授和沒標到講者的行。沒有分講者的錄音就全部都算。
        spoken = [line for _, line in ch if "（教授）" in line or "講者?" in line] or [line for _, line in ch]
        source = _bigrams(" ".join(line.split("：", 1)[-1] for line in spoken))
        obj, _ = await summarizer._chat_json(
            MAP_SYSTEM, extra + "\n".join(line for _, line in ch),
            MAP_MAX_TOKENS, MAP_SCHEMA)
        fields = (obj or {}).get("fields") if isinstance(obj, dict) else None
        if not isinstance(fields, dict):
            failed += 1
            await tick()
            continue
        for k in COLUMN_KEYS:
            for it in fields.get(k) or []:
                if not isinstance(it, dict):
                    continue
                text = clean_item(it.get("text"))
                if not grounded(it, source):
                    dropped += 1
                    log.info("丟掉找不到依據的筆記：%s（%s）", text, it.get("quote"))
                    continue
                if text:
                    raw[k].append({"text": text, "t": parse_ts(it.get("t"), ch[0][0], hi)})
        await tick()
    if failed == len(chunks):
        raise RuntimeError("摘要模型沒有產出可用的內容")

    columns = {}
    for key, title, desc, limit in COLUMNS:
        if before_call:
            await before_call()
        columns[key] = await _reduce(summarizer, title, desc, limit, raw[key])
        await tick()
    return {"columns": columns, "chunks": len(chunks), "degraded": failed > 0,
            "dropped": dropped,
            "generated_at": datetime.datetime.now().astimezone().isoformat()}


async def _reduce(summarizer, title, desc, limit, items) -> list:
    if len(items) <= 1:
        return items
    await summarizer.manager.ensure("final")
    user = "\n".join("#%d %s" % (i, it["text"]) for i, it in enumerate(items, 1))
    obj, _ = await summarizer._chat_json(
        REDUCE_SYSTEM.format(title=title, desc=desc, limit=limit), user,
        REDUCE_MAX_TOKENS, REDUCE_SCHEMA)
    fields = (obj or {}).get("fields") if isinstance(obj, dict) else None
    merged = (fields or {}).get("items") if isinstance(fields, dict) else None
    if not isinstance(merged, list) or not merged:
        log.warning("「%s」合併失敗，保留未合併的候選", title)
        return items[:limit]
    out = []
    for m in merged:
        if not isinstance(m, dict):
            continue
        text = clean_item(m.get("text"))
        ids = [i for i in (m.get("ids") or [])
               if isinstance(i, int) and 1 <= i <= len(items)]
        # 說不出來自哪個候選的，就是模型自己寫的
        if text and ids:
            out.append({"text": text, "t": min(items[i - 1]["t"] for i in ids)})
    out.sort(key=lambda it: it["t"])
    return out[:limit] or items[:limit]


# ── 工作表名稱 ────────────────────────────────────────────────────────
_CN_DIGITS = "零一二三四五六七八九"
_NTH_RE = re.compile(r"第([零一二三四五六七八九十\d]+)次")


def cn_number(n: int) -> str:
    if n <= 10:
        return "十" if n == 10 else _CN_DIGITS[max(0, n)]
    if n < 20:
        return "十" + _CN_DIGITS[n - 10]
    if n < 100:
        return _CN_DIGITS[n // 10] + "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    return str(n)


def parse_nth(name) -> int:
    m = _NTH_RE.search(str(name or ""))
    if not m:
        return 0
    t = m.group(1)
    if t.isdigit():
        return int(t)
    if "十" in t:
        a, _, b = t.partition("十")
        return (_CN_DIGITS.index(a) if a else 1) * 10 + (_CN_DIGITS.index(b) if b else 0)
    return _CN_DIGITS.index(t)


def default_sheet_name(started_at: str, earlier_names=()) -> str:
    """民國年.月.日第幾次。次數接著前面的紀錄算，使用者改過前一筆，後面就跟著對。"""
    try:
        d = datetime.datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        d = datetime.datetime.now()
    nth = max([parse_nth(n) for n in earlier_names] or [0]) + 1
    return "%d.%d.%d第%s次" % (d.year - 1911, d.month, d.day, cn_number(nth))


_BAD_SHEET_RE = re.compile(r"[\[\]:*?/\\]")


def clean_sheet_name(name) -> str:
    """Excel 工作表名稱不能有這幾個符號，也不能超過 31 字。"""
    return _BAD_SHEET_RE.sub("-", str(name or "")).strip().strip("'")[:31]


def load(session: dict):
    try:
        return json.loads(session.get("meeting_notes") or "null")
    except ValueError:
        return None


# ── Excel ─────────────────────────────────────────────────────────────
# 跟使用者原本的整理表一樣：五欄、標楷體、標題灰底
_WIDTHS = (53.125, 48.375, 43.625, 55.125, 36.625)
_HEAD_FILL = "D0CECE"
_LINE_PT = 16.2


def _row_height(texts) -> float:
    lines = 1
    for text, width in zip(texts, _WIDTHS):
        # 欄寬以半形字計，12pt 標楷體全形字約佔 2.3 格
        lines = max(lines, math.ceil(len(text or "") * 2.3 / width) or 1)
    return lines * _LINE_PT


def build_xlsx(sheets) -> bytes:
    """sheets：[(工作表名稱, notes)]，一次會議一張。"""
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    used = set()
    for name, notes in sheets:
        name = clean_sheet_name(name) or "會議記錄"
        base, n = name, 2
        while name.lower() in used:
            name = "%s(%d)" % (base[:27], n)
            n += 1
        used.add(name.lower())
        ws = wb.create_sheet(name)
        for col, width in enumerate(_WIDTHS, 1):
            ws.column_dimensions[get_column_letter(col)].width = width
        for col, (_, title, _, _) in enumerate(COLUMNS, 1):
            c = ws.cell(row=1, column=col, value=title)
            c.font = Font(name="標楷體", size=16, bold=True)
            c.fill = PatternFill("solid", fgColor=_HEAD_FILL)
            c.alignment = Alignment(wrap_text=True)
        ws.row_dimensions[1].height = 22.2
        cols = []
        for key in COLUMN_KEYS:
            items = (notes.get("columns") or {}).get(key) or []
            # 只有一則就不用編號
            cols.append([it["text"] if len(items) == 1 else "%d.%s" % (i, it["text"])
                         for i, it in enumerate(items, 1)])
        body = Font(name="標楷體", size=12)
        for r in range(max([len(c) for c in cols] or [0])):
            texts = [c[r] if r < len(c) else None for c in cols]
            for col, text in enumerate(texts, 1):
                if text:
                    cell = ws.cell(row=r + 2, column=col, value=text)
                    cell.font = body
                    cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
            ws.row_dimensions[r + 2].height = _row_height(texts)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
