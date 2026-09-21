"""Markdown / TXT / JSON 匯出（規格 §6.4、§10）。"""
from __future__ import annotations

import json
import re

from .summarizer import fmt_ts


def _date_of(started_at: str) -> str:
    return (started_at or "")[:10]


def _slide_tag(align, start_s, end_s) -> str:
    """這一段對到的投影片頁碼，寫在段落標題後面。

    有對到才寫。對不上是常態（老師離題、投影片只有圖、講的是別份教材），
    硬標一個頁碼比不標更糟——使用者會照著去翻，然後發現不是那頁。
    """
    if not align or not align.get("pages"):
        return ""
    from . import materials
    hits = materials.pages_for_span(align["pages"], start_s, end_s)
    if not hits:
        return ""
    parts = []
    for h in hits:
        ps = h["pages"]
        rng = ("p%d" % ps[0]) if len(ps) == 1 else "p%d–p%d" % (ps[0], ps[-1])
        parts.append(rng)
    return "　投影片 " + "、".join(parts)


NL = chr(10)
_HEAD_RE = re.compile(r"^(###\s+.*?)\s*`\[(\d{2}):(\d{2}):(\d{2})\]`\s*(.*)$")


def annotate_slides(md: str, align) -> str:
    """把投影片頁碼補進**已經存好的**筆記裡。

    完整版在課後整理時就寫進資料庫了，那時候還沒對照過投影片。
    重新產生一份要再跑一次模型，而且 overview 沒有單獨存下來、重建
    會失真——所以改成照時間戳後製，舊的紀錄也適用。
    """
    if not align or not align.get("pages") or not md:
        return md
    from . import materials

    lines = md.splitlines()
    heads = []          # (行號, 秒數)
    for i, ln in enumerate(lines):
        m = _HEAD_RE.match(ln)
        if m:
            heads.append((i, int(m.group(2)) * 3600 + int(m.group(3)) * 60
                          + int(m.group(4))))
    if not heads:
        return md
    last = max(e["end_s"] for e in align["pages"])
    for n, (i, start) in enumerate(heads):
        end = heads[n + 1][1] if n + 1 < len(heads) else max(last, start + 1)
        hits = materials.pages_for_span(align["pages"], start, end)
        if not hits:
            continue
        parts = []
        for h in hits:
            ps = h["pages"]
            parts.append("p%d" % ps[0] if len(ps) == 1
                         else "p%d–p%d" % (ps[0], ps[-1]))
        m = _HEAD_RE.match(lines[i])
        if "投影片" in (m.group(5) or ""):
            continue
        lines[i] = "%s `[%s]`　投影片 %s" % (
            m.group(1), "%s:%s:%s" % (m.group(2), m.group(3), m.group(4)),
            "、".join(parts))
    return NL.join(lines) + (NL if md.endswith(NL) else "")


def build_markdown(course, session, sections, final, align=None) -> str:
    """組出規格 §6.4 指定的 Markdown 格式。"""
    lines = []
    lines.append("# %s — %s" % (course.name, _date_of(session.get("started_at", ""))))
    lines.append("")

    lines.append("## 本堂重點")
    # 要寫成清單項目。純文字一行一句在 Markdown 裡會被當成同一段的軟換行，
    # 十幾條重點會黏成一大段；轉成 Word 也只會是一堆 Normal 段落。
    for s in (final.get("overview") or []):
        lines.append("- " + (s if s.endswith(("。", "！", "？", ".")) else s + "。"))
    lines.append("")

    lines.append("## 詳細筆記")
    for sec in sections:
        lines.append("")
        lines.append("### %s `[%s]`%s"
                     % (sec.get("title", "未命名段落"),
                        fmt_ts(sec.get("start_s", 0.0)),
                        _slide_tag(align, sec.get("start_s", 0.0),
                                   sec.get("end_s", 0.0))))
        # 導言：讀的人先知道這段在講什麼，再看細節
        if sec.get("summary"):
            lines.append("")
            lines.append(str(sec["summary"]))
        # 分子題。二十幾個同一層級的句子不是筆記，那只是把逐字稿剁碎；
        # 舊紀錄沒有分組時 storage 會給一組空標題的，走同一條路徑。
        groups = sec.get("groups") or [{"heading": "",
                                        "points": sec.get("bullets") or []}]
        for g in groups:
            head = str(g.get("heading") or "").strip()
            pts = [str(x) for x in (g.get("points") or []) if str(x).strip()]
            if not pts:
                continue
            lines.append("")
            if head:
                lines.append("#### " + head)
            for b in pts:
                lines.append("- " + b)
        if sec.get("user_note"):
            lines.append("")
            lines.append("> 使用者標註：%s" % sec["user_note"])
    lines.append("")

    oq = final.get("open_questions") or []
    if oq:
        lines.append("## 待釐清")
        for q in oq:
            lines.append("- " + str(q))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_CN_NUM = "〇一二三四五六七八九十"


def build_handcopy(course, session, notes) -> str:
    """手抄版：使用者會照著抄在紙上，所以刻意壓短、層次淺
    （長度由 config.HANDCOPY_TARGET_CHARS 控制，隨逐字稿長度調整）。

    與 build_markdown 的差別是用途不同——那份是課後複習用的完整紀錄，
    這份是下課前要抄完交出去的。抄不完的筆記等於沒有用。
    """
    lines = ["# %s — %s" % (course.name, _date_of(session.get("started_at", "")))]
    exam = notes.get("exam") or []
    if exam:
        lines.append("")
        lines.append("## ★ 老師說會考")
        for e in exam:
            lines.append("- " + str(e))
    worksheet = bool(notes.get("worksheet"))
    for i, t in enumerate(notes.get("topics") or [], 1):
        lines.append("")
        title = str(t.get("title", ""))
        if worksheet:
            # 學習單的欄位是固定的，標上編號讓使用者對得上紙本的欄位；
            # pick 是「這欄實際只要抄幾則」，其餘是給他挑的備選。
            pick = int(t.get("pick") or 0)
            n = len(t.get("points") or [])
            hint = "（挑 %d 則抄，共 %d 則可選）" % (pick, n) if pick and n > pick else ""
            lines.append("## %s、%s%s" % (_CN_NUM[i] if i < len(_CN_NUM) else i,
                                          title, hint))
            if not (t.get("points") or []):
                lines.append("- （這欄沒產出內容，請按重新產生）")
        else:
            lines.append("## " + title)
        for p in (t.get("points") or []):
            lines.append("- " + str(p))
    body = "\n".join(lines)
    n = len(body.replace("\n", "").replace(" ", ""))
    return body.rstrip() + "\n\n<!-- 約 %d 字 -->\n" % n


def build_txt(session, segments) -> str:
    lines = ["# %s  %s" % (session.get("id", ""), session.get("started_at", "")), ""]
    for s in segments:
        lines.append("[%s] %s" % (fmt_ts(s.get("start_s", 0.0)), s.get("text", "")))
    return "\n".join(lines) + "\n"


def build_json(session, segments, sections) -> str:
    return json.dumps({
        "session": session,
        "segments": segments,
        "sections": sections,
    }, ensure_ascii=False, indent=2)
