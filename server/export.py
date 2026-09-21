"""Markdown / TXT / JSON 匯出（規格 §6.4、§10）。"""
from __future__ import annotations

import json

from .summarizer import fmt_ts


def _date_of(started_at: str) -> str:
    return (started_at or "")[:10]


def build_markdown(course, session, sections, final) -> str:
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
        lines.append("### %s `[%s]`" % (sec.get("title", "未命名段落"),
                                        fmt_ts(sec.get("start_s", 0.0))))
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
