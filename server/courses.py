"""課程設定檔載入（規格 §9.3）。

asr_prompt 給 Whisper 當 initial_prompt；glossary 注入摘要 LLM 的
system prompt。兩者共用同一份 YAML，維護一處即可。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import config

log = logging.getLogger(__name__)

# Whisper 的 prompt 上限是 224 token，留安全邊界（規格 §4.3）
ASR_PROMPT_MAX_TOKENS = 200


@dataclass
class Course:
    id: str
    name: str
    instructor: str = ""
    asr_prompt: str = ""
    glossary: list = field(default_factory=list)
    # 手抄版的欄位結構。有些課要求邊上課邊填學習單，下課就交，
    # 欄位是固定的（例如「今天上課內容」「今日反思」）。留空就是
    # 一般的主題式手抄版。
    handcopy_sections: list = field(default_factory=list)
    handcopy_target_chars: int = 0

    @property
    def glossary_line(self) -> str:
        return "、".join(self.glossary)

    @property
    def is_worksheet(self) -> bool:
        return bool(self.handcopy_sections)


def _estimate_tokens(text: str) -> int:
    """粗估 token 數：CJK 約 1 token/字，其餘每 3 字元約 1 token。

    拿不到 Whisper tokenizer 時的保守估計，只用於截斷警告。
    """
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    rest = len(text) - cjk
    return cjk + rest // 3


def _truncate_prompt(prompt: str, course_id: str) -> str:
    prompt = prompt.strip()
    if _estimate_tokens(prompt) <= ASR_PROMPT_MAX_TOKENS:
        return prompt
    lines = prompt.splitlines()
    while lines and _estimate_tokens("\n".join(lines)) > ASR_PROMPT_MAX_TOKENS:
        lines.pop()
    log.warning("課程 %s 的 asr_prompt 超過 %d token 估計上限，已截斷尾端",
                course_id, ASR_PROMPT_MAX_TOKENS)
    return "\n".join(lines).strip()


def _parse_handcopy_sections(raw, course_id: str) -> list:
    """把 YAML 的 handcopy.sections 正規化成 [{id, title, hint, pick}]。

    pick 表示這欄要產幾則候選讓使用者自己挑——反思、想法這類欄位寫的是
    使用者自己的感受，模型只能從課堂內容生素材，不能代替他下結論。
    """
    out = []
    for i, item in enumerate(raw or []):
        if isinstance(item, str):
            item = {"title": item}
        if not isinstance(item, dict):
            log.warning("課程 %s 的 handcopy.sections 第 %d 項格式不對，已略過",
                        course_id, i + 1)
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        count = item.get("count")
        if isinstance(count, (list, tuple)) and len(count) == 2:
            count = (int(count[0]), int(count[1]))
        else:
            count = None
        chars = item.get("chars")
        if isinstance(chars, (list, tuple)) and len(chars) == 2:
            chars = (int(chars[0]), int(chars[1]))
        else:
            chars = None
        out.append({
            "id": str(item.get("id") or ("s%d" % (i + 1))),
            "title": title,
            "hint": str(item.get("hint") or "").strip(),
            "pick": int(item.get("pick") or 0),
            # count 直接指定則數上下限，蓋過由 pick 推出來的預設值
            "count": count,
            # chars 指定每則的字數範圍。反思這種要寫出推論過程的欄位，
            # 用預設的長度會被壓成一句結論。
            "chars": chars,
            # style=detail 代表這欄要寫細節與名詞解釋，不是條列標題
            "style": str(item.get("style") or "").strip(),
        })
    return out


def load_course(course_id: str, courses_dir=None) -> Course:
    d = Path(courses_dir or config.COURSES_DIR)
    path = d / (course_id + ".yaml")
    if not path.exists():
        path = d / (course_id + ".yml")
    if not path.exists():
        raise FileNotFoundError("找不到課程設定檔：" + str(d / (course_id + ".yaml")))
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hc = raw.get("handcopy") or {}
    return Course(
        id=str(raw.get("id", course_id)),
        name=str(raw.get("name", course_id)),
        instructor=str(raw.get("instructor", "")),
        asr_prompt=_truncate_prompt(str(raw.get("asr_prompt", "")), course_id),
        glossary=[str(g) for g in (raw.get("glossary") or [])],
        handcopy_sections=_parse_handcopy_sections(hc.get("sections"), course_id),
        handcopy_target_chars=int(hc.get("target_chars") or 0),
    )


@lru_cache(maxsize=32)
def get_course(course_id: str) -> Course:
    return load_course(course_id)


def list_courses(courses_dir=None):
    d = Path(courses_dir or config.COURSES_DIR)
    out = []
    for p in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
        try:
            c = load_course(p.stem, d)
        except Exception:
            continue
        out.append({"id": c.id, "name": c.name, "instructor": c.instructor})
    return out
