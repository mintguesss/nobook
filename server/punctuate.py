"""幫逐字稿加標點：語音模型輸出的中文幾乎沒有標點，一整段字黏在一起很難讀。

只插標點，原文的字一個都不動。模型吃簡體時斷句比較準，所以先轉簡體給模型判斷，
再把標點放回原本的繁體文字。全程跑在 CPU，不佔 VRAM；模型缺席時 available()
回 False，逐字稿照常產生，只是沒有標點。
"""
from __future__ import annotations

import logging
import re

from . import config

log = logging.getLogger(__name__)

MODEL_DIR = config.MODELS_DIR / "punctuation"
MODEL = MODEL_DIR / "model.onnx"
MODEL_NAME = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "punctuation-models/" + MODEL_NAME + ".tar.bz2")

_THREADS = 4
_MARKS = set("，。？！、；：,.?!;:")
_SPACE_BY_MARK_RE = re.compile(r"\s+(?=[，。？！、；：])|(?<=[，。？！、；：])\s+")


def available() -> bool:
    if not MODEL.exists():
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def transfer_marks(original: str, punctuated: str) -> str:
    """把 punctuated 裡的標點放回 original。字數對不起來就不動原文，寧可沒標點也不要弄壞字。"""
    gaps = {}
    seen = 0
    for ch in punctuated:
        if ch in _MARKS:
            gaps[seen] = gaps.get(seen, "") + ch
        elif not ch.isspace():
            seen += 1
    if seen != sum(1 for ch in original if not ch.isspace()):
        return original
    out = [gaps.get(0, "")]
    seen = 0
    for ch in original:
        out.append(ch)
        if not ch.isspace():
            seen += 1
            out.append(gaps.get(seen, ""))
    return _SPACE_BY_MARK_RE.sub("", "".join(out))


def apply_to_rows(rows, add) -> None:
    """add: 文字 → 加好標點的文字。同一人的下一列是接著講的，句尾不收成句號。"""
    for i, r in enumerate(rows):
        text = transfer_marks(r["text"], add(r["text"]))
        if i + 1 < len(rows) and rows[i + 1].get("speaker") == r.get("speaker"):
            text = text.rstrip("。")
        r["text"] = text


def _to_simplified():
    try:
        from opencc import OpenCC
        return OpenCC("t2s").convert
    except Exception as e:
        log.warning("OpenCC 不可用（%s），標點直接用原文判斷", e)
        return lambda text: text


def punctuate_rows(rows) -> None:
    """就地幫 rows 加標點。模型約 300 MB，用完就放掉，不常駐記憶體。"""
    import sherpa_onnx
    model = sherpa_onnx.OfflinePunctuation(sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(
            ct_transformer=str(MODEL), num_threads=_THREADS)))
    simplify = _to_simplified()
    apply_to_rows(rows, lambda text: model.add_punctuation(simplify(text)))
