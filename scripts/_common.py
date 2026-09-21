"""驗證腳本共用工具（規格 §14）。

設計原則：逐項印 PASS/FAIL、最後一行印總結、
exit code 0 = 全通過，所有實測數字合併寫入 data/bench.json（不覆蓋既有欄位）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_PATH = ROOT / "data" / "bench.json"

# Windows 主控台預設 cp950，中文輸出會變亂碼
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"

if os.name == "nt" and not os.getenv("WT_SESSION"):
    try:
        import colorama  # type: ignore
        colorama.just_fix_windows_console()
    except Exception:
        _GREEN = _RED = _YELLOW = _DIM = _RESET = ""


def load_bench() -> dict:
    if not BENCH_PATH.exists():
        return {}
    try:
        with BENCH_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def merge_bench(patch: dict) -> dict:
    """合併更新 bench.json，不覆蓋既有欄位（規格 §14）。"""
    data = load_bench()
    _deep_merge(data, patch)
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCH_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def note_deviation(key: str, expected: float, measured: float,
                   threshold: float = 0.30) -> None:
    """實測與規格書估計偏離超過門檻時標註（規格 §0.1）。"""
    if not expected:
        return
    dev = abs(measured - expected) / float(expected)
    if dev <= threshold:
        return
    merge_bench({"deviations": {key: {
        "spec_estimate": expected, "measured": measured,
        "deviation_pct": round(dev * 100, 1),
        "note": "以實測為準，未調整至規格書估計值",
    }}})


class Gate:
    """逐項 PASS/FAIL 收集器。"""

    def __init__(self, name: str):
        self.name = name
        self.items = []      # (ok, label, detail)

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.items.append((bool(ok), label, detail))
        tag = (_GREEN + "PASS" + _RESET) if ok else (_RED + "FAIL" + _RESET)
        line = "  [%s] %s" % (tag, label)
        if detail:
            line += "\n         " + _DIM + detail + _RESET
        print(line, flush=True)
        return bool(ok)

    def skip(self, label: str, reason: str) -> None:
        """無法執行但不算失敗的項目（例如缺 fixture）。仍會讓閘門不通過。"""
        self.items.append((False, label, "SKIP: " + reason))
        print("  [%sSKIP%s] %s\n         %s%s%s"
              % (_YELLOW, _RESET, label, _DIM, reason, _RESET), flush=True)

    def info(self, msg: str) -> None:
        print("  " + _DIM + msg + _RESET, flush=True)

    @property
    def failed(self):
        return [i for i in self.items if not i[0]]

    def finish(self) -> int:
        n = len(self.items)
        bad = self.failed
        print()
        if not bad:
            print("%s%s: %d/%d PASS — 閘門通過%s"
                  % (_GREEN, self.name, n, n, _RESET))
            return 0
        print("%s%s: %d/%d PASS，%d 項未通過 — 閘門未通過%s"
              % (_RED, self.name, n - len(bad), n, len(bad), _RESET))
        for _, label, detail in bad:
            print("   - " + label + ((" (" + detail + ")") if detail else ""))
        print("規格 §11：閘門未通過就不要開始下一個里程碑，請先修正上列項目。")
        return 1


def header(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68, flush=True)


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def find_fixture(*names):
    d = ROOT / "tests" / "fixtures"
    for n in names:
        p = d / n
        if p.exists():
            return p
    return None
