"""依序執行所有 verify 腳本並彙總（規格 §14.7）。改動後的回歸測試入口。

用法：
    python scripts/verify_all.py                # M0~M4 全跑
    python scripts/verify_all.py --up-to m1     # 只跑到 G1
    python scripts/verify_all.py --skip m2      # 跳過需要服務的 G2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from _common import ROOT, header

STAGES = ["m0", "m1", "m2", "m3", "m4"]
EXTRA_ARGS = {
    # G1 完整 3 小時是驗收要求；回歸測試用 --hours 讓使用者自己決定
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-to", default="m4", choices=STAGES)
    ap.add_argument("--skip", default="", help="逗號分隔，例如 m2,m3")
    ap.add_argument("--m1-hours", default=None,
                    help="傳給 verify_m1.py 的 --hours（預設 3 小時）")
    ap.add_argument("--m1-speed", default=None,
                    help="傳給 verify_m1.py 的 --speed；非 1.0 時延遲項會 SKIP")
    ap.add_argument("--selftest", action="store_true",
                    help="先跑 scripts/selftest.py（純邏輯，不需模型／GPU）")
    args, passthrough = ap.parse_known_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    stop = STAGES.index(args.up_to)
    results = []

    if args.selftest:
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "selftest.py")],
                             cwd=str(ROOT))
        results.append(("selftest", rc, 0.0))
        if rc != 0:
            print("\nselftest 未通過，先修純邏輯層再往下。")
            _summary(results)
            return 1

    for stage in STAGES[:stop + 1]:
        if stage in skip:
            results.append((stage, None, 0.0))
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / ("verify_%s.py" % stage))]
        if stage == "m1" and args.m1_hours:
            cmd += ["--hours", args.m1_hours]
        if stage == "m1" and args.m1_speed:
            cmd += ["--speed", args.m1_speed]
        cmd += passthrough
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        results.append((stage, rc, time.time() - t0))
        if rc != 0:
            print("\n閘門 G%s 未通過，停止後續里程碑（規格 §11）。"
                  % stage[1:])
            break

    return _summary(results)


def _summary(results):
    header("verify_all.py 彙總")
    failed = 0
    for stage, rc, secs in results:
        label = stage if not stage.startswith("m") else "G" + stage[1:]
        if rc is None:
            print("  %-8s SKIP" % label)
            continue
        failed += (rc != 0)
        print("  %-8s %s  (%.0fs)" % (label, "PASS" if rc == 0 else "FAIL", secs))
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
