"""
check_settling.py
==================
Is the \\SI{20}{\\second} settling period long enough?

The manuscript states that a follower released away from equilibrium oscillates
with a decay time of roughly \\SI{25}{\\second}, and then discards only
\\SI{20}{\\second}. Taken together those two numbers invite the obvious
objection that the recorded window still opens inside the transient. The
settling period itself is not retained, so the question cannot be answered from
the discarded frames; it can be answered from the recorded ones, which is what
matters. If a transient survived into the recording, the opening seconds of a
scene would sit systematically further from the equilibrium gap than its later
seconds, and would drift in one direction.

We therefore compare, per scene, the first and last \\SI{5}{\\second} of
recorded injection-free driving on two statistics: the absolute deviation from
the scene's own median spacing, and the signed drift. A residual transient
shows up as a larger opening deviation and a consistent drift sign across
scenes; its absence bounds what any residual could be.

    python check_settling.py --dataset <root> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

logger = logging.getLogger("settling")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] settling: %(message)s")

HZ = 10
WIN = 5 * HZ           # 5 s comparison windows
MIN_RUN = 15 * HZ      # need enough clean frames to hold two windows apart


def longest_clean_run(frames: List[dict]) -> tuple:
    best = (0, 0, 0)
    start = None
    for i, fr in enumerate(frames + [None]):
        active = fr is not None and fr["kinematics"].get("injection_active")
        if fr is not None and not active:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start > best[0]:
                    best = (i - start, start, i)
                start = None
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.dataset)
    open_dev, late_dev, drifts, opens_at = [], [], [], []
    for split in ("train", "val", "test"):
        f = root / "infos" / f"cf_infos_{split}.json"
        if not f.exists():
            continue
        for sc in json.load(open(f))["data"]:
            fr = sc["scene_data"]
            n, a, b = longest_clean_run(fr)
            if n < MIN_RUN:
                continue
            # Only scenes whose clean run starts at the very beginning of the
            # recording can speak to settling: a run starting later has had
            # extra simulated time to relax.
            if a > 2:
                continue
            s = np.array([x["kinematics"]["spacing_m"] for x in fr[a:b]])
            med = float(np.median(s))
            open_dev.append(float(np.abs(s[:WIN] - med).mean()))
            late_dev.append(float(np.abs(s[-WIN:] - med).mean()))
            drifts.append(float(s[:WIN].mean() - s[-WIN:].mean()))
            opens_at.append(float(s[0] - med))

    od, ld, dr = map(np.asarray, (open_dev, late_dev, drifts))
    logger.info("%d scenes whose clean run starts at recording onset", len(od))

    # Paired test: a surviving transient makes the opening window deviate more.
    w = stats.wilcoxon(od, ld) if len(od) > 10 else None
    # And a one-sample test on drift: a transient decays in a consistent
    # direction, so its mean would differ from zero.
    t = stats.wilcoxon(dr) if len(dr) > 10 else None

    out = {
        "n_scenes": int(len(od)),
        "window_s": WIN / HZ,
        "mean_abs_dev_first_5s_m": round(float(od.mean()), 3),
        "mean_abs_dev_last_5s_m": round(float(ld.mean()), 3),
        "median_abs_dev_first_5s_m": round(float(np.median(od)), 3),
        "median_abs_dev_last_5s_m": round(float(np.median(ld)), 3),
        "paired_wilcoxon_p": float(w.pvalue) if w else None,
        "mean_drift_first_minus_last_m": round(float(dr.mean()), 3),
        "drift_wilcoxon_p": float(t.pvalue) if t else None,
        "frac_scenes_drift_positive": round(float((dr > 0).mean()), 3),
        "mean_open_offset_from_median_m": round(float(np.mean(opens_at)), 3),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    for k, v in out.items():
        print(f"  {k:<34}{v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
