"""
validate_styles.py
===================
Are the three advertised driving styles empirically distinct in the executed
trajectories?

Style is a design input: it selects a parameter set and a pair of multipliers in
the calibration layer. Whether it survives to the recorded behaviour is a
separate, testable question, and the dataset's claim to span three styles rests
on the answer. The parameter-source ablation already found style recoverable
from raw parameters at \\num{0.93} accuracy but only \\num{0.46} from executed
trajectories; this quantifies that per behavioural variable.

Per scene we compute six summaries a traffic engineer would use to characterise
a following style: mean time headway, peak acceleration, peak deceleration,
mean absolute jerk, minimum time-to-collision, and mean spacing. For each we
report the per-style distribution, an omnibus Kruskal--Wallis test across the
three styles, and pairwise Cliff's delta with Holm-corrected Mann--Whitney
tests.

Cliff's delta is used rather than Cohen's d because these distributions are
skewed and bounded; delta is the probability that a random scene of one style
exceeds a random scene of another, rescaled to [-1, 1], and needs no normality
assumption. The conventional reading is |d| < 0.147 negligible, < 0.33 small,
< 0.474 medium, else large.

    python validate_styles.py --dataset <root> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

logger = logging.getLogger("styles")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] styles: %(message)s")

STYLES = ["cautious", "normal", "aggressive"]
DT = 0.1
SMOOTH = 5


def _smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def scene_metrics(k: List[dict]) -> Dict[str, float]:
    s = np.array([f["kinematics"]["spacing_m"] for f in k])
    v = np.array([f["kinematics"]["ego_speed_mps"] for f in k])
    dv = np.array([f["kinematics"]["relative_speed_mps"] for f in k])
    a = np.gradient(_smooth(v), DT)
    j = np.gradient(_smooth(a), DT)
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = s[v > 0.5] / v[v > 0.5]
        closing = -dv
        ttc = np.where(closing > 0.1, s / np.maximum(closing, 1e-6), np.inf)
    return {
        "mean_thw_s": float(np.mean(thw)) if thw.size else np.nan,
        "peak_accel_mps2": float(np.percentile(a, 95)),
        "peak_decel_mps2": float(-np.percentile(a, 5)),
        "mean_abs_jerk": float(np.mean(np.abs(j))),
        "min_ttc_s": float(np.min(ttc[np.isfinite(ttc)])) if np.isfinite(ttc).any() else np.nan,
        "mean_spacing_m": float(np.mean(s)),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b), computed by rank rather than by O(n*m) pairs."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    r = stats.rankdata(allv)
    ra = r[:n].sum()
    u = ra - n * (n + 1) / 2.0            # Mann-Whitney U for a
    return float(2.0 * u / (n * m) - 1.0)


def magnitude(d: float) -> str:
    x = abs(d)
    return ("negligible" if x < 0.147 else "small" if x < 0.33
            else "medium" if x < 0.474 else "large")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--injection-free-only", action="store_true",
                   help="restrict to scenes with no injected frames")
    args = p.parse_args()

    root = Path(args.dataset)
    rows: Dict[str, List[Dict[str, float]]] = {s: [] for s in STYLES}
    for split in ("train", "val", "test"):
        f = root / "infos" / f"cf_infos_{split}.json"
        if not f.exists():
            continue
        for sc in json.load(open(f))["data"]:
            st = sc["scene_info"].get("driving_style")
            if st not in rows:
                continue
            k = sc["scene_data"]
            if args.injection_free_only and any(
                    fr["kinematics"].get("injection_active") for fr in k):
                continue
            rows[st].append(scene_metrics(k))
    logger.info("scenes per style: %s", {s: len(v) for s, v in rows.items()})

    metrics = ["mean_thw_s", "peak_accel_mps2", "peak_decel_mps2",
               "mean_abs_jerk", "min_ttc_s", "mean_spacing_m"]
    out = {"n_per_style": {s: len(v) for s, v in rows.items()},
           "injection_free_only": args.injection_free_only, "metrics": []}

    for m in metrics:
        vals = {s: np.array([r[m] for r in rows[s] if np.isfinite(r[m])])
                for s in STYLES}
        if min(len(v) for v in vals.values()) < 5:
            continue
        H, pH = stats.kruskal(*[vals[s] for s in STYLES])
        entry = {"metric": m,
                 "median": {s: round(float(np.median(vals[s])), 3) for s in STYLES},
                 "iqr": {s: [round(float(np.percentile(vals[s], 25)), 3),
                             round(float(np.percentile(vals[s], 75)), 3)]
                         for s in STYLES},
                 "kruskal_H": round(float(H), 2),
                 "kruskal_p": float(pH), "pairs": []}
        raw = []
        for a, b in combinations(STYLES, 2):
            U, pu = stats.mannwhitneyu(vals[a], vals[b], alternative="two-sided")
            raw.append((a, b, pu, cliffs_delta(vals[a], vals[b])))
        # Holm correction across the three pairwise tests of this metric.
        order = np.argsort([r[2] for r in raw])
        adj = [0.0] * len(raw)
        for rank, idx in enumerate(order):
            adj[idx] = min(1.0, raw[idx][2] * (len(raw) - rank))
        for k2 in range(1, len(order)):
            adj[order[k2]] = max(adj[order[k2]], adj[order[k2 - 1]])
        for (a, b, pu, d), pa in zip(raw, adj):
            entry["pairs"].append({"a": a, "b": b, "p_holm": float(pa),
                                   "cliffs_delta": round(d, 3),
                                   "magnitude": magnitude(d),
                                   "significant": bool(pa < 0.05)})
        out["metrics"].append(entry)

    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\nscenes per style: {out['n_per_style']}")
    hdr = (f"{'metric':<18}{'cautious':>12}{'normal':>12}{'aggressive':>12}"
           f"{'Kruskal p':>12}{'largest |delta|':>17}")
    print(hdr); print("-" * len(hdr))
    for e in out["metrics"]:
        big = max(e["pairs"], key=lambda x: abs(x["cliffs_delta"]))
        print(f"{e['metric']:<18}"
              f"{e['median']['cautious']:>12.2f}{e['median']['normal']:>12.2f}"
              f"{e['median']['aggressive']:>12.2f}{e['kruskal_p']:>12.1e}"
              f"{abs(big['cliffs_delta']):>10.2f} {big['magnitude']:<6}")
    print("\nordered pairs (cautious vs aggressive) - the contrast the taxonomy claims:")
    for e in out["metrics"]:
        ca = [x for x in e["pairs"] if {x["a"], x["b"]} == {"cautious", "aggressive"}][0]
        print(f"  {e['metric']:<18} delta={ca['cliffs_delta']:+.3f} "
              f"({ca['magnitude']}, Holm p={ca['p_holm']:.1e}"
              f"{'' if ca['significant'] else ', n.s.'})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
