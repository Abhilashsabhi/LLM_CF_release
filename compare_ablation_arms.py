"""
compare_ablation_arms.py
=========================
Behavioural comparison of the parameter-source arms after identical calibration
and execution.

Parameter-level differences are only interesting if they survive Stage 3, which
overwrites the ego desired speed, the ego time headway and the initial
conditions. This script reads what each arm actually produced and asks whether
the source is still visible in the trajectories.

Metrics
-------
  yield             fraction of scenes passing the pre-specified admission
                    criterion -- the cost of a parameter source in wasted
                    simulation
  W1                Wasserstein-1 distance to the regime's reference, on
                    admitted injection-free frames, exactly as Stage 4 scores it
  TTC<3s            safety-critical frame rate: does the source generate the
                    content the dataset exists to provide
  trajectory        mean pairwise distance between scene-level summary vectors:
  diversity         does the source produce varied behaviour, or one behaviour
  style recovery    leave-one-out accuracy of a nearest-centroid classifier
                    predicting driving style from EXECUTED trajectory
                    statistics. This is the decisive test: if a source's style
                    distinction does not survive calibration, it contributed
                    nothing the pipeline could not have supplied itself.

    python compare_ablation_arms.py --runs <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("cmp")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] cmp: %(message)s")

REF = {"urban": {"spacing": 18.0, "ego_speed": 9.0},
       "highway": {"spacing": 22.0, "ego_speed": 28.0}}
STYLES = ["cautious", "normal", "aggressive"]


def w1(sample: np.ndarray, ref_mean: float, ref_std: float,
       n: int = 20000, seed: int = 0) -> float:
    """W1 against a normal reference with the given mean and spread."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(ref_mean, ref_std, n)
    a = np.sort(rng.choice(sample, size=min(n, len(sample)), replace=len(sample) < n))
    b = np.sort(ref)[:len(a)]
    return float(np.mean(np.abs(a - np.sort(rng.choice(ref, len(a), replace=False)))))


def load_arm(root: Path) -> Dict:
    scenes = []
    for split in ("train", "val", "test"):
        f = root / "infos" / f"cf_infos_{split}.json"
        if f.exists():
            scenes += json.load(open(f))["data"]
    return scenes


def summarise(scenes: List[dict]) -> Dict:
    per_regime = {r: {"spacing": [], "speed": []} for r in REF}
    feats, styles = [], []
    crit = tot = 0
    for sc in scenes:
        info = sc["scene_info"]
        reg = info.get("regime")
        k = sc["scene_data"]
        sp = np.array([f["kinematics"]["spacing_m"] for f in k])
        ve = np.array([f["kinematics"]["ego_speed_mps"] for f in k])
        ac = np.array([f["kinematics"]["ego_acceleration_mps2"] for f in k])
        tt = np.array([f["kinematics"]["ttc_s"] for f in k])
        inj = np.array([bool(f["kinematics"].get("injection_active")) for f in k])
        crit += int((tt < 3.0).sum()); tot += len(tt)
        if reg in per_regime and (~inj).any():
            per_regime[reg]["spacing"] += list(sp[~inj])
            per_regime[reg]["speed"] += list(ve[~inj])
        # Scene-level behavioural signature.
        feats.append([sp.mean(), sp.std(), ve.mean(), ve.std(),
                      np.abs(ac).mean(), ac.std(),
                      float(np.percentile(tt[np.isfinite(tt)], 5)) if np.isfinite(tt).any() else 0.0])
        styles.append(STYLES.index(info["driving_style"])
                      if info.get("driving_style") in STYLES else -1)

    X = np.asarray(feats, dtype=float)
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    y = np.asarray(styles)

    idx = list(range(len(Xn)))
    sub = idx if len(idx) <= 400 else list(np.random.default_rng(0).choice(idx, 400, replace=False))
    div = float(np.mean([np.linalg.norm(Xn[i] - Xn[j]) for i, j in combinations(sub, 2)]))

    ok = y >= 0
    correct = 0
    for i in np.flatnonzero(ok):
        cents = []
        for kcls in range(3):
            j = [t for t in np.flatnonzero(ok) if y[t] == kcls and t != i]
            cents.append(Xn[j].mean(axis=0))
        correct += int(np.argmin([np.linalg.norm(Xn[i] - c) for c in cents]) == y[i])
    style_acc = correct / max(int(ok.sum()), 1)

    out = {"n_scenes": len(scenes),
           "ttc_lt3s_pct": round(100 * crit / max(tot, 1), 3),
           "trajectory_diversity": round(div, 3),
           "style_recovery_acc": round(style_acc, 3),
           "style_chance": round(1 / 3, 3)}
    for r in REF:
        s = np.asarray(per_regime[r]["spacing"]); v = np.asarray(per_regime[r]["speed"])
        if len(s):
            out[f"{r}_spacing_mean"] = round(float(s.mean()), 2)
            out[f"{r}_speed_mean"] = round(float(v.mean()), 2)
            out[f"{r}_w1_spacing"] = round(w1(s, REF[r]["spacing"], 0.35 * REF[r]["spacing"]), 3)
            out[f"{r}_w1_speed"] = round(w1(v, REF[r]["ego_speed"], 0.25 * REF[r]["ego_speed"]), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+",
                   default=["llm", "uniform", "lhs", "handcrafted"])
    args = p.parse_args()

    runs = Path(args.runs)
    rows = []
    for arm in args.arms:
        root = runs / arm
        rep = json.load(open(root / "infos" / "validation_report.json"))
        scenes = load_arm(root)
        logger.info("%s: %d admitted scenes", arm, len(scenes))
        r = {"arm": arm,
             "yield_pct": round(100 * rep["scene_admission"]["yield"], 1),
             "n_executed": rep["scene_admission"]["n_scenes"],
             "n_rejected": rep["scene_admission"]["n_rejected"]}
        r.update(summarise(scenes))
        rows.append(r)

    json.dump(rows, open(args.out, "w"), indent=2)
    hdr = (f"{'arm':<13}{'yield':>8}{'TTC<3s':>9}{'traj div':>10}{'style acc':>11}"
           f"{'W1 sp urb':>11}{'W1 sp hwy':>11}{'W1 v urb':>10}{'W1 v hwy':>10}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<13}{r['yield_pct']:>7.1f}%{r['ttc_lt3s_pct']:>8.2f}%"
              f"{r['trajectory_diversity']:>10.2f}{r['style_recovery_acc']:>11.2f}"
              f"{r.get('urban_w1_spacing', float('nan')):>11.2f}"
              f"{r.get('highway_w1_spacing', float('nan')):>11.2f}"
              f"{r.get('urban_w1_speed', float('nan')):>10.2f}"
              f"{r.get('highway_w1_speed', float('nan')):>10.2f}")
    print(f"\nstyle-recovery chance = {1/3:.2f}: can the driving style be recovered "
          f"from EXECUTED trajectories?")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
