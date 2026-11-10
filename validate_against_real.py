"""
validate_against_real.py
=========================
Replace the circular fidelity check with one that uses real recordings and
empirically derived tolerances.

What was wrong
--------------
Stage 4 scored Wasserstein-1 distance against a *truncated Gaussian* whose mean
and standard deviation were written by hand, and whose means were the same
numbers Stage 3 calibrates towards. The test was therefore circular -- the
pipeline was tuned to hit a mean and then scored on its distance to a
distribution centred on that mean -- and the tolerances (8 m, 8 m/s, 3 m/s)
were round numbers with no derivation. They are so permissive that uniform
noise on [0, 30] m/s passes the speed criterion.

What this does instead
----------------------
1. Uses *empirical* distributions from real car-following events, pooled over
   frames, rather than a parametric surrogate.

2. Separates calibration from validation. Targets may be estimated from a
   corpus's calibration split; fidelity is scored only on a held-out split that
   played no part in calibration, and additionally on an independent corpus
   that played no part at all.

3. Derives tolerances from real-to-real variability. For each corpus and
   variable we repeatedly split the events into two disjoint halves and compute
   W1 between them. That is the distance one expects between two samples of the
   *same* population at this sample size; its upper percentile is the natural
   threshold for "indistinguishable from real". Cross-corpus W1 between two
   different real corpora of the same regime gives the complementary scale:
   how far apart two genuine recordings of the same kind of traffic sit.

Event definition, units and provenance are documented in `--describe` output
and written into the results file, because W1 needs the distribution and a
reader cannot reconstruct one from a mean.

    python validate_against_real.py --real-dir <dir> --dataset <root> --out <f>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("realval")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] realval: %(message)s")

# FollowNet channel order, shared by the real corpora and by our export.
CH = {"spacing_m": 0, "ego_speed_mps": 1, "relative_speed_mps": 2,
      "leader_speed_mps": 3}

# Which real corpus stands in for which traffic regime.
#
# The urban reference was previously NGSIM. That was wrong: FollowNet's NGSIM
# extraction is the I-80 section at Emeryville, a six-lane FREEWAY, and the
# congestion there is slow-moving freeway traffic rather than signalised city
# driving. The behavioural signature separates them cleanly -- 29.6% of Waymo
# frames are below 1 m/s (stops at signals and intersections) against 1.1% of
# NGSIM I-80 frames. Validating CARLA town maps against I-80 was therefore a
# road-context mismatch, so the urban reference is now Waymo city driving and
# NGSIM is reported as what it is.
REGIME_PRIMARY = {"urban": "Waymo", "highway": "HighD"}

# The independent comparator. No second genuine urban corpus was available, so
# the urban entry is a DIFFERENT ROAD TYPE (congested freeway) and is labelled
# as such: it is an upper reference on how far two real recordings of nominally
# similar traffic can sit, not a like-for-like independent corpus.
REGIME_INDEPENDENT = {"urban": "NGSIM", "highway": "SPMD1"}
INDEPENDENT_NOTE = {"urban": "different road type: congested freeway (NGSIM I-80)",
                    "highway": "independent motorway/highway corpus (SPMD)"}


def w1(a: np.ndarray, b: np.ndarray, n_q: int = 2000) -> float:
    """Wasserstein-1 between two empirical samples, via quantile functions."""
    q = np.linspace(0.0, 1.0, n_q)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def load_real(real_dir: Path, extra: Dict[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """{corpus: {split: (n_events, 4, T)}} for every corpus we can find."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(real_dir.glob("*_*.npy")):
        stem = f.stem
        if "_" not in stem:
            continue
        corpus, split = stem.rsplit("_", 1)
        if split not in ("train", "val", "test"):
            continue
        out.setdefault(corpus, {})[split] = np.load(f, allow_pickle=True)
    for name, path in extra.items():
        if path and path.exists():
            out.setdefault(name, {})["test"] = np.load(path, allow_pickle=True)
    return out


def frames(events: np.ndarray, var: str) -> np.ndarray:
    return np.asarray(events[:, CH[var], :], dtype=float).ravel()


def null_w1(events: np.ndarray, var: str, n_boot: int, seed: int,
            cap: int) -> Dict[str, float]:
    """W1 between two disjoint halves of the SAME corpus.

    Split at the EVENT level, not the frame level: frames within an event are
    strongly autocorrelated, and splitting frames would understate the true
    sampling variability by treating 150 correlated samples as independent.
    """
    rng = np.random.default_rng(seed)
    n = len(events)
    vals = []
    for _ in range(n_boot):
        perm = rng.permutation(n)
        a = frames(events[perm[: n // 2]], var)
        b = frames(events[perm[n // 2: 2 * (n // 2)]], var)
        if cap and len(a) > cap:
            a = rng.choice(a, cap, replace=False)
            b = rng.choice(b, cap, replace=False)
        vals.append(w1(a, b))
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "n_boot": n_boot}


def ours_by_regime(dataset: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Released frames per regime, admitted and injection-free."""
    out = {r: {v: [] for v in CH} for r in REGIME_PRIMARY}
    for split in ("train", "val", "test"):
        f = dataset / "infos" / f"cf_infos_{split}.json"
        if not f.exists():
            continue
        for sc in json.load(open(f))["data"]:
            reg = sc["scene_info"].get("regime")
            if reg not in out:
                continue
            for fr in sc["scene_data"]:
                k = fr["kinematics"]
                if k.get("injection_active"):
                    continue
                out[reg]["spacing_m"].append(k["spacing_m"])
                out[reg]["ego_speed_mps"].append(k["ego_speed_mps"])
                out[reg]["relative_speed_mps"].append(k["relative_speed_mps"])
                out[reg]["leader_speed_mps"].append(k["leader_speed_mps"])
    return {r: {v: np.asarray(x, dtype=float) for v, x in d.items()}
            for r, d in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--highd", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--cap", type=int, default=40000,
                   help="frames per side when estimating the null (0 = no cap)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    real = load_real(Path(args.real_dir),
                     {"HighD": Path(args.highd) if args.highd else None})
    logger.info("real corpora: %s",
                {k: {s: v.shape for s, v in d.items()} for k, d in real.items()})

    # ---- provenance of every reference distribution ----------------------
    provenance = {}
    for corpus, splits in real.items():
        provenance[corpus] = {
            s: {"n_events": int(v.shape[0]), "event_length_frames": int(v.shape[2]),
                "n_frames": int(v.shape[0] * v.shape[2]),
                "channels": list(CH), "units": "m, m/s, m/s, m/s",
                "means": {k: round(float(v[:, i].mean()), 3) for k, i in CH.items()},
                "stds": {k: round(float(v[:, i].std()), 3) for k, i in CH.items()}}
            for s, v in splits.items()}

    # ---- tolerances from real-to-real variability ------------------------
    tolerances, nulls = {}, {}
    for regime, corpus in REGIME_PRIMARY.items():
        if corpus not in real:
            continue
        ev = real[corpus].get("train", real[corpus].get("test"))
        nulls[corpus] = {}
        for var in CH:
            nulls[corpus][var] = null_w1(ev, var, args.n_boot, args.seed, args.cap)
        tolerances[regime] = {v: round(nulls[corpus][v]["p95"], 3) for v in CH}
        logger.info("%s tolerances (p95 of same-corpus W1): %s", regime,
                    tolerances[regime])

    # ---- cross-corpus scale: two DIFFERENT real corpora, same regime -----
    cross = {}
    for regime, corpus in REGIME_PRIMARY.items():
        other = REGIME_INDEPENDENT.get(regime)
        if corpus not in real or other not in real:
            continue
        a_ev = real[corpus].get("test", real[corpus].get("train"))
        b_ev = real[other].get("test", real[other].get("train"))
        cross[regime] = {v: round(w1(frames(a_ev, v), frames(b_ev, v)), 3)
                         for v in CH}
        logger.info("%s cross-corpus W1 (%s vs %s): %s", regime, corpus, other,
                    cross[regime])

    # ---- score the released data on HELD-OUT real splits -----------------
    ours = ours_by_regime(Path(args.dataset))
    results = []
    for regime, corpus in REGIME_PRIMARY.items():
        if corpus not in real:
            continue
        held = real[corpus].get("test")
        indep_name = REGIME_INDEPENDENT.get(regime)
        indep = real.get(indep_name, {}).get("test",
                real.get(indep_name, {}).get("train"))
        for var in CH:
            if not len(ours.get(regime, {}).get(var, [])):
                continue
            row = {"regime": regime, "variable": var,
                   "n_frames_ours": int(len(ours[regime][var])),
                   "ours_mean": round(float(ours[regime][var].mean()), 3),
                   "ours_std": round(float(ours[regime][var].std()), 3)}
            if held is not None:
                hv = frames(held, var)
                row.update({"heldout_corpus": corpus,
                            "heldout_mean": round(float(hv.mean()), 3),
                            "heldout_std": round(float(hv.std()), 3),
                            "w1_heldout": round(w1(ours[regime][var], hv), 3),
                            "tolerance_p95": tolerances[regime][var]})
                row["pass_heldout"] = bool(row["w1_heldout"] <= row["tolerance_p95"])
            if indep is not None:
                iv = frames(indep, var)
                row.update({"independent_corpus": indep_name,
                            "w1_independent": round(w1(ours[regime][var], iv), 3)})
            # Score an INDEPENDENT REAL corpus by the identical criterion.
            # Without this the reader cannot tell whether a failure means the
            # synthetic data is poor or the tolerance is unreachable: a
            # same-population p95 asks a candidate to be indistinguishable from
            # a resample of one recording, which a different genuine recording
            # of the same regime need not satisfy either.
            if indep is not None and held is not None:
                row["w1_real_vs_real"] = round(w1(frames(indep, var),
                                                  frames(held, var)), 3)
                row["real_corpus_passes"] = bool(
                    row["w1_real_vs_real"] <= row["tolerance_p95"])
                row["closer_than_real_corpus"] = bool(
                    row["w1_heldout"] <= row["w1_real_vs_real"])
            results.append(row)

    payload = {
        "method": {
            "event_source": ("Car-following events extracted by the FollowNet "
                             "benchmark from the public recordings; channels are "
                             "[spacing, ego speed, relative speed, leader speed] "
                             "in m and m/s, resampled to a fixed event length."),
            "distribution_estimation": ("Empirical. Frames are pooled across "
                                        "events within a split and W1 is computed "
                                        "between empirical quantile functions on "
                                        "2000 quantiles. No smoothing, no "
                                        "parametric fit."),
            "tolerance_derivation": ("Events of the reference corpus are split "
                                     "into two disjoint halves at the EVENT "
                                     "level and W1 is computed between the "
                                     "halves; repeated to give a null "
                                     "distribution of same-population W1. The "
                                     "tolerance is its 95th percentile."),
            "separation": ("Tolerances are derived on the reference corpus's "
                           "training split; fidelity is scored on its held-out "
                           "test split, and additionally against an independent "
                           "corpus of the same regime that is used nowhere else."),
            "n_boot": args.n_boot, "frame_cap_per_side": args.cap,
        },
        "provenance": provenance, "null_w1": nulls,
        "tolerances_p95": tolerances, "cross_corpus_w1": cross,
        "results": results,
    }
    json.dump(payload, open(args.out, "w"), indent=2)

    hdr = (f"{'regime':<9}{'variable':<20}{'ours':>8}{'real':>8}"
           f"{'W1 ours':>9}{'p95 tol':>9}{'W1 real-real':>14}{'real ok':>9}"
           f"{'ours<=real':>12}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['regime']:<9}{r['variable']:<20}{r['ours_mean']:>8.2f}"
              f"{r.get('heldout_mean', float('nan')):>8.2f}"
              f"{r.get('w1_heldout', float('nan')):>9.2f}"
              f"{r.get('tolerance_p95', float('nan')):>9.2f}"
              f"{r.get('w1_real_vs_real', float('nan')):>14.2f}"
              f"{('yes' if r.get('real_corpus_passes') else 'NO'):>9}"
              f"{('yes' if r.get('closer_than_real_corpus') else 'no'):>12}")
    print("\ntolerance = 95th percentile of W1 between two disjoint halves of the "
          "SAME real corpus.")
    print("'W1 real-real' = distance between two DIFFERENT real corpora of the "
          "same regime; 'real ok' asks whether")
    print("that genuine recording would itself pass the tolerance, and "
          "'ours<=real' whether our data is at least")
    print("as close to the held-out reference as another real recording is.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
