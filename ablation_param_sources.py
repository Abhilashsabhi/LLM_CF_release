"""
ablation_param_sources.py
==========================
Ablate the parameter source: does the LLM contribute anything a cheap sampler
could not?

The pipeline's title claim rests on Stage 1, yet Stage 3 subsequently overwrites
the ego desired speed, the ego time headway and the initial conditions. It is
therefore not self-evident that the LLM contributes anything beyond what uniform
sampling, Latin-hypercube sampling, or hand-written style priors would give. This
script builds four sets of base parameters that are interchangeable inputs to the
*identical* Stage 3 calibration and Stage 2 execution, so the parameter source is
the only thing that varies.

Arms
----
  llm          the released base parameters (135 LLM calls -> 27 base sets)
  uniform      each field drawn uniformly from the admissible bounds
  lhs          Latin-hypercube over the same bounds, stratified across combos
  handcrafted  the pipeline's own hand-written cautious/normal/aggressive
               priors, with the same per-category edge-case tweaks

Every arm produces exactly 27 base sets (3 styles x 9 categories) with the same
schema, so `main.py --params <arm>.yaml` runs each through the unchanged
downstream pipeline.

Parameter-level metrics computed here need no simulator:

  diversity        mean pairwise Euclidean distance in bound-normalised
                   parameter space -- how much of the feasible set is covered
  style separation silhouette score of the three style clusters, and the
                   leave-one-out accuracy of a nearest-centroid style
                   classifier: does the source encode driving style at all?
  clamp rate       fraction of proposed values that had to be projected back
                   onto the admissible set (LLM only; the samplers are
                   in-bounds by construction, which is itself worth stating)

    python ablation_param_sources.py --released <configs dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

logger = logging.getLogger("ablate")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] ablate: %(message)s")

STYLES = ["cautious", "normal", "aggressive"]
CATEGORIES = ["steady_follow", "stop_and_go", "aggressive_leader", "cut_in",
              "cut_out", "speed_limit_change", "grade_change",
              "adverse_weather", "pedestrian_crossing"]

# The fields that actually reach the simulator, grouped as Stage 1 stores them.
GROUPS = {
    "idm": ["desired_speed_mps", "time_headway_s", "max_acceleration_mps2",
                "comfortable_decel_mps2", "jam_spacing_m", "reaction_time_s",
                "delta"],
    "leader_idm": ["desired_speed_mps", "time_headway_s", "max_acceleration_mps2",
                   "comfortable_decel_mps2", "jam_spacing_m", "reaction_time_s",
                   "delta"],
    "mobil": ["politeness_factor", "accel_threshold_mps2", "safe_decel_mps2"],
    "scene": ["initial_spacing_m", "initial_ego_speed_mps",
              "initial_leader_speed_mps"],
    "edge": ["edge_trigger_time_s", "edge_decel_mps2", "edge_cut_in_ttc_s",
             "ped_cross_distance_m"],
}


def bounds_for(group: str) -> Dict[str, tuple]:
    import stage1_llm_generator as S
    return {"idm": S.IDM_BOUNDS, "leader_idm": S.IDM_BOUNDS,
            "mobil": S.MOBIL_BOUNDS, "scene": S.SCENE_BOUNDS,
            "edge": S.EDGE_BOUNDS}[group]


def flat_vector(entry: Dict) -> np.ndarray:
    """Bound-normalised vector, so every field contributes comparably."""
    v = []
    for g, fields in GROUPS.items():
        b = bounds_for(g)
        for f in fields:
            lo, hi = b[f]
            x = float(entry[g][f])
            v.append((x - lo) / max(hi - lo, 1e-9))
    return np.asarray(v, dtype=float)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def _finish(e: Dict) -> None:
    """Add the non-behavioural scene fields the released schema carries."""
    e["scene"]["scene_duration_s"] = 30.0
    e["scene"]["sampling_interval_s"] = 0.1


def arm_llm(released: Path) -> Dict:
    return yaml.safe_load(open(released / "llm_base_params.yaml"))


def _blank(style: str, cat: str) -> Dict:
    """Schema-identical to the released base file, so the same loader reads it."""
    return {"scenario_id": f"{style}_{cat}", "driving_style": style,
            "scenario_category": cat, "road_grade_deg": 0.0,
            "llm_reasoning": "", "consistency_votes": 0}


def arm_uniform(seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    out: Dict = {}
    for s in STYLES:
        out[s] = {}
        for c in CATEGORIES:
            e = _blank(s, c)
            for g, fields in GROUPS.items():
                b = bounds_for(g)
                e[g] = {f: float(rng.uniform(*b[f])) for f in fields}
            _finish(e)
            out[s][c] = e
    return out


def arm_lhs(seed: int) -> Dict:
    """Latin hypercube over the 27 combinations.

    Each field's range is cut into 27 equal strata and every stratum is used
    exactly once across the 27 combinations, so the arm covers the feasible
    set far more evenly than independent uniform draws -- which is precisely
    the comparison that matters, since even coverage is the usual reason to
    prefer a designed sample.
    """
    rng = np.random.default_rng(seed)
    n = len(STYLES) * len(CATEGORIES)
    combos = [(s, c) for s in STYLES for c in CATEGORIES]
    out: Dict = {s: {} for s in STYLES}
    plan: Dict[str, np.ndarray] = {}
    for g, fields in GROUPS.items():
        for f in fields:
            u = (rng.permutation(n) + rng.random(n)) / n     # stratified
            plan[f"{g}.{f}"] = u
    for i, (s, c) in enumerate(combos):
        e = _blank(s, c)
        for g, fields in GROUPS.items():
            b = bounds_for(g)
            e[g] = {}
            for f in fields:
                lo, hi = b[f]
                e[g][f] = float(lo + plan[f"{g}.{f}"][i] * (hi - lo))
        _finish(e)
        out[s][c] = e
    return out


def arm_handcrafted() -> Dict:
    """The pipeline's own hand-written style priors.

    This is the honest 'no LLM' baseline: it already exists in the codebase as
    the offline fallback, so it costs nothing and encodes exactly the domain
    knowledge an author would write down unaided.
    """
    import stage1_llm_generator as S
    fb = S._fallback_base()
    out: Dict = {}
    for s in STYLES:
        out[s] = {}
        base = fb[s]
        for c in CATEGORIES:
            e = _blank(s, c)
            for g, src in (("idm", base["ego_idm"]),
                           ("leader_idm", base["leader_idm"]),
                           ("mobil", base["mobil"]), ("scene", base["scene"]),
                           ("edge", base["edge"])):
                e[g] = {f: float(getattr(src, f)) for f in GROUPS[g]}
            # Same per-category tweaks the fallback path applies.
            if c == "pedestrian_crossing":
                e["edge"].update(edge_trigger_time_s=8.0, edge_decel_mps2=6.0,
                                 ped_cross_distance_m=25.0)
            elif c == "aggressive_leader":
                e["edge"]["edge_decel_mps2"] = 8.0
            _finish(e)
            out[s][c] = e
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(arm: Dict, name: str) -> Dict:
    X, y = [], []
    for si, s in enumerate(STYLES):
        for c in CATEGORIES:
            X.append(flat_vector(arm[s][c])); y.append(si)
    X = np.vstack(X); y = np.asarray(y)

    d = [float(np.linalg.norm(X[i] - X[j])) for i, j in combinations(range(len(X)), 2)]
    diversity = float(np.mean(d))

    # Silhouette of the three style clusters.
    sil = []
    for i in range(len(X)):
        same = [np.linalg.norm(X[i] - X[j]) for j in range(len(X))
                if j != i and y[j] == y[i]]
        a = float(np.mean(same)) if same else 0.0
        bvals = []
        for k in range(3):
            if k == y[i]:
                continue
            other = [np.linalg.norm(X[i] - X[j]) for j in range(len(X)) if y[j] == k]
            if other:
                bvals.append(float(np.mean(other)))
        b = min(bvals) if bvals else 0.0
        sil.append((b - a) / max(a, b, 1e-9))
    silhouette = float(np.mean(sil))

    # Leave-one-out nearest-centroid style classification.
    correct = 0
    for i in range(len(X)):
        cents = []
        for k in range(3):
            idx = [j for j in range(len(X)) if y[j] == k and j != i]
            cents.append(X[idx].mean(axis=0))
        correct += int(np.argmin([np.linalg.norm(X[i] - c) for c in cents]) == y[i])
    style_acc = correct / len(X)

    # Category separability. Style separation alone is easy to fake: three
    # constant points score perfectly on it while encoding nothing about the
    # scenario. This asks the complementary question -- can the scenario be
    # recovered from the parameters? -- where a constant-per-style source must
    # sit at chance.
    yc = np.array([ci for _ in STYLES for ci in range(len(CATEGORIES))])
    corr_c = 0
    for i in range(len(X)):
        cents, labs = [], []
        for k in range(len(CATEGORIES)):
            idx = [j for j in range(len(X)) if yc[j] == k and j != i]
            if idx:
                cents.append(X[idx].mean(axis=0)); labs.append(k)
        corr_c += int(labs[int(np.argmin([np.linalg.norm(X[i] - c) for c in cents]))] == yc[i])
    cat_acc = corr_c / len(X)

    # The same question restricted to the blocks where scenario semantics can
    # live. Scenario categories are realised mainly by scripted injections and
    # by the leader, not by the ego's own IDM constants, so a classifier over
    # the full vector is swamped by the near-constant ego block and understates
    # category sensitivity.
    sub_dims = []
    off = 0
    for g, fields in GROUPS.items():
        for _ in fields:
            if g in ("leader_idm", "scene", "edge"):
                sub_dims.append(off)
            off += 1
    Xs = X[:, sub_dims]
    corr_s = 0
    for i in range(len(Xs)):
        cents, labs = [], []
        for k in range(len(CATEGORIES)):
            idx = [j for j in range(len(Xs)) if yc[j] == k and j != i]
            if idx:
                cents.append(Xs[idx].mean(axis=0)); labs.append(k)
        corr_s += int(labs[int(np.argmin([np.linalg.norm(Xs[i] - c) for c in cents]))] == yc[i])
    cat_acc_sub = corr_s / len(Xs)

    # How much the source varies across categories within a style at all.
    spread = []
    for st in STYLES:
        M = np.vstack([flat_vector(arm[st][c]) for c in CATEGORIES])
        spread.append(float(M.std(axis=0).mean()))
    cat_spread = float(np.mean(spread))

    # Physical admissibility of the raw proposal, judged by the pipeline's own
    # validator. Uniform and LHS are inside the per-field boxes by
    # construction, so any failure here is a JOINT infeasibility -- a
    # combination of individually legal values that is not a coherent driver.
    import stage1_llm_generator as S
    val = S.ParameterValidator()
    n_ok = 0
    for st in STYLES:
        for c in CATEGORIES:
            e = arm[st][c]
            p_ = S.ScenarioParams(
                driving_style=st, scenario_category=c,
                idm=S.IDMParams(**{k: e["idm"][k] for k in GROUPS["idm"]}),
                leader_idm=S.IDMParams(**{k: e["leader_idm"][k] for k in GROUPS["leader_idm"]}),
                mobil=S.MOBILParams(**{k: e["mobil"][k] for k in GROUPS["mobil"]}),
                scene=S.SceneConfig(**{k: e["scene"][k] for k in GROUPS["scene"]}),
                edge=S.EdgeConfig(**{k: e["edge"][k] for k in GROUPS["edge"]}))
            ok, _ = val.validate(p_)
            n_ok += int(ok)
    admissible = n_ok / (len(STYLES) * len(CATEGORIES))

    # Spread of the two fields Stage 3 does NOT overwrite and that most shape
    # the follower's response.
    accel = np.array([arm[s][c]["idm"]["max_acceleration_mps2"]
                      for s in STYLES for c in CATEGORIES])
    decel = np.array([arm[s][c]["idm"]["comfortable_decel_mps2"]
                      for s in STYLES for c in CATEGORIES])
    return {
        "arm": name, "n_base_sets": int(len(X)), "n_dims": int(X.shape[1]),
        "diversity_mean_pairwise": round(diversity, 4),
        "style_silhouette": round(silhouette, 4),
        "style_loo_accuracy": round(style_acc, 4),
        "style_chance": round(1 / 3, 4),
        "category_loo_accuracy": round(cat_acc, 4),
        "category_chance": round(1 / len(CATEGORIES), 4),
        "category_loo_accuracy_scenario_blocks": round(cat_acc_sub, 4),
        "category_spread_within_style": round(cat_spread, 4),
        "admissible_fraction": round(admissible, 4),
        "accel_mean": round(float(accel.mean()), 3),
        "accel_std": round(float(accel.std()), 3),
        "decel_mean": round(float(decel.mean()), 3),
        "decel_std": round(float(decel.std()), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--released", required=True, help="configs dir of the release")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = {
        "llm": arm_llm(Path(args.released)),
        "uniform": arm_uniform(args.seed),
        "lhs": arm_lhs(args.seed),
        "handcrafted": arm_handcrafted(),
    }
    rows = []
    for name, arm in arms.items():
        with open(out / f"base_{name}.yaml", "w") as f:
            yaml.dump(arm, f, default_flow_style=False, sort_keys=False)
        rows.append(metrics(arm, name))
        logger.info("wrote base_%s.yaml", name)

    json.dump(rows, open(out / "param_metrics.json", "w"), indent=2)
    hdr = (f"{'arm':<13}{'diversity':>11}{'silhouette':>12}{'style acc':>11}"
           f"{'cat acc':>9}{'cat acc*':>10}{'cat spread':>12}{'admiss':>9}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<13}{r['diversity_mean_pairwise']:>11.3f}"
              f"{r['style_silhouette']:>12.3f}{r['style_loo_accuracy']:>11.2f}"
              f"{r['category_loo_accuracy']:>9.2f}"
              f"{r['category_loo_accuracy_scenario_blocks']:>10.2f}"
              f"{r['category_spread_within_style']:>12.4f}"
              f"{r['admissible_fraction']:>9.2f}")
    print(f"\n'cat acc*' restricts the classifier to the leader/scene/edge "
          f"blocks, where scenario semantics live.")
    print(f"chance: style {1/3:.2f}, category {1/9:.2f}. "
          f"'admissible' = fraction passing the pipeline's own joint validator.")
    print(f"wrote {out}/param_metrics.json")


if __name__ == "__main__":
    main()
