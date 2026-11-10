"""
make_paper_figures.py
======================
Render the two figures that carry claims currently made only by tables.

fig_distributions.pdf
    Per-regime spacing and speed distributions of the released data against the
    reference means the fidelity criteria are scored on. The Wasserstein column
    of the fidelity table states that the criteria are met; this shows the
    shapes behind those numbers, including the motorway spacing tail that the
    single summary statistic hides.

fig_ablation.pdf
    Validation loss and the three probe metrics against training duration. The
    reported checkpoint is selected at the validation-loss minimum, and the
    figure shows both that the minimum exists and that the probe metrics do not
    all agree with it -- which is stated in the text and should be visible.

Distributions are computed on admitted, injection-free frames, the same
population the fidelity criteria are scored on, so the figure and the table
cannot disagree.

    python make_paper_figures.py --dataset <root> --results <dir> --out paper/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("figs")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] figs: %(message)s")

# Reference means from the held-out evaluation splits used in Table IV:
# urban = Waymo test split (mean spacing 18.14 m, ego speed 5.866 m/s);
# highway = HighD validation half (mean spacing 29.51 m, ego speed 20.88 m/s).
REF = {"urban":   {"spacing": 18.14, "ego_speed": 5.866, "name": "Waymo"},
       "highway": {"spacing": 29.51, "ego_speed": 20.88, "name": "HighD"}}


def collect(dataset: Path, splits=("train", "val", "test")):
    """Spacing and ego speed per regime, on admitted injection-free frames."""
    out = {r: {"spacing": [], "ego_speed": []} for r in REF}
    for split in splits:
        with open(dataset / "infos" / f"cf_infos_{split}.json") as f:
            scenes = json.load(f)["data"]
        for sc in scenes:
            reg = sc["scene_info"].get("regime")
            if reg not in out:
                continue
            for fr in sc["scene_data"]:
                k = fr["kinematics"]
                if k.get("injection_active"):
                    continue
                out[reg]["spacing"].append(k["spacing_m"])
                out[reg]["ego_speed"].append(k["ego_speed_mps"])
        logger.info("  %s done", split)
    return {r: {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}
            for r, d in out.items()}


def fig_distributions(data, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    panels = [("urban", "spacing", "Spacing (m)", (0, 35)),
              ("urban", "ego_speed", "Ego speed (m/s)", (0, 12)),
              ("highway", "spacing", "Spacing (m)", (0, 100)),
              ("highway", "ego_speed", "Ego speed (m/s)", (0, 35))]
    for ax, (reg, var, xlabel, xlim) in zip(axes.ravel(), panels):
        v = data[reg][var]
        ax.hist(v, bins=70, range=xlim, density=True, color="#4878a8",
                edgecolor="none", alpha=0.85)
        ax.axvline(REF[reg][var], color="#c44e52", ls="--", lw=1.4,
                   label=f"{REF[reg]['name']} mean")
        ax.axvline(float(v.mean()), color="#1a1a1a", ls="-", lw=1.2,
                   label="released mean")
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{reg.capitalize()} ({REF[reg]['name']})", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_distributions.pdf", bbox_inches="tight")
    logger.info("wrote %s", out / "fig_distributions.pdf")
    for reg in REF:
        for var in ("spacing", "ego_speed"):
            v = data[reg][var]
            logger.info("  %s %s: n=%d mean=%.2f std=%.2f",
                        reg, var, len(v), v.mean(), v.std())


def fig_ablation(results: Path, out: Path) -> None:
    tl = json.load(open(results / "train_log.json"))
    vl = {h["epoch"]: h["val_loss"] for h in tl["history"] if "checkpoint" in h}
    eps, files = [0.0, 0.25, 0.5, 1.0, 2.0], \
        ["before", "after_epoch0p25", "after_epoch0p5", "after_epoch1",
         "after_epoch2"]
    J = [json.load(open(results / f"{f}.json")) for f in files]
    mae = [d["spacing"]["mae"] for d in J]
    bal = [100 * d["safety"]["balanced_accuracy"] for d in J]
    sce = [100 * d["scenario"]["accuracy"] for d in J]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax = axes[0]
    ke = sorted(vl)
    ax.plot(ke, [vl[e] for e in ke], "o-", color="#4878a8")
    best = min(vl, key=vl.get)
    ax.plot([best], [vl[best]], "o", ms=9, mfc="none", mec="#c44e52", mew=1.6)
    ax.set_ylabel("validation loss", fontsize=8)
    ax.set_title("Validation loss", fontsize=8.5)

    ax = axes[1]
    ax.plot(eps, mae, "o-", color="#4878a8", label="LoRA")
    ax.axhline(10.47, color="#c44e52", ls="--", lw=1.2, label="median baseline")
    ax.axhline(1.18, color="#55a868", ls=":", lw=1.2, label="ResNet-18")
    ax.set_yscale("log")
    ax.set_ylabel("spacing MAE (m)", fontsize=8)
    ax.set_title("Spacing", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    ax.plot(eps, bal, "o-", color="#4878a8", label="safety bal. acc.")
    ax.plot(eps, sce, "s-", color="#8172b2", label="scenario acc.")
    ax.axhline(50, color="#c44e52", ls="--", lw=1.0)
    ax.axhline(11.1, color="#8172b2", ls=":", lw=1.0)
    ax.set_ylabel("accuracy (%)", fontsize=8)
    ax.set_title("Classification", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)

    for ax in axes:
        ax.set_xlabel("training epochs", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation.pdf", bbox_inches="tight")
    logger.info("wrote %s", out / "fig_ablation.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-distributions", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   metavar="SPLIT",
                   help="which infos splits to include (default: all three)")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.skip_distributions:
        fig_distributions(collect(Path(args.dataset), splits=args.splits), out)
    if not args.skip_ablation:
        fig_ablation(Path(args.results), out)


if __name__ == "__main__":
    main()
