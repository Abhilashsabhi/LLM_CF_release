"""
make_adapt_figure.py
=====================
Regenerate the training-duration figure from the released-pair adaptation run.

The previous figure plotted the superseded three-probe study and would have been
silently wrong beside the new table. It is redrawn here from the same JSONs that
produce Table~\\ref{tab:adapt_v2}, so figure and table cannot disagree.

    python make_adapt_figure.py --out paper/fig_adapt_v2.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path("/var/tmp/adapt_v3")
CONDS = [(0.0, "before.json"), (0.25, "after_epoch0p25.json"),
         (0.5, "after_epoch0p5.json"), (1.0, "after_epoch1.json")]

CLIP = ["gap_trend", "relative_motion", "safety_ttc", "ego_action",
        "scenario_temporal"]
FRAME = ["lead_presence", "visual_conditions"]
NICE = {"gap_trend": "gap trend", "relative_motion": "relative motion",
        "safety_ttc": "safety (TTC)", "ego_action": "ego action",
        "scenario_temporal": "scenario (9-way)",
        "lead_presence": "lead presence", "visual_conditions": "conditions"}

# chance level 1/k per task type
CHANCE = {"gap_trend": 1/3, "relative_motion": 1/3, "safety_ttc": 1/2,
          "ego_action": 1/3, "scenario_temporal": 1/9,
          "lead_presence": 1/2, "visual_conditions": 1/4}


def skill(ba: float, t: str) -> float:
    """Chance-normalised skill: 0 at chance, 1 at perfect, negative below."""
    c = CHANCE[t]
    return (ba - c) / (1.0 - c)


def load():
    out = {}
    for ep, fn in CONDS:
        p = BASE / fn
        if p.exists():
            out[ep] = {e["qa_type"]: e for e in json.load(open(p))["types"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load()
    eps = sorted(data)
    hist = json.load(open(BASE / "train_log.json"))["history"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # --- validation loss -------------------------------------------------
    ax = axes[0]
    ax.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist],
            "o-", color="#c0392b")
    ax.set_xlabel("training epochs"); ax.set_ylabel("validation loss")
    ax.set_title("Validation loss (held-out scenes)")
    ax.grid(alpha=0.3)
    # The curve is still descending at the point we stopped; say so on the
    # figure rather than let a reader infer a minimum that is not there.
    ax.annotate("still falling at 1 epoch", xy=(1.0, hist[-1]["val_loss"]),
                xytext=(0.45, hist[0]["val_loss"] * 0.98), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    # --- spacing ---------------------------------------------------------
    ax = axes[1]
    mae = [data[e]["spacing_estimation"]["mae_m"] for e in eps]
    base = data[eps[0]]["spacing_estimation"].get("baseline_median_mae_m")
    ax.plot(eps, mae, "o-", color="#2c3e50", label="adapted model")
    if base:
        ax.axhline(base, ls="--", color="#7f8c8d",
                   label=f"median-gap baseline ({base:.2f} m)")
    ax.axhline(0.61, ls=":", color="#27ae60",
               label="supervised ResNet-18 (0.61 m, same items)")
    ax.set_yscale("log")
    ax.set_xlabel("training epochs"); ax.set_ylabel("spacing MAE (m), log")
    ax.set_title("Metric spacing")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # --- categorical (chance-normalised skill) ---------------------------
    ax = axes[2]
    for t in CLIP + FRAME:
        ys = [skill(data[e][t]["balanced_accuracy"], t) for e in eps]
        ax.plot(eps, ys, marker="o", lw=1.2, ms=3,
                ls="-" if t in CLIP else "--", label=NICE[t])
    ax.axhline(0.0, ls=":", color="#7f8c8d", lw=0.8, label="chance")
    ax.set_xlabel("training epochs")
    ax.set_ylabel(r"skill $\frac{\mathrm{BA}-1/k}{1-1/k}$")
    ax.set_title("Categorical questions (solid: clip, dashed: frame)")
    ax.set_ylim(-0.25, 1.05)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
