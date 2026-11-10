"""
make_adapt_table.py
====================
Emit the LaTeX table for the adaptation study on the released question--answer
pairs, and the epoch ablation alongside it.

Numbers are generated from the result JSONs rather than transcribed. Two tables
in an earlier draft acquired swapped values through manual copying, which is the
class of error a reader cannot detect and an author cannot easily find.

    python make_adapt_table.py > paper/tab_adapt_v2.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

BASE = Path("/var/tmp/adapt_v3")
CONDS = [("before", "before.json", "Unadapted (bfloat16)"),
         ("ep0p25", "after_epoch0p25.json", "+ LoRA, 0.25 epoch"),
         ("ep0p5", "after_epoch0p5.json", "+ LoRA, 0.5 epoch"),
         ("ep1", "after_epoch1.json", "+ LoRA, 1 epoch")]

ORDER = [
    ("spacing_estimation", "Spacing (MAE, m)$\\downarrow$", "frame"),
    ("lead_presence",      "Lead presence", "frame"),
    ("visual_conditions",  "Conditions", "frame"),
    ("gap_trend",          "Gap trend", "clip"),
    ("relative_motion",    "Relative motion", "clip"),
    ("safety_ttc",         "Safety (TTC)", "clip"),
    ("ego_action",         "Ego action", "clip"),
    ("scenario_temporal",  "Scenario (9-way)", "clip"),
]


def load() -> Dict[str, Dict[str, Dict]]:
    out = {}
    for key, fn, _ in CONDS:
        p = BASE / fn
        if not p.exists():
            continue
        out[key] = {e["qa_type"]: e for e in json.load(open(p))["types"]}
    return out


def val(e: Dict) -> str:
    if e is None or not e.get("n"):
        return "---"
    if e.get("kind") == "numeric":
        return f"\\num{{{e['mae_m']:.2f}}}"
    return f"\\num{{{e['balanced_accuracy']:.3f}}}"


def chance_of(data: Dict, qa_type: str) -> str:
    for key in ("before", "ep1"):
        e = data.get(key, {}).get(qa_type)
        if not e or not e.get("n"):
            continue
        if e.get("kind") == "numeric":
            b = e.get("baseline_median_mae_m")
            return f"\\num{{{b:.2f}}}" if b is not None else "---"
        return f"\\num{{{e['chance']:.3f}}}"
    return "---"


def main() -> None:
    data = load()
    if "before" not in data:
        sys.exit("no before.json found")
    have = [c for c in CONDS if c[0] in data]

    print(r"\begin{table*}[!t]")
    print(r"\caption{Parameter-efficient adaptation on the \emph{released} "
          r"question--answer pairs. Training, evaluation and the released "
          r"supervision are the same pairs; prompts are never re-worded. "
          r"Balanced accuracy (mean per-class recall) for the categorical "
          r"types and mean absolute error for spacing, on the held-out test "
          r"split, \num{100} class-balanced items per type. The unadapted row "
          r"is the same bfloat16 stack as the adapted rows, so the comparison "
          r"isolates the adapter rather than a change in numerical precision. "
          r"Only the language tower is adapted; the vision encoder is frozen.}")
    print(r"\label{tab:adapt_v2}")
    print(r"\centering")
    cols = "l l " + "r" * len(have) + " r"
    print(rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    print(r"\toprule")
    heads = " & ".join(h.replace("+ LoRA, ", "") for _, _, h in have)
    print(rf"Question & Level & {heads} & Chance/base \\")
    print(r"\midrule")
    for qa_type, label, level in ORDER:
        cells = " & ".join(val(data[k].get(qa_type)) for k, _, _ in have)
        print(rf"{label} & {level} & {cells} & {chance_of(data, qa_type)} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

    # Validation losses, for the training-duration discussion.
    tl = BASE / "train_log.json"
    if tl.exists():
        h = json.load(open(tl))["history"]
        print()
        print("% validation loss by checkpoint:")
        for r in h:
            print(f"%   epoch {r['epoch']:<6} step {r['step']:<6} "
                  f"val_loss {r['val_loss']:.4f}")

    # Collapse diagnostics.
    print()
    print("% modal prediction share, per condition and type:")
    for k, _, label in have:
        for qa_type, _, _ in ORDER:
            e = data[k].get(qa_type)
            if e and e.get("kind") == "categorical" and e.get("n"):
                print(f"%   {label:<22}{qa_type:<22}{e['modal_pred']:<18}"
                      f"{e['modal_pred_share']:.2f}")


if __name__ == "__main__":
    main()
