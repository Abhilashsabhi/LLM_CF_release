"""
make_v2_bench_table.py
=======================
Emit the LaTeX table for the released-pair vision--language benchmark.

Numbers in the paper are generated from the result JSONs rather than copied by
hand. Two tables in an earlier draft acquired swapped values through manual
transcription, which is exactly the class of error a reader cannot detect and
an author cannot easily find.

    python make_v2_bench_table.py /var/tmp/v2bench_*.json > paper/tab_vlm_v2.tex
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List

# Display order and short column labels, chosen so the frame-level questions
# precede the clip-level ones as they do in the release.
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

TEX_MODEL = {
    "qwen2.5vl:7b": r"\texttt{qwen2.5vl:7b}",
    "llava:7b": r"\texttt{llava:7b}",
    "minicpm-v": r"\texttt{minicpm-v}",
}


def load(paths: List[str]) -> List[Dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def cell(res: Dict, qa_type: str) -> str:
    e = next((t for t in res["types"] if t["qa_type"] == qa_type), None)
    if e is None or not e.get("n"):
        return "---"
    if e.get("kind") == "numeric":
        return f"\\num{{{e['mae_m']:.2f}}}"
    return f"\\num{{{e['balanced_accuracy']:.3f}}}"


def baseline_cell(runs: List[Dict], qa_type: str) -> str:
    for r in runs:
        e = next((t for t in r["types"] if t["qa_type"] == qa_type), None)
        if e is None or not e.get("n"):
            continue
        if e.get("kind") == "numeric":
            b = e.get("baseline_median_mae_m")
            return f"\\num{{{b:.2f}}}" if b is not None else "---"
        return f"\\num{{{e['chance']:.3f}}}"
    return "---"


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: make_v2_bench_table.py <result json> ...")
    runs = load(paths)
    runs.sort(key=lambda r: list(TEX_MODEL).index(r["model"])
              if r["model"] in TEX_MODEL else 99)

    n = runs[0].get("n_per_type")
    clip_k = runs[0].get("clip_images")

    print(r"\begin{table*}[!t]")
    print(r"\caption{Vision--language benchmark on the \emph{released} "
          r"question--answer pairs, scored verbatim: every prompt is the exact "
          r"string shipped in the test split and every target the shipped "
          rf"answer. \num{{{n}}} items per type per model, temperature $0$, "
          r"identical sample and seed. Categorical types are sampled "
          r"class-balanced, so chance is exactly $1/k$; the score is balanced "
          r"accuracy (mean per-class recall), which a constant-answer model "
          rf"cannot inflate. Clip questions are presented as \num{{{clip_k}}} "
          r"evenly spaced frames of the released \num{20}. Spacing is scored by "
          r"mean absolute error against a baseline predicting the training-set "
          r"median gap; that baseline is evaluated on whichever items a model "
          r"answered, so it varies slightly across models "
          r"(\num{8.09}--\num{8.26}\,\si{\metre}) and the column quotes the "
          r"common value.}")
    print(r"\label{tab:vlm_v2}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}llrrrr@{}}")
    print(r"\toprule")
    cols = " & ".join(TEX_MODEL.get(r["model"], r["model"]) for r in runs)
    print(rf"Question & Level & {cols} & Chance/base \\")
    print(r"\midrule")
    for qa_type, label, level in ORDER:
        cells = " & ".join(cell(r, qa_type) for r in runs)
        base = baseline_cell(runs, qa_type)
        print(rf"{label} & {level} & {cells} & {base} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

    # Collapse diagnostics: the characteristic failure here is answering with a
    # single constant label, which an accuracy column alone does not reveal.
    print()
    print("% modal prediction and share, per model and type:")
    for r in runs:
        for t in r["types"]:
            if t.get("kind") == "categorical" and t.get("n"):
                print(f"%   {r['model']:<22}{t['qa_type']:<22}"
                      f"{t['modal_pred']:<20}{t['modal_pred_share']:.2f}"
                      f"  bal={t['balanced_accuracy']:.3f}"
                      f"  chance={t['chance']:.3f}")
            elif t.get("kind") == "numeric" and t.get("n"):
                print(f"%   {r['model']:<22}{t['qa_type']:<22}"
                      f"MAE={t['mae_m']:.2f}  base={t.get('baseline_median_mae_m')}"
                      f"  within5m={t['within_5m']:.3f}"
                      f"  unparsed={t['n_unparsed']}")


if __name__ == "__main__":
    main()
