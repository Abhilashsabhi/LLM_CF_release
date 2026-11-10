"""
make_multimodal_table.py
=========================
Emit the LaTeX table for the multimodal spacing benchmark.

    python make_multimodal_table.py <mm_results.json> > paper/tab_multimodal.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


COND_LABELS = {
    "rgb":             "Front camera",
    "radar":           "Front radar",
    "lidar":           "LiDAR",
    "rgb+radar":       r"Camera\,+\,radar",
    "rgb+lidar":       r"Camera\,+\,LiDAR",
    "rgb+radar+lidar": r"Camera\,+\,radar\,+\,LiDAR",
}


def delta_cell(paired: dict, key: str) -> str:
    if key not in paired:
        return "---"
    p = paired[key]
    v = p["mean_improvement_m"]
    sig = p["significant"]
    sign = "+" if v > 0 else ""
    suffix = "" if sig else r"\,n.s."
    return rf"\num{{{sign}{v:.3f}}}{suffix}"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/mm_results.json"
    d = json.load(open(path))

    conds = d["conditions"]
    paired = d.get("paired_vs_rgb", {})
    n_by_vis = d.get("n_by_visibility", {})
    sc_by_vis = d.get("scenes_by_visibility", {})
    base = d.get("baseline_median_mae_m", float("nan"))

    n_test = sum(n_by_vis.values())
    n_seeds = d.get("seeds", 5)

    print(r"\begin{table*}[!t]")
    print(rf"""\caption{{Multimodal spacing estimation on synchronised single frames: \num{{{n_test}}} test frames, \num{{{n_seeds}}} seeds per condition. Fusion is at the feature level---the camera's \num{{512}}-d penultimate activations, not its scalar output---with a residual head, so a fusion row is not handicapped by having to rediscover what the camera branch already encodes. ``$\Delta$ vs camera'' is a scene-cluster paired bootstrap (resampling scenes, not frames) against the camera-only row, positive meaning fusion is better. \emph{{Skill}} is measured against the median-gap baseline computed within each visibility group, $1-\mathrm{{MAE}}/\mathrm{{MAE}}_{{\text{{base}}}}$; per-group skill rather than raw error is reported because the groups are not equally hard.}}""")
    print(r"\label{tab:multimodal}")
    print(r"\centering")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{}lrlrrrr@{}}")
    print(r"\toprule")
    print(r"& & & \multicolumn{4}{c}{Skill vs.\ baseline, by visibility} \\")
    print(r"\cmidrule(l){4-7}")
    print(r"Sensors & MAE (\si{\metre})$\downarrow$ & $\Delta$ vs camera & clear & rain & fog & night \\")
    print(r"\midrule")

    for key, label in COND_LABELS.items():
        if key not in conds:
            continue
        c = conds[key]
        mae_mean = c["mae_mean"]
        mae_sd = c["mae_sd"]
        mae_cell = rf"\num{{{mae_mean:.3f}}}\,$\pm$\,\num{{{mae_sd:.3f}}}"

        if key == "rgb":
            delta = "---"
        else:
            delta = delta_cell(paired, key)

        bv = c.get("skill_by_visibility", {})
        def sv(k):
            v = bv.get(k)
            return rf"\num{{{v:.3f}}}" if v is not None else "---"

        print(rf"{label} & {mae_cell} & {delta} & {sv('CLEAR')} & {sv('RAIN')} & {sv('FOG')} & {sv('NIGHT')} \\")

    print(r"\midrule")
    print(rf"Median-gap baseline & \num{{{base:.2f}}} & --- & \num{{0.000}} & \num{{0.000}} & \num{{0.000}} & \num{{0.000}} \\")

    vis_order = ["CLEAR", "RAIN", "FOG", "NIGHT"]
    frame_parts = " & ".join(
        rf"\num{{{n_by_vis.get(v, 0)}}}\,/\,\num{{{sc_by_vis.get(v, 0)}}}"
        for v in vis_order
    )
    print(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{test frames / scenes}}}} & {frame_parts} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
