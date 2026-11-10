"""
make_generalization_table.py
=============================
Emit the LaTeX table for the spacing CNN generalisation probes.

Reads:
  spacing_gen.json    from eval_spacing_generalization.py
  crosstown.json      from eval_spacing_crosstown.py

    python make_generalization_table.py <gen.json> <crosstown.json> > paper/tab_generalization.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def ci_cell(ci: list) -> str:
    return rf"[\num{{{ci[0]:.2f}}}, \num{{{ci[1]:.2f}}}]"


def main() -> None:
    gen_path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/spacing_gen.json"
    cross_path = sys.argv[2] if len(sys.argv) > 2 else "/var/tmp/crosstown.json"

    gen = json.load(open(gen_path))
    cross = json.load(open(cross_path))

    by_vehicle = gen.get("by_leader_vehicle", {})
    tesla = by_vehicle.get("tesla_model3", {})
    audi = by_vehicle.get("audi_a2_cutin", {})

    holdouts = cross.get("holdouts", {})
    urban = holdouts.get("urban_T01_T02", {})
    motorway = holdouts.get("motorway_T06", {})

    urban_cross = urban.get("cross_town", {})
    urban_within = urban.get("within_town_control", {})
    urban_deg = urban.get("degradation_m", float("nan"))
    motor_cross = motorway.get("cross_town", {})
    motor_within = motorway.get("within_town_control", {})
    motor_deg = motorway.get("degradation_m", float("nan"))

    held_towns_urban = "+".join(urban.get("held_out_towns", ["Town01", "Town02"]))
    held_towns_motor = urban.get("held_out_towns", ["Town06"])[-1]

    print(r"\begin{table}[!t]")
    print(r"""\caption{Generalisation probes for the supervised monocular spacing reference. All intervals are \SI{95}{\percent} bootstrap resampling \emph{scenes}, not frames. Upper block: the effect of the leader's vehicle model. Every scene in the release uses the same lead vehicle, and the cut-in manoeuvre is the only source of a different one, so those frames are the sole held-out-vehicle evidence available. Lower block: models retrained with whole towns withheld. Each holdout is paired on its own row with a control trained on an identical number of frames and tested on held-out \emph{scenes} of the towns it did train on, so the difference isolates the town shift rather than the training-set size.}""")
    print(r"\label{tab:generalization}")
    print(r"\centering")
    print(r"\setlength{\tabcolsep}{4pt}\footnotesize")
    print(r"\begin{tabular}{@{}lrrl@{}}")
    print(r"\toprule")
    print(r"Condition & Scenes & MAE (\si{\metre})$\downarrow$ & \SI{95}{\percent} CI \\")
    print(r"\midrule")
    print(r"\multicolumn{4}{@{}l}{\emph{(a) Leader vehicle, released checkpoint}} \\")

    t_scenes = tesla.get("n_scenes", "---")
    t_mae = tesla.get("mae", float("nan"))
    t_ci = tesla.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"Tesla Model 3 (in all training scenes) & \num{{{t_scenes}}} & \num{{{t_mae:.2f}}} & {ci_cell(t_ci)} \\")

    a_scenes = audi.get("n_scenes", "---")
    a_mae = audi.get("mae", float("nan"))
    a_ci = audi.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"Audi A2 (cut-in frames only) & \num{{{a_scenes}}} & \num{{{a_mae:.2f}}} & {ci_cell(a_ci)} \\")

    print(r"\midrule")
    print(r"\multicolumn{4}{@{}l}{\emph{(b) Held-out towns, each retrained from scratch}} \\")

    print(rf"\multicolumn{{4}}{{@{{}}l}}{{\quad holding out \textbf{{{held_towns_urban}}} (degradation \SI{{{urban_deg:.2f}}}{{\metre}})}} \\")
    wsc = urban_within.get("n_test_scenes", "---")
    wm = urban_within.get("mae", float("nan"))
    wci = urban_within.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"\qquad control: seen towns, unseen scenes & \num{{{wsc}}} & \num{{{wm:.2f}}} & {ci_cell(wci)} \\")
    csc = urban_cross.get("n_test_scenes", "---")
    cm = urban_cross.get("mae", float("nan"))
    cci = urban_cross.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"\qquad test: unseen towns & \num{{{csc}}} & \num{{{cm:.2f}}} & {ci_cell(cci)} \\")

    print(rf"\multicolumn{{4}}{{@{{}}l}}{{\quad holding out \textbf{{Town06}} (degradation \SI{{{motor_deg:.2f}}}{{\metre}})}} \\")
    wsc2 = motor_within.get("n_test_scenes", "---")
    wm2 = motor_within.get("mae", float("nan"))
    wci2 = motor_within.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"\qquad control: seen towns, unseen scenes & \num{{{wsc2}}} & \num{{{wm2:.2f}}} & {ci_cell(wci2)} \\")
    csc2 = motor_cross.get("n_test_scenes", "---")
    cm2 = motor_cross.get("mae", float("nan"))
    cci2 = motor_cross.get("ci95_scene_cluster", [float("nan"), float("nan")])
    print(rf"\qquad test: unseen towns & \num{{{csc2}}} & \num{{{cm2:.2f}}} & {ci_cell(cci2)} \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
