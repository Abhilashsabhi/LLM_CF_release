"""
make_cfbench_table.py
======================
Emit the LaTeX table for the stratified car-following benchmark.

Two kinds of uncertainty appear here and they must not be mixed. A
deterministic policy is run once, so its uncertainty is over which events were
sampled, and we bootstrap that. A trained model is run under several seeds, so
its dominant uncertainty is over initialisation, and we report the spread
across seeds. Quoting an across-seed mean beside a single-seed event interval
produces figures that contradict each other -- the mean can fall outside its
own stated interval -- so each row carries the interval that belongs to it, and
the caption says which.

    python make_cfbench_table.py /var/tmp/cfbench_v2/cfbench_v2.json
"""

from __future__ import annotations

import json
import sys
from typing import Dict

ROWS = [
    ("constant", r"Constant (hold $a=0$)", False),
    ("idm", r"IDM (calibrated)", False),
    ("idm_residual", r"IDM\,+\,residual", True),
    ("mlp", r"MLP", True),
    ("lstm", r"LSTM (16 units)", True),
    ("gru", r"GRU ($2\times128$)", True),
    ("transformer", r"Transformer encoder", True),
]


def cell(s: Dict, key: str, neural: bool) -> str:
    if key not in s:
        return "---"
    v = s[key]
    dp = 2 if key == "spacing_rmse" else 3
    if neural:
        return (f"\\num{{{v['mean']:.{dp}f}}}\\,$\\pm$\\,"
                f"\\num{{{v['sd']:.{dp}f}}}")
    lo, hi = s.get("event_rmse_ci" if key == "spacing_rmse"
                   else "event_collision_ci", (None, None))
    if lo is None:
        return f"\\num{{{v['mean']:.2f}}}"
    fmt = "%.2f" if key == "spacing_rmse" else "%.3f"
    return (f"\\num{{{fmt % v['mean']}}} "
            f"[\\num{{{fmt % lo}}}, \\num{{{fmt % hi}}}]")


def main() -> None:
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                       else "/var/tmp/cfbench_v2/cfbench_v2.json"))
    st, ex = d["strata"], d["excitation_mps2"]

    print(r"\begin{table*}[!t]")
    print(rf"""\caption{{Car-following benchmark under the FollowNet closed-loop
protocol, stratified by whether the event contains a scripted disturbance.
\emph{{Nominal}} is the two scenario categories that inject nothing
(\num{{{st['nominal']}}} events, leader-acceleration standard deviation
\SI{{{ex['nominal']}}}{{\mpss}}); \emph{{excited}} is the seven that do
(\num{{{st['excited']}}} events, \SI{{{ex['excited']}}}{{\mpss}}). Trained
models are run under \num{{{d['seeds']}}} seeds and reported as
mean\,$\pm$\,standard deviation across seeds; the two deterministic policies
are run once and carry a \SI{{95}}{{\percent}} bootstrap interval over events.
The two uncertainties are not interchangeable and are not mixed within a row.}}""")
    print(r"\label{tab:cfbench_v2}")
    print(r"\centering")
    # The deterministic rows carry bracketed intervals, which are wide; trim
    # the inter-column padding rather than drop the intervals.
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{}lrrrr@{}}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{Nominal} & \multicolumn{2}{c}{Excited} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(l){4-5}")
    print(r"Policy & RMSE (\si{\metre})$\downarrow$ & collisions$\downarrow$"
          r" & RMSE (\si{\metre})$\downarrow$ & collisions$\downarrow$ \\")
    print(r"\midrule")
    for key, label, neural in ROWS:
        m = d["models"].get(key)
        if not m:
            continue
        n, e = m["strata"]["nominal"], m["strata"]["excited"]
        print(f"{label} & {cell(n,'spacing_rmse',neural)} & "
              f"{cell(n,'collision_rate',neural)} & "
              f"{cell(e,'spacing_rmse',neural)} & "
              f"{cell(e,'collision_rate',neural)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
