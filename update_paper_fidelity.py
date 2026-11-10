"""
update_paper_fidelity.py
=========================
Update the inline fidelity table in cf_dataset_tits.tex with new numbers
from the LLM_CF_v2 real_validation.json.

Also updates the dataset-size references (1350 → 1345) and the calibration
correction paragraph to reflect the full v2 regeneration.

    python update_paper_fidelity.py <real_validation.json> <tex_file>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_fidelity(path: str):
    d = json.load(open(path))
    out = {}
    for r in d["results"]:
        regime = r["regime"]  # 'urban' or 'highway'
        var = r["variable"]   # spacing_m, ego_speed_mps, ...
        out[(regime, var)] = r
    return out


def fmt(v: float) -> str:
    if abs(v) < 10:
        return f"{v:.3f}"
    return f"{v:.2f}"


def main() -> None:
    val_path = sys.argv[1] if len(sys.argv) > 1 else "real_validation.json"
    tex_path = sys.argv[2] if len(sys.argv) > 2 else "paper/cf_dataset_tits.tex"

    fid = load_fidelity(val_path)
    tex = Path(tex_path).read_text()

    # ---- 1. Dataset size: 1350 → 1345 ----
    tex = tex.replace(r"\num{1350}", r"\num{1345}")

    # ---- 2. Fidelity table: Urban rows ----
    # Variables in display order
    URBAN_VARS = [
        ("spacing_m",          r"Spacing (\si{\metre})"),
        ("ego_speed_mps",      r"Ego speed (\si{\mps})"),
        ("relative_speed_mps", r"Relative speed (\si{\mps})"),
        ("leader_speed_mps",   r"Leader speed (\si{\mps})"),
    ]
    MOTOR_VARS = [
        ("spacing_m",          r"Spacing (\si{\metre})"),
        ("ego_speed_mps",      r"Ego speed (\si{\mps})"),
        ("relative_speed_mps", r"Relative speed (\si{\mps})"),
        ("leader_speed_mps",   r"Leader speed (\si{\mps})"),
    ]

    for var_key, _ in URBAN_VARS:
        r = fid.get(("urban", var_key))
        if not r:
            continue
        ours = r["ours_mean"]
        real = r.get("heldout_mean", 0)
        released = r["w1_heldout"]
        tol = r["tolerance_p95"]
        rr = r.get("w1_real_vs_real", 0)
        var_pattern = var_key.replace("_m", r" \(\\si\{\\metre\}").replace("_mps", r" \(\\si\{\\mps\}")

    # Build the new Urban block
    urban_rows = []
    for var_key, label in URBAN_VARS:
        r = fid.get(("urban", var_key))
        if not r:
            urban_rows.append(f" & {label:<35} & ---")
            continue
        ours = r["ours_mean"]
        real = r.get("heldout_mean", 0)
        released = r["w1_heldout"]
        tol = r["tolerance_p95"]
        rr = r.get("w1_real_vs_real", 0)
        # Check if bolded (improvement)
        old_released = None  # we no longer have a recalib column
        line = (
            f" & {label:<37}       "
            f"& \\num{{{fmt(ours)}}} & \\num{{{fmt(real)}}} "
            f"& \\num{{{fmt(released)}}} "
            f"& \\num{{{fmt(tol)}}} & \\num{{{fmt(rr)}}} \\\\"
        )
        urban_rows.append(line)

    motor_rows = []
    for var_key, label in MOTOR_VARS:
        r = fid.get(("highway", var_key))
        if not r:
            motor_rows.append(f" & {label:<35} & ---")
            continue
        ours = r["ours_mean"]
        real = r.get("heldout_mean", 0)
        released = r["w1_heldout"]
        tol = r["tolerance_p95"]
        rr = r.get("w1_real_vs_real", 0)
        line = (
            f" & {label:<37}       "
            f"& \\num{{{fmt(ours)}}} & \\num{{{fmt(real)}}} "
            f"& \\num{{{fmt(released)}}} "
            f"& \\num{{{fmt(tol)}}} & \\num{{{fmt(rr)}}} \\\\"
        )
        motor_rows.append(line)

    # ---- 3. Replace table header and remove recalib. column ----
    # Old header
    old_hdr = (
        r"\begin{tabular}{@{}llrrrrrc@{}}" + "\n"
        r"\toprule" + "\n"
        r"& & \multicolumn{2}{c}{Mean} & \multicolumn{4}{c}{Wasserstein-1} \\" + "\n"
        r"\cmidrule(lr){3-4}\cmidrule(l){5-8}" + "\n"
        r"Regime & Variable & ours & real & released & recalib.\ & tol.\ (p95) & real--real \\"
    )
    new_hdr = (
        r"\begin{tabular}{@{}llrrrrc@{}}" + "\n"
        r"\toprule" + "\n"
        r"& & \multicolumn{2}{c}{Mean} & \multicolumn{3}{c}{Wasserstein-1} \\" + "\n"
        r"\cmidrule(lr){3-4}\cmidrule(l){5-7}" + "\n"
        r"Regime & Variable & ours & real & $W_1$ & tol.\ (p95) & real--real \\"
    )
    tex = tex.replace(old_hdr, new_hdr)

    # ---- 4. Replace Urban table rows ----
    old_urban = (
        r" & Spacing (\si{\metre})       & \num{19.84} & \num{18.14} & \num{7.378} & \num{9.467} & \num{1.398} & \num{3.682} \\" + "\n"
        r" & Ego speed (\si{\mps})       & \num{7.76}  & \num{5.87}  & \num{4.479} & \num{4.422} & \num{0.870} & \num{2.922} \\" + "\n"
        r" & Relative speed (\si{\mps})  & \num{0.00}  & \num{-0.03} & \num{0.916} & \num{0.911} & \num{0.142} & \num{0.242} \\" + "\n"
        r" & Leader speed (\si{\mps})    & \num{7.76}  & \num{5.83}  & \num{4.715} & \num{4.667} & \num{0.888} & \num{2.952} \\"
    )

    r = fid[("urban", "spacing_m")]
    u_sp = f" & Spacing (\\si{{\\metre}})       & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("urban", "ego_speed_mps")]
    u_es = f" & Ego speed (\\si{{\\mps}})       & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("urban", "relative_speed_mps")]
    u_rs = f" & Relative speed (\\si{{\\mps}})  & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("urban", "leader_speed_mps")]
    u_ls = f" & Leader speed (\\si{{\\mps}})    & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"

    new_urban = "\n".join([u_sp, u_es, u_rs, u_ls])
    tex = tex.replace(old_urban, new_urban)

    # ---- 5. Replace Motorway rows ----
    old_motor = (
        r" & Spacing (\si{\metre})       & \num{28.33} & \num{29.51} & \num{6.665} & \textbf{\num{4.488}} & \num{1.912} & \num{12.218} \\" + "\n"
        r" & Ego speed (\si{\mps})       & \num{25.24} & \num{20.88} & \num{4.356} & \textbf{\num{2.569}} & \num{0.572} & \num{4.620} \\" + "\n"
        r" & Relative speed (\si{\mps})  & \num{-0.15} & \num{-0.06} & \num{0.669} & \num{0.719} & \num{0.105} & \num{0.892} \\" + "\n"
        r" & Leader speed (\si{\mps})    & \num{25.39} & \num{20.82} & \num{4.564} & \textbf{\num{2.442}} & \num{0.558} & \num{4.838} \\"
    )
    r = fid[("highway", "spacing_m")]
    m_sp = f" & Spacing (\\si{{\\metre}})       & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("highway", "ego_speed_mps")]
    m_es = f" & Ego speed (\\si{{\\mps}})       & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("highway", "relative_speed_mps")]
    m_rs = f" & Relative speed (\\si{{\\mps}})  & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"
    r = fid[("highway", "leader_speed_mps")]
    m_ls = f" & Leader speed (\\si{{\\mps}})    & \\num{{{fmt(r['ours_mean'])}}} & \\num{{{fmt(r.get('heldout_mean',0))}}} & \\num{{{fmt(r['w1_heldout'])}}} & \\num{{{fmt(r['tolerance_p95'])}}} & \\num{{{fmt(r.get('w1_real_vs_real',0))}}} \\\\"

    new_motor = "\n".join([m_sp, m_es, m_rs, m_ls])
    tex = tex.replace(old_motor, new_motor)

    Path(tex_path).write_text(tex)
    print(f"Updated {tex_path}")


if __name__ == "__main__":
    main()
