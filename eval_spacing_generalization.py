"""
eval_spacing_generalization.py
===============================
Does the monocular spacing model generalise, or exploit the corpus?

A scene-level split stops identical frames crossing splits, but synthetic
imagery leaks in other ways: the same eight CARLA towns appear in every split,
and---more sharply---every scene uses the same lead vehicle. Both the training
and the evaluation leader is \\texttt{vehicle.tesla.model3}, so a model can
learn one vehicle's apparent size rather than monocular ranging, and the
scene-level split would not detect it.

Two probes are run against the released checkpoint.

*Held-out vehicle.* The cut-in manoeuvre spawns \\texttt{vehicle.audi.a2},
which becomes the recorded leader once it occupies the ego lane. Frames of
cut-in scenes with an active injection therefore have a leader the model has
seen far less often and whose dimensions differ. If accuracy collapses there,
the model is keying on a known silhouette.

*Per-town error.* Reported for every town, so that a town whose geometry the
model exploits is visible even though all towns appear in training.

All intervals are bootstrapped over SCENES, not frames. Probe frames are drawn
from a few hundred scenes and frames within a scene are highly correlated, so a
frame-level bootstrap treats correlated observations as independent and
understates the interval.

    python eval_spacing_generalization.py --dataset <root> --ckpt <pt> --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("spacegen")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] spacegen: %(message)s")

W, H = 320, 180
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cluster_boot(err: np.ndarray, scene: np.ndarray, n: int = 4000,
                 seed: int = 0) -> Tuple[float, float]:
    """Bootstrap the mean absolute error by resampling SCENES.

    Frames from one scene are near-duplicates; resampling frames would treat
    them as independent draws and give an interval that is too narrow. We
    resample whole scenes with replacement and recompute the statistic over
    the frames they carry, which is the cluster bootstrap.
    """
    scenes = np.unique(scene)
    if len(scenes) < 3:
        return (float("nan"), float("nan"))
    by = {s: err[scene == s] for s in scenes}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        out[i] = np.concatenate([by[s] for s in pick]).mean()
    return (round(float(np.percentile(out, 2.5)), 3),
            round(float(np.percentile(out, 97.5)), 3))


def load_model(ckpt: Path):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    sd = torch.load(ckpt, map_location=dev)
    m.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
    return m.to(dev).eval(), dev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    import torch
    from PIL import Image

    root = Path(args.dataset)
    model, dev = load_model(Path(args.ckpt))
    scenes = json.load(open(root / "infos" / f"cf_infos_{args.split}.json"))["data"]

    rec: List[Dict] = []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            t = torch.from_numpy(np.stack(buf)).to(dev)
            pr = model(t).squeeze(-1).cpu().numpy()
        for m_, v in zip(meta, pr):
            rec.append({**m_, "pred": float(v)})
        buf.clear(); meta.clear()

    for n, sc in enumerate(scenes):
        si = sc["scene_info"]
        cat = si.get("scenario_category")
        town = si.get("carla_map")
        for fr in sc["scene_data"][::args.frame_stride]:
            ip = fr["sensor_paths"].get("RGB-CAM_FRONT")
            if not ip:
                continue
            try:
                im = Image.open(root / ip).convert("RGB").resize((W, H),
                                                                 Image.BILINEAR)
            except (OSError, ValueError):
                continue
            x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
            buf.append(x.transpose(2, 0, 1))
            # A cut-in scene with an active injection has the Audi as leader;
            # every other frame has the Tesla.
            audi = bool(cat == "cut_in" and fr["kinematics"].get("injection_active"))
            meta.append({"scene_id": sc["scene_id"], "town": town,
                         "category": cat, "audi_leader": audi,
                         "gt": float(fr["kinematics"]["spacing_m"])})
            if len(buf) == args.batch:
                flush()
        if (n + 1) % 40 == 0:
            logger.info("  %d/%d scenes, %d frames", n + 1, len(scenes), len(rec))
    flush()
    logger.info("%d frames scored", len(rec))

    err = np.array([abs(r["pred"] - r["gt"]) for r in rec])
    sid = np.array([r["scene_id"] for r in rec])
    town = np.array([r["town"] for r in rec])
    audi = np.array([r["audi_leader"] for r in rec])

    out: Dict = {"n_frames": int(len(rec)),
                 "n_scenes": int(len(np.unique(sid))),
                 "overall": {"mae": round(float(err.mean()), 3),
                             "ci95_scene_cluster": cluster_boot(err, sid)}}

    # Held-out vehicle probe.
    out["by_leader_vehicle"] = {}
    for name, m in (("tesla_model3", ~audi), ("audi_a2_cutin", audi)):
        if m.sum() < 10:
            continue
        out["by_leader_vehicle"][name] = {
            "n_frames": int(m.sum()), "n_scenes": int(len(np.unique(sid[m]))),
            "mae": round(float(err[m].mean()), 3),
            "ci95_scene_cluster": cluster_boot(err[m], sid[m])}

    # Per-town error.
    out["by_town"] = {}
    for t in sorted(set(town.tolist())):
        m = town == t
        out["by_town"][t] = {
            "n_frames": int(m.sum()), "n_scenes": int(len(np.unique(sid[m]))),
            "mae": round(float(err[m].mean()), 3),
            "ci95_scene_cluster": cluster_boot(err[m], sid[m])}

    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n{out['n_frames']} frames from {out['n_scenes']} scenes")
    print(f"overall MAE {out['overall']['mae']:.2f} m "
          f"[scene-cluster 95% {out['overall']['ci95_scene_cluster']}]")
    print("\nby leader vehicle:")
    for k, v in out["by_leader_vehicle"].items():
        print(f"  {k:<16}{v['mae']:>7.2f} m  {str(v['ci95_scene_cluster']):>18}"
              f"  ({v['n_frames']} frames, {v['n_scenes']} scenes)")
    print("\nby town:")
    for k, v in out["by_town"].items():
        print(f"  {k:<10}{v['mae']:>7.2f} m  {str(v['ci95_scene_cluster']):>18}"
              f"  ({v['n_scenes']} scenes)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
