"""
export_cf_arrays.py
====================
Export the dataset in the standard car-following array format:

    train.npy / val.npy / test.npy   shape (n_events, 4, T), float32
    features (axis 1), in order:
        0  spacing                 [m]   bumper-to-bumper
        1  subject_vehicle_speed   [m/s]
        2  relative_speed          [m/s] (see --dv-convention)
        3  leading_vehicle_speed   [m/s]

Sign convention for relative speed
----------------------------------
The dataset's internal ground truth stores  dv = v_ego - v_lead  (positive
means the follower is CLOSING). Much of the car-following literature uses the
opposite,  dv = v_lead - v_ego  (positive means the leader is pulling away).

The two are trivially interconvertible and both speeds are present in the
array, so nothing is lost either way — but a silent mismatch will corrupt any
model that assumes one and receives the other. The convention is therefore an
explicit flag, recorded in the sidecar metadata, and defaults to the more
common `leader_minus_ego`.

Windowing
---------
Scenes are 300 frames (30 s at 10 Hz). Passing --window/--stride emits
fixed-length sub-sequences instead of whole events, which is what most
sequence models want. Windows never span two scenes.

Usage
-----
    python export_cf_arrays.py --dataset /path/to/cf_dataset --out ./cf_arrays
    python export_cf_arrays.py --dataset ... --window 100 --stride 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("export")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] export: %(message)s")

FEATURES = ["spacing", "subject_vehicle_speed", "relative_speed",
            "leading_vehicle_speed"]
SPLITS = ("train", "val", "test")


def scene_matrix(scene: Dict[str, Any], dv_convention: str) -> np.ndarray:
    """One scene -> (4, T) float32."""
    frames = scene["scene_data"]
    sp, ve, vl = [], [], []
    for f in frames:
        k = f["kinematics"]
        sp.append(k["spacing_m"])
        ve.append(k["ego_speed_mps"])
        vl.append(k["leader_speed_mps"])
    sp = np.asarray(sp, dtype=np.float32)
    ve = np.asarray(ve, dtype=np.float32)
    vl = np.asarray(vl, dtype=np.float32)
    dv = (vl - ve) if dv_convention == "leader_minus_ego" else (ve - vl)
    return np.stack([sp, ve, dv, vl], axis=0)


def window(mat: np.ndarray, size: int, stride: int) -> List[np.ndarray]:
    """Fixed-length windows from one scene; never spans scenes."""
    T = mat.shape[1]
    if size <= 0 or size >= T:
        return [mat]
    return [mat[:, s:s + size] for s in range(0, T - size + 1, stride)]


def export_split(dataset: Path, split: str, out: Path,
                 dv_convention: str, size: int, stride: int
                 ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    info_path = dataset / "infos" / f"cf_infos_{split}.json"
    if not info_path.exists():
        raise SystemExit(f"missing {info_path}")
    with open(info_path) as f:
        scenes = json.load(f)["data"]

    arrays: List[np.ndarray] = []
    meta:   List[Dict[str, Any]] = []
    for sc in scenes:
        mat = scene_matrix(sc, dv_convention)
        info = sc.get("scene_info", {})
        for w_i, w in enumerate(window(mat, size, stride)):
            arrays.append(w)
            meta.append({
                "scene_id":          sc.get("scene_id"),
                "window_index":      w_i,
                "driving_style":     info.get("driving_style"),
                "scenario_category": info.get("scenario_category"),
                "regime":            info.get("regime"),
                "carla_map":         info.get("carla_map"),
            })

    if not arrays:
        raise SystemExit(f"no events for split {split}")
    data = np.stack(arrays, axis=0).astype(np.float32)
    np.save(out / f"{split}.npy", data)
    with open(out / f"{split}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return data, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="./cf_arrays")
    p.add_argument("--dv-convention", default="leader_minus_ego",
                   choices=["leader_minus_ego", "ego_minus_leader"],
                   help="Sign of the relative-speed channel "
                        "(default: leader_minus_ego)")
    p.add_argument("--window", type=int, default=0,
                   help="Window length in frames (0 = whole 300-frame scenes)")
    p.add_argument("--stride", type=int, default=0,
                   help="Window stride (default: = window, non-overlapping)")
    args = p.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stride = args.stride or (args.window if args.window else 1)

    summary: Dict[str, Any] = {
        "features": FEATURES,
        "axis_order": "(n_events, n_features=4, sequence_length)",
        "dtype": "float32",
        "units": {"spacing": "m", "speeds": "m/s"},
        "sampling_rate_hz": 10,
        "dv_convention": args.dv_convention,
        "dv_definition": ("v_lead - v_ego (positive: leader pulling away)"
                          if args.dv_convention == "leader_minus_ego"
                          else "v_ego - v_lead (positive: follower closing)"),
        "window": args.window or 300,
        "stride": stride,
        "source_dataset": str(dataset),
        "note": ("Only scenes admitted by the dataset's scene-admission "
                 "criterion are exported; rejected scenes are listed in "
                 "infos/excluded_scenes.json of the source dataset."),
        "splits": {},
    }

    for split in SPLITS:
        data, meta = export_split(dataset, split, out,
                                  args.dv_convention, args.window, stride)
        # Report per-channel statistics so a mismatch against an existing
        # pipeline is visible immediately rather than after training.
        stats = {FEATURES[i]: {"mean": float(data[:, i].mean()),
                               "std":  float(data[:, i].std()),
                               "min":  float(data[:, i].min()),
                               "max":  float(data[:, i].max())}
                 for i in range(4)}
        summary["splits"][split] = {
            "shape": list(data.shape),
            "n_events": int(data.shape[0]),
            "n_scenes": len({m["scene_id"] for m in meta}),
            "channel_stats": stats,
        }
        logger.info("%-5s %s  (%d scenes)", split, data.shape,
                    len({m["scene_id"] for m in meta}))

    with open(out / "format.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Identity that must hold in every exported row, regardless of convention.
    d = np.load(out / "test.npy")
    sp, ve, dv, vl = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    expect = (vl - ve) if args.dv_convention == "leader_minus_ego" else (ve - vl)
    err = float(np.abs(dv - expect).max())
    logger.info("consistency check: max |dv - expected| = %.2e", err)
    if err > 1e-3:
        raise SystemExit("relative-speed channel is inconsistent with the speeds")
    logger.info("wrote %s/{train,val,test}.npy + *_meta.json + format.json", out)


if __name__ == "__main__":
    main()
