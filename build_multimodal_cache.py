"""
build_multimodal_cache.py
==========================
Cache per-frame sensor features for the multimodal spacing benchmark.

The release ships fourteen modalities and the perception experiments so far use
one of them, which leaves open why the rest are worth releasing. This builds the
inputs for a compact comparison -- front camera against front radar against
LiDAR, and their fusions -- on the quantity car-following actually needs.

Feature design is deliberately plain. The point is to establish whether the
modalities carry the information and whether synchronising them buys anything,
not to propose an architecture; an elaborate encoder would confound the two.

*Radar* returns are \\texttt{[depth, altitude, azimuth, velocity]}. Only depth
and the angles are used: the velocity channel of this release is defective
(Section~\\ref{sec:limitations}) and correlates \\num{0.017} with true relative
speed, so a model given it would be learning noise. Features are the nearest
boresight return, low percentiles of boresight depth, the return count, and an
azimuth-binned minimum-depth profile.

*LiDAR* points are \\texttt{(x, y, z)} with $+x$ forward, established
empirically rather than assumed. Ground and overhead returns are removed by
height, and the forward sector is reduced to an azimuth-binned minimum-range
profile -- the natural analogue of the radar feature, so the two modalities are
compared on like representations rather than on encoder sophistication.

    python build_multimodal_cache.py --dataset <root> --out <file.npz>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mmcache")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] mmcache: %(message)s")

RAD_BINS = 8
LID_BINS = 16
LID_FOV = np.deg2rad(35.0)      # forward sector half-angle
MAX_RANGE = 100.0


def radar_features(a: np.ndarray) -> np.ndarray:
    """[nearest boresight, p10, p25 boresight, n_boresight, n_all, profile...]"""
    out = np.full(5 + RAD_BINS, MAX_RANGE, dtype=np.float32)
    if len(a) == 0:
        out[3] = out[4] = 0.0
        return out
    d, alt, az = a[:, 0], a[:, 1], a[:, 2]
    bore = (np.abs(az) < 0.05) & (np.abs(alt) < 0.05)
    if bore.any():
        db = d[bore]
        out[0] = db.min()
        out[1] = np.percentile(db, 10)
        out[2] = np.percentile(db, 25)
    out[3] = float(bore.sum())
    out[4] = float(len(a))
    edges = np.linspace(-0.25, 0.25, RAD_BINS + 1)
    idx = np.digitize(az, edges) - 1
    for b in range(RAD_BINS):
        m = idx == b
        if m.any():
            out[5 + b] = d[m].min()
    return out


def lidar_features(p: np.ndarray) -> np.ndarray:
    """Azimuth-binned minimum forward range, plus occupancy counts."""
    out = np.full(LID_BINS + 3, MAX_RANGE, dtype=np.float32)
    if len(p) == 0:
        out[LID_BINS:] = 0.0
        return out
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    keep = (z > -1.2) & (z < 1.5) & (x > 1.5)
    if not keep.any():
        out[LID_BINS:] = 0.0
        return out
    x, y = x[keep], y[keep]
    az = np.arctan2(y, x)
    rng = np.hypot(x, y)
    sector = np.abs(az) < LID_FOV
    edges = np.linspace(-LID_FOV, LID_FOV, LID_BINS + 1)
    idx = np.digitize(az[sector], edges) - 1
    r = rng[sector]
    for b in range(LID_BINS):
        m = idx == b
        if m.any():
            out[b] = r[m].min()
    lane = sector & (np.abs(y) < 1.6)
    out[LID_BINS] = rng[lane].min() if lane.any() else MAX_RANGE
    out[LID_BINS + 1] = float(lane.sum())
    out[LID_BINS + 2] = float(sector.sum())
    return out


def collect(root: Path, split: str, scene_stride: int, frame_stride: int
            ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    f = root / "infos" / f"cf_infos_{split}.json"
    R, L, imgs, meta = [], [], [], []
    scenes = json.load(open(f))["data"][::scene_stride]
    for n, sc in enumerate(scenes):
        sid = sc["scene_id"]
        for fr in sc["scene_data"][::frame_stride]:
            i = fr["frame_idx"]
            k = fr["kinematics"]
            rp = root / f"sweeps/RAD_FRONT/{sid}/{sid}-frame-{i:04d}-RAD_FRONT.npy"
            lp = root / f"sweeps/LIDAR/{sid}/{sid}-frame-{i:04d}-LIDAR.npy"
            ip = fr["sensor_paths"].get("RGB-CAM_FRONT")
            if not (rp.exists() and lp.exists() and ip):
                continue
            try:
                R.append(radar_features(np.load(rp)))
                L.append(lidar_features(np.load(lp)))
            except (OSError, ValueError):
                continue
            imgs.append(ip)
            meta.append({"scene_id": sid, "frame_idx": i,
                         "spacing_m": float(k["spacing_m"]),
                         "ttc_s": float(k["ttc_s"]),
                         "regime": sc["scene_info"].get("regime"),
                         "scenario_category": sc["scene_info"].get("scenario_category")})
        if (n + 1) % 25 == 0:
            logger.info("  %s: %d/%d scenes, %d frames", split, n + 1,
                        len(scenes), len(meta))
    return np.asarray(R), np.asarray(L), imgs, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scene-stride", type=int, default=2)
    p.add_argument("--frame-stride", type=int, default=15)
    p.add_argument("--test-frame-stride", type=int, default=10)
    args = p.parse_args()

    root = Path(args.dataset)
    store: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        fs = args.test_frame_stride if split == "test" else args.frame_stride
        ss = 1 if split != "train" else args.scene_stride
        R, L, imgs, meta = collect(root, split, ss, fs)
        logger.info("%s: %d frames", split, len(meta))
        store[f"{split}_radar"] = R
        store[f"{split}_lidar"] = L
        store[f"{split}_y"] = np.array([m["spacing_m"] for m in meta], np.float32)
        store[f"{split}_ttc"] = np.array([m["ttc_s"] for m in meta], np.float32)
        store[f"{split}_img"] = np.array(imgs)
        store[f"{split}_meta"] = np.array([json.dumps(m) for m in meta])
    np.savez_compressed(args.out, **store)
    print(f"\nwrote {args.out}")
    for s in ("train", "val", "test"):
        print(f"  {s:<6}{len(store[f'{s}_y']):>7} frames  "
              f"radar {store[f'{s}_radar'].shape[1]}d  "
              f"lidar {store[f'{s}_lidar'].shape[1]}d")


if __name__ == "__main__":
    main()
