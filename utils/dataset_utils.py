"""
utils/dataset_utils.py
=======================
SimBEV-compatible folder layout, file naming, GT kinematic saving,
and info-file builder.

Folder structure (mirrors SimBEV exactly):
    cf_dataset/
    ├── configs/
    ├── console_logs/
    ├── ground-truth/
    │   ├── det/
    │   └── hd_map/
    ├── infos/
    ├── logs/
    └── sweeps/
        ├── RGB-CAM_FRONT/          ← .jpg
        ├── RGB-CAM_FRONT_LEFT/
        ├── RGB-CAM_FRONT_RIGHT/
        ├── RGB-CAM_BACK/
        ├── RGB-CAM_BACK_LEFT/
        ├── RGB-CAM_BACK_RIGHT/
        ├── LIDAR/                  ← .npy  (n,3)
        ├── SEG-LIDAR/              ← .npy  (n,6)
        ├── RAD_FRONT/              ← .npy  (n,4)
        ├── RAD_LEFT/
        ├── RAD_RIGHT/
        ├── RAD_BACK/
        ├── GNSS/                   ← .npy  (3,)
        └── IMU/                    ← .npy  (7,)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SWEEP_DIRS: List[str] = [
    "RGB-CAM_FRONT", "RGB-CAM_FRONT_LEFT",  "RGB-CAM_FRONT_RIGHT",
    "RGB-CAM_BACK",  "RGB-CAM_BACK_LEFT",   "RGB-CAM_BACK_RIGHT",
    "LIDAR", "SEG-LIDAR",
    "RAD_FRONT", "RAD_LEFT", "RAD_RIGHT", "RAD_BACK",
    "GNSS", "IMU",
]


class DatasetLayout:
    def __init__(self, root: str):
        self.root = Path(root)

    def initialise(self) -> None:
        dirs = (["configs", "console_logs", "logs", "infos",
                 "ground-truth/det", "ground-truth/hd_map"]
                + [f"sweeps/{d}" for d in SWEEP_DIRS])
        for d in dirs:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────

    # Files are SHARDED into a per-scene sub-directory.
    #
    # A flat directory holds n_scenes x 300 files — 405,000 per sensor at
    # journal scale. On exFAT (the usual format for a portable drive) insert
    # cost grows with directory size: measured 4248 files/s at 1k entries but
    # 178 files/s at 20k, i.e. roughly O(n^2) overall, which extrapolates to
    # ~13 hours to fill ONE sensor directory. Sharding holds each directory
    # at 300 entries and keeps the rate flat at ~9000 files/s.
    #
    # The filename still carries the scene id, so a file remains
    # self-describing and SimBEV-style tooling that parses the stem keeps
    # working; only the directory nesting changed.

    def sweep_path(self, sensor: str, scene_id: str, frame_idx: int) -> Path:
        ext = ".jpg" if sensor.startswith("RGB") else ".npy"
        return (self.root / "sweeps" / sensor / scene_id
                / f"{scene_id}-frame-{frame_idx:04d}-{sensor}{ext}")

    def gt_det_path(self, scene_id: str, frame_idx: int) -> Path:
        return (self.root / "ground-truth" / "det" / scene_id
                / f"{scene_id}-frame-{frame_idx:04d}-GT_DET.npy")

    def gt_hdmap_path(self, scene_id: str, frame_idx: int) -> Path:
        return (self.root / "ground-truth" / "hd_map" / scene_id
                / f"{scene_id}-frame-{frame_idx:04d}-HD_MAP.json")

    def config_path(self, scene_id: str) -> Path:
        return self.root / "configs" / f"{scene_id}.yaml"

    def log_path(self, scene_id: str) -> Path:
        return self.root / "logs" / f"{scene_id}.log"


# ---------------------------------------------------------------------------
# GT kinematic record (one float64 per GT_FIELD_NAMES entry)
# ---------------------------------------------------------------------------

GT_FIELD_NAMES: List[str] = [
    "timestamp_s",
    "spacing_m",
    "ego_speed_mps",
    "leader_speed_mps",
    "relative_speed_mps",    # ego_v − leader_v  (positive → approaching)
    "ego_acceleration_mps2",
    "leader_acceleration_mps2",
    "ttc_s",
    "ego_x", "ego_y", "ego_z",
    "leader_x", "leader_y", "leader_z",
    "ego_yaw_rad",
    "leader_yaw_rad",
    "injection_active",      # 0 or 1
    "driving_style_id",      # 0=cautious 1=normal 2=aggressive
    "scenario_category_id",  # 0-8
    "pedestrian_present",    # 0 or 1
    "collision",             # 0 or 1 — CARLA collision sensor, sticky
]

STYLE_TO_ID    = {"cautious": 0, "normal": 1, "aggressive": 2}
CATEGORY_TO_ID = {cat: i for i, cat in enumerate([
    "steady_follow", "stop_and_go", "aggressive_leader",
    "cut_in", "cut_out", "speed_limit_change",
    "grade_change", "adverse_weather", "pedestrian_crossing",
])}


def save_gt_kinematics(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array([
        rec.get("timestamp_s",              0.0),
        rec.get("spacing_m",                0.0),
        rec.get("ego_speed_mps",            0.0),
        rec.get("leader_speed_mps",         0.0),
        rec.get("relative_speed_mps",       0.0),
        rec.get("ego_acceleration_mps2",    0.0),
        rec.get("leader_acceleration_mps2", 0.0),
        rec.get("ttc_s",                    9999.0),
        rec.get("ego_x",    0.0), rec.get("ego_y",    0.0), rec.get("ego_z",    0.0),
        rec.get("leader_x", 0.0), rec.get("leader_y", 0.0), rec.get("leader_z", 0.0),
        rec.get("ego_yaw_rad",     0.0),
        rec.get("leader_yaw_rad",  0.0),
        float(rec.get("injection_active",   False)),
        float(STYLE_TO_ID.get(rec.get("driving_style",    "normal"),   1)),
        float(CATEGORY_TO_ID.get(rec.get("scenario_category", "steady_follow"), 0)),
        float(rec.get("pedestrian_present", False)),
        float(rec.get("collision", False)),
    ], dtype=np.float64)
    np.save(str(path), arr)


def load_gt_kinematics(path: Path) -> Dict[str, Any]:
    """Tolerant of records written before a field was appended."""
    arr = np.load(str(path))
    return {GT_FIELD_NAMES[i]: float(arr[i])
            for i in range(min(len(arr), len(GT_FIELD_NAMES)))}


# ---------------------------------------------------------------------------
# Sensor calibration metadata
# ---------------------------------------------------------------------------

def _camera_geometry() -> tuple:
    """Resolution and FOV of the front camera, taken from the collector.

    These were duplicated as literals here. When the capture resolution was
    lowered from 1920x1080 to 1280x720 the copy did not follow, so the release
    would have advertised fx = 1371 px for imagery actually rendered at 720p
    (true fx = 914) with the principal point off by a factor of 1.5. Nothing
    would have failed loudly; any geometric use of the data would simply have
    been wrong. Reading the collector's own table keeps the two in step.
    """
    try:
        import stage2_carla_executor as S          # late: avoids a cycle
        for name, *_rest in S.CAMERA_CFGS:
            if name == "RGB-CAM_FRONT":
                w, h, fov = _rest[6], _rest[7], _rest[8]
                return int(w), int(h), float(fov)
    except Exception:                              # collector not importable
        pass
    return 1280, 720, 70.0


def sensor_metadata() -> Dict:
    W, H, fov = _camera_geometry()
    fx = W / (2.0 * math.tan(math.radians(fov / 2.0)))
    return {
        "camera_intrinsic": {
            "fx": fx, "fy": fx,
            "cx": W / 2.0, "cy": H / 2.0,
            "width": W, "height": H, "fov_deg": fov,
        },
        # FLU coords; CARLA UE4 needs y-flip (done at spawn time)
        "sensor_extrinsics": {
            "RGB-CAM_FRONT":       {"xyz": [ 1.5,  0.0, 2.1], "rpy_deg": [0,   0,   0]},
            "RGB-CAM_FRONT_LEFT":  {"xyz": [ 1.0, -0.8, 1.9], "rpy_deg": [0,   0, -55]},
            "RGB-CAM_FRONT_RIGHT": {"xyz": [ 1.0,  0.8, 1.9], "rpy_deg": [0,   0,  55]},
            "RGB-CAM_BACK":        {"xyz": [-1.8,  0.0, 1.9], "rpy_deg": [0,   0, 180]},
            "RGB-CAM_BACK_LEFT":   {"xyz": [-1.0, -0.8, 1.9], "rpy_deg": [0,   0,-125]},
            "RGB-CAM_BACK_RIGHT":  {"xyz": [-1.0,  0.8, 1.9], "rpy_deg": [0,   0, 125]},
            "LIDAR":               {"xyz": [ 0.0,  0.0, 2.5], "rpy_deg": [0,   0,   0]},
            "SEG-LIDAR":           {"xyz": [ 0.0,  0.0, 2.5], "rpy_deg": [0,   0,   0]},
            "RAD_FRONT":           {"xyz": [ 2.1,  0.0, 0.8], "rpy_deg": [0,   0,   0]},
            "RAD_LEFT":            {"xyz": [ 0.0, -0.8, 0.8], "rpy_deg": [0,   0, -90]},
            "RAD_RIGHT":           {"xyz": [ 0.0,  0.8, 0.8], "rpy_deg": [0,   0,  90]},
            "RAD_BACK":            {"xyz": [-2.1,  0.0, 0.8], "rpy_deg": [0,   0, 180]},
            "GNSS":                {"xyz": [ 0.0,  0.0, 0.0], "rpy_deg": [0,   0,   0]},
            "IMU":                 {"xyz": [ 0.0,  0.0, 0.0], "rpy_deg": [0,   0,   0]},
        },
    }
