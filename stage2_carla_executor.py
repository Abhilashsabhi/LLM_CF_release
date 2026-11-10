"""
stage2_carla_executor.py
=========================
Synchronous CARLA executor.

Key design points
-----------------
* Physics at 20 Hz (fixed_delta_seconds = 0.05 s)
* Data recorded every 2 ticks → 10 Hz = 0.1 s sampling interval
* Duration: 30 s → 600 physics ticks → 300 recorded frames per scene
* EGO   controlled by EgoIDMController  (IDM + reaction-time lag)
* LEADER controlled by LeaderIDMController (free-flow IDM + scripted phases)
  → Leader drives HUMAN-LIKE: speed noise, own desired speed, own style
* Pedestrian spawned for `pedestrian_crossing` scenario
* All 9 injection types handled
* Sensors mirror SimBEV exactly (6 RGB, LiDAR, SemanticLiDAR, 4 Radar, GNSS, IMU)
* Data saved in SimBEV-compatible sweeps/ structure
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import queue
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yaml

try:
    import carla
except ImportError:
    raise ImportError(
        "CARLA Python API not found. "
        "Install: pip install /path/to/carla/PythonAPI/carla/dist/*.whl"
    )

from stage3_scenario_manager import SceneSpec, InjectionType
from stage1_llm_generator import EMERGENCY_DECEL_MPS2
from utils.idm_controller import (
    EgoIDMController, LeaderIDMController, IDMConfig, CarFollowState,
)
from utils.dataset_utils import (
    DatasetLayout, save_gt_kinematics, sensor_metadata, GT_FIELD_NAMES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Sensor specs (matching SimBEV SensorCoordTab)
# CARLA coordinate system: X=forward, Y=right, Z=up
# FLU→CARLA: y-axis is flipped   (FLU y=-right → CARLA y=+right)
# ---------------------------------------------------------------------------

# Camera resolution: 1280x720.
#
# The first release rendered six cameras at 1920x1080. That is the dominant
# cost of the corpus in both GPU time and storage -- 82 % of 3.57 TB -- and the
# original collection machine could not sustain it: the GPU halted (Xid 62)
# every few scenes under that load. 720p halves the pixel count per camera
# (2.07 Mpx -> 0.92 Mpx, a 56 % reduction), which cuts render load and brings
# the corpus to roughly 1.7 TB.
#
# Nothing downstream needs more. Every perception experiment in the paper
# resizes the front camera to 320x180 before use -- the supervised spacing
# reference, the multimodal benchmark and the vision-language adaptation all
# do -- so 720p is still far above the resolution any model actually sees.
# Vertical field of view is unchanged: the aspect ratio stays 16:9 and the
# horizontal FOV stays 70 degrees, so the imagery frames the scene identically
# and only the sampling density changes.
CAMERA_CFGS = [
    # (name,               x,     y,    z,  pitch, yaw,   roll, w,    h,   fov)
    ("RGB-CAM_FRONT",       1.5,   0.0,  2.1,   0,    0,    0, 1280, 720, 70),
    ("RGB-CAM_FRONT_LEFT",  1.0,   0.8,  1.9,   0,  -55,   0, 1280, 720, 70),
    ("RGB-CAM_FRONT_RIGHT", 1.0,  -0.8,  1.9,   0,   55,   0, 1280, 720, 70),
    ("RGB-CAM_BACK",       -1.8,   0.0,  1.9,   0,  180,   0, 1280, 720, 70),
    ("RGB-CAM_BACK_LEFT",  -1.0,   0.8,  1.9,   0, -125,   0, 1280, 720, 70),
    ("RGB-CAM_BACK_RIGHT", -1.0,  -0.8,  1.9,   0,  125,   0, 1280, 720, 70),
]

LIDAR_CFG = dict(
    name="LIDAR", x=0.0, y=0.0, z=2.5,
    channels=64, range=100.0, points_per_second=1_280_000,
    rotation_frequency=20.0, upper_fov=15.0, lower_fov=-25.0,
)
SEG_LIDAR_CFG = {**LIDAR_CFG, "name": "SEG-LIDAR"}

RADAR_CFGS = [
    # (name,       x,    y,    z,   yaw,  h_fov, v_fov, range, pps)
    ("RAD_FRONT",  2.1,  0.0,  0.8,   0,   30,   10,  100, 1500),
    ("RAD_LEFT",   0.0,  0.8,  0.8,  90,   30,   10,   70, 1500),
    ("RAD_RIGHT",  0.0, -0.8,  0.8, -90,   30,   10,   70, 1500),
    ("RAD_BACK",  -2.1,  0.0,  0.8, 180,   30,   10,   80, 1500),
]

# Discarded spin-up before recording starts. The vehicles are launched with
# set_target_velocity, but the leader (no interaction term) reaches its
# cruise speed faster than the follower, so the gap opens well beyond the
# IDM equilibrium before the loop settles. 1.5 s was not enough to shed that
# transient and it biased the recorded spacing upward by ~15 m; 5 s lets the
# pair converge so the 30 s window is genuine steady car-following.
WARMUP_TICKS    = 400   # 20.0 s at 20 Hz
#
# 20 s, not 5 s. The car-following loop needs that long to settle: a follower
# that starts even slightly off its IDM equilibrium overshoots and the
# oscillation decays over ~25 s. At 5 s the recorded window was dominated by
# that transient, which is what held highway spacing ~30 m against a 22 m
# equilibrium. Warmup is cheap because the camera and LiDAR sensors are not
# attached until it finishes (see run_scene) — only physics runs here.
SPAWN_ATTEMPTS  = 25    # distinct spawn points to try before giving up
SPAWN_RANK_CANDIDATES = 60   # spawn points scored for junction-free run
JUNCTION_PROBE_STEP_M = 10.0   # coarsening this to 20 m degraded spawn
                               # quality (highway W1 8.7 -> 10.5); the
                               # early-exit below is the safe speedup
JUNCTION_PROBE_LIMIT_M= 900.0
MIN_STRAIGHT_RUN_M    = 250.0
# Shared-route following
ROUTE_STEP_M          = 2.0    # polyline resolution
ROUTE_SEARCH_WINDOW   = 60     # points scanned forward per tick
ROUTE_LOST_M          = 8.0    # beyond this the vehicle is off-route
ROUTE_LENGTH_MARGIN   = 1.4    # route length vs nominal scene distance
SPAWN_Z_OFFSET_M= 0.3   # lift above the road surface to avoid ground clip
# Vehicle pools rather than fixed models.
#
# The first release fixed the ego to a Ford Mustang and the leader to a Tesla
# Model 3 in every scene. A perception model could then learn one silhouette's
# apparent size instead of monocular ranging, and the scene-level split could
# not detect it: measured on the release, spacing error was 2.7x higher on the
# only other leader that ever appears. Sampling per scene from disjoint pools
# removes the shortcut and makes a held-out-vehicle split possible, which is
# what lets a user show that a model generalises across vehicle geometry.
#
# The pools are disjoint by construction so that a train/test split on vehicle
# identity is available to users, and each spans a range of body sizes -- the
# cue a ranging model is tempted to memorise.
EGO_BLUEPRINT   = "vehicle.ford.mustang"     # retained as the default/fallback
LEADER_BLUEPRINT= "vehicle.tesla.model3"     # retained as the default/fallback

EGO_BLUEPRINT_POOL = [
    "vehicle.ford.mustang",
    "vehicle.audi.tt",
    "vehicle.bmw.grandtourer",
    "vehicle.chevrolet.impala",
    "vehicle.nissan.patrol",
]
# Disjoint from the ego pool, and spanning small hatchback to van so that
# apparent width and height are not a reliable proxy for distance.
LEADER_BLUEPRINT_POOL = [
    "vehicle.tesla.model3",
    "vehicle.citroen.c3",
    "vehicle.seat.leon",
    "vehicle.toyota.prius",
    "vehicle.volkswagen.t2",
    "vehicle.mercedes.coupe",
    "vehicle.mini.cooper_s",
    "vehicle.nissan.micra",
    "vehicle.jeep.wrangler_rubicon",
]
PED_BLUEPRINTS  = [      # realistic pedestrian blueprints
    "walker.pedestrian.0001",
    "walker.pedestrian.0004",
    "walker.pedestrian.0010",
    "walker.pedestrian.0027",
]
VEHICLE_LENGTH  = 4.8   # metres (approx)

# Injections with no scripted leader phase — CARLA physics realise them,
# so `injection_active` cannot be read off the leader controller.
PASSIVE_INJECTIONS = {InjectionType.GRADE_CHANGE}

# Cut-in vehicle
CUT_IN_BLUEPRINT      = "vehicle.audi.a2"
CUT_IN_SPEED_RATIO    = 0.80   # of ego speed — slower, so the gap closes
CUT_IN_MIN_AHEAD_M    = 8.0    # minimum bumper-to-bumper gap at spawn
CUT_IN_HALF_LENGTH_M  = 2.0    # audi.a2 half-length (approx)
CUT_IN_LOOKAHEAD_M    = 6.0    # pure-pursuit lookahead during the merge
CUT_IN_MAX_STEER_DEG  = 25.0   # steering angle mapped to full lock
CUT_IN_MERGE_TIME_S   = 2.0    # approx time spent crossing the lane line
CUT_IN_LAG_SAFETY     = 1.6    # first-order braking lag allowance

# Lane keeping (ego + leader)
LANE_KEEP_LOOKAHEAD_S     = 0.6   # seconds of travel ahead to aim at
LANE_KEEP_MIN_LOOKAHEAD_M = 5.0
LANE_KEEP_MAX_STEER_DEG   = 35.0
# Lateral tolerance for "in the ego's lane". Half a lane (~1.75 m) plus a
# vehicle half-width, so an encroaching car is seen while still merging
# rather than popping into existence once fully across.
SAME_LANE_LATERAL_M   = 2.6


# ---------------------------------------------------------------------------
# Sensor attachment helpers
# ---------------------------------------------------------------------------

def _attach_rgb(world, ego, cfg):
    n, x, y, z, pitch, yaw, roll, w, h, fov = cfg
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov",          str(fov))
    t  = carla.Transform(carla.Location(x=x, y=y, z=z),
                         carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
    s  = world.spawn_actor(bp, t, attach_to=ego)
    q: queue.Queue = queue.Queue()
    s.listen(q.put)
    return s, q


def _attach_lidar(world, ego, cfg, semantic=False):
    bid = "sensor.lidar.ray_cast_semantic" if semantic else "sensor.lidar.ray_cast"
    bp  = world.get_blueprint_library().find(bid)
    for attr in ("channels", "range", "points_per_second",
                 "rotation_frequency", "upper_fov", "lower_fov"):
        bp.set_attribute(attr, str(cfg[attr]))
    t = carla.Transform(carla.Location(x=cfg["x"], y=cfg["y"], z=cfg["z"]))
    s = world.spawn_actor(bp, t, attach_to=ego)
    q: queue.Queue = queue.Queue()
    s.listen(q.put)
    return s, q


def _radar_mount(ego, x: float, y: float, z: float):
    """Place a radar just OUTSIDE the ego bounding box.

    The offsets in RADAR_CFGS were tuned to a Ford Mustang, the single ego of
    the first release. With the ego sampled from a pool they no longer hold:
    on a taller body the sensor sits inside the mesh and every detection comes
    back at the body surface. Measured on a smoke run, front-radar depth was a
    constant \SI{0.14}{\metre} for a Nissan Patrol and \SI{0.03}{\metre} for
    a BMW Grand Tourer, against a correct \SIrange{5}{100}{\metre} for the
    Mustang -- the sensor was ranging the car it was bolted to.

    Mounting relative to the actual bounding box makes the geometry correct
    for every vehicle: a longitudinal sensor is pushed clear of the bumper, a
    lateral one clear of the flank, and the height is lifted above the body so
    the beam leaves the vehicle. The clearance is small so the sensor still
    represents a bumper-mounted radar.
    """
    ext = ego.bounding_box.extent
    clear = 0.25
    if abs(x) > 0.5:                       # longitudinal (front or rear)
        x = (ext.x + clear) * (1.0 if x > 0 else -1.0)
    if abs(y) > 0.5:                       # lateral (left or right)
        y = (ext.y + clear) * (1.0 if y > 0 else -1.0)
    # Lift above the body so the beam is not occluded by the vehicle itself.
    z = max(z, 2.0 * ext.z * 0.5 + 0.35)
    return x, y, z


def _attach_radar(world, ego, cfg):
    n, x, y, z, yaw, h_fov, v_fov, rng, pps = cfg
    bp = world.get_blueprint_library().find("sensor.other.radar")
    bp.set_attribute("horizontal_fov",   str(h_fov))
    bp.set_attribute("vertical_fov",     str(v_fov))
    bp.set_attribute("range",            str(rng))
    bp.set_attribute("points_per_second",str(pps))
    x, y, z = _radar_mount(ego, x, y, z)
    t = carla.Transform(carla.Location(x=x, y=y, z=z),
                        carla.Rotation(yaw=yaw))
    s = world.spawn_actor(bp, t, attach_to=ego)
    q: queue.Queue = queue.Queue()
    s.listen(q.put)
    return s, q


def _attach_collision(world, ego):
    """Event-based collision sensor — fires only on impact, never per tick.

    Kept out of the per-frame sensor queues on purpose: those are drained
    every tick, which would throw collision events away.
    """
    bp = world.get_blueprint_library().find("sensor.other.collision")
    s  = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    state = {"hit": False, "n": 0, "last_frame": -1, "with": []}

    def _on_collision(event):
        state["hit"] = True
        state["n"]  += 1
        state["last_frame"] = event.frame
        # Record WHAT was hit. A rear-end into the lead vehicle is a
        # car-following failure; scraping a guard rail is a lateral-control
        # failure. They need different fixes, so the audit must tell them
        # apart rather than reporting one collision count.
        try:
            other = event.other_actor
            tag = getattr(other, "type_id", "unknown") or "unknown"
            role = other.attributes.get("role_name", "") if other else ""
        except Exception:
            tag, role = "unknown", ""
        if len(state["with"]) < 20:
            state["with"].append(f"{tag}|{role}")

    s.listen(_on_collision)
    return s, state


def _attach_gnss(world, ego):
    bp = world.get_blueprint_library().find("sensor.other.gnss")
    s  = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    q: queue.Queue = queue.Queue()
    s.listen(q.put)
    return s, q


def _attach_imu(world, ego):
    bp = world.get_blueprint_library().find("sensor.other.imu")
    s  = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    q: queue.Queue = queue.Queue()
    s.listen(q.put)
    return s, q


# ---------------------------------------------------------------------------
# Data saving helpers
# ---------------------------------------------------------------------------

def _save_rgb(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(path))


def _save_lidar(data, path: Path, semantic: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if semantic:
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 6)
    else:
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    np.save(str(path), pts)


# One warning per process is enough; a per-frame warning would drown the log.
_RADAR_WARNED = [False]

# Physically attainable radial velocity for this scenario family. Nothing in a
# car-following corpus closes faster than this, so a magnitude above it means
# the value is not a velocity.
_RADAR_MAX_ABS_VELOCITY = 120.0


def _save_radar(data, path: Path):
    """Serialise one radar sweep as (n, 4) float32: depth, altitude, azimuth,
    velocity.

    The velocity is read from the raw measurement buffer rather than through
    the per-detection Python accessor. In the build used for the first release
    (server c93d684fa3) the accessor returned values in the thousands with no
    recoverable relation to true relative speed -- correlation \num{0.017}
    against ground truth -- while depth, azimuth and altitude from the same
    objects were sound. The raw buffer is a packed array of
    ``RadarDetection`` structs laid out as (velocity, azimuth, altitude,
    depth), which is the same access pattern the LiDAR path already uses and
    does not depend on the accessor.

    Values are validated before writing. A physically impossible velocity is
    stored as NaN rather than passed through, so that a downstream consumer
    sees missing data instead of plausible-looking noise, and a warning is
    emitted once per process.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        vel, azi, alt, dep = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    except (ValueError, AttributeError):
        # Fall back to the accessor if the buffer layout is not as expected.
        vel = np.array([d.velocity for d in data], dtype=np.float32)
        azi = np.array([d.azimuth for d in data], dtype=np.float32)
        alt = np.array([d.altitude for d in data], dtype=np.float32)
        dep = np.array([d.depth for d in data], dtype=np.float32)

    bad = ~np.isfinite(vel) | (np.abs(vel) > _RADAR_MAX_ABS_VELOCITY)
    if bad.any():
        vel = vel.copy()
        vel[bad] = np.nan
        if not _RADAR_WARNED[0]:
            _RADAR_WARNED[0] = True
            logger.warning(
                "radar velocity out of physical range for %d/%d detections "
                "(first seen at %s); those entries are written as NaN. If this "
                "fires for most frames the simulator build is not reporting "
                "Doppler correctly and the channel must not be released as "
                "valid.", int(bad.sum()), int(bad.size), path.name)

    pts = np.stack([dep, alt, azi, vel], axis=1).astype(np.float32)
    if pts.shape[0] == 0:
        pts = pts.reshape(0, 4)
    np.save(str(path), pts)


def _save_gnss(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.array([data.latitude, data.longitude, data.altitude]))


def _save_imu(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z,
                    data.gyroscope.x,     data.gyroscope.y,     data.gyroscope.z,
                    data.compass])
    np.save(str(path), arr)


# ---------------------------------------------------------------------------
# CARLA Executor
# ---------------------------------------------------------------------------

class CarlaExecutor:
    """
    Executes one SceneSpec at a time.

    Physics : 20 Hz
    Recording: every 2nd tick → 10 Hz (0.1 s)
    Duration : 30 s → 600 ticks → 300 recorded frames
    """

    def __init__(self, dataset_root: str,
                 host: str = "localhost", port: int = 2000,
                 timeout: float = 120.0, record_log: bool = True,
                 collect_sensors: bool = True):
        # NOTE: keep this generous. A load_world() right after a heavy scene
        # can take far longer than the wall-clock map-load time, and a client
        # time-out surfaces as an uncatchable C++ terminate() that kills the
        # whole collection run.
        self.layout     = DatasetLayout(dataset_root)
        self.layout.initialise()
        self.record_log = record_log
        # Kinematics-only mode: skips the 14 sensors entirely. One scene drops
        # from ~2.6 GB to a few hundred kB, which makes it practical to sweep
        # the whole scene list and check the kinematic distributions BEFORE
        # committing terabytes of sensor data.
        self.collect_sensors = collect_sensors
        self._rng       = random.Random(42)

        logger.info("Connecting to CARLA %s:%d ...", host, port)
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        logger.info("Connected — CARLA %s", self.client.get_server_version())
        self._actors: List = []
        self._recover_world()

    def _recover_world(self) -> None:
        """Undo damage left by a previously crashed run.

        A scene killed by a signal never reaches its `finally`, so the world
        stays in SYNCHRONOUS mode with no client ticking it and stale actors
        still spawned. Every later client then blocks forever — one crashed
        scene silently froze an entire collection for an hour. Forcing the
        world back to asynchronous and clearing leftovers at connect time
        makes each run independent of how the previous one ended.
        """
        try:
            world = self.client.get_world()
            s = world.get_settings()
            if s.synchronous_mode:
                logger.warning("World was left in synchronous mode by a "
                               "previous run — resetting to asynchronous.")
                s.synchronous_mode    = False
                s.fixed_delta_seconds = None
                world.apply_settings(s)
            self._destroy_stale_actors(world)
        except Exception as exc:
            logger.warning("World recovery skipped: %s", exc)

    # ── public ─────────────────────────────────────────────────────

    def run_scene(self, spec: SceneSpec) -> List[Dict]:
        """
        Execute one 30-second scene. Returns list of per-frame metadata dicts.
        """
        self._save_config(spec)
        logger.info("Scene %s | %s | %s | map=%s | weather=%s",
                    spec.scene_id, spec.driving_style,
                    spec.scenario_category, spec.carla_map,
                    list(spec.weather.keys())[0] if spec.weather else "?")

        world  = self._load_map(spec.carla_map)
        self._set_weather(world, spec.weather)
        self._set_sync(world, spec.physics_fps)
        smap   = world.get_map()   # cached once — get_map() is expensive

        # Everything below runs inside the try, so that a failure during
        # SETUP (a spawn collision, a sensor that will not attach) still hits
        # the cleanup path. Previously setup sat outside it: a failed leader
        # spawn leaked the already-spawned ego and left the world in
        # synchronous mode, and the next scene's load_world() then segfaulted
        # the client. That was the cause of the "intermittent" mid-run deaths.
        ego = leader = sensors = None
        collision_sensor = None
        collision_state  = {"hit": False, "n": 0, "last_frame": -1}
        ped_actor    = None
        ped_spawned  = False
        cut_in_actor = None
        cut_in_speed = 0.0
        cut_in_ctrl  = None
        recorder_on  = False
        frame_records: List[Dict] = []
        frame_idx = 0
        injection_fired  = False
        injection_active = False

        try:
            ego, leader, route = self._spawn_vehicles(world, smap, spec)
            ego_cur, ldr_cur = {"idx": 0}, {"idx": 0}

            # IDM controllers
            ego_ctrl    = EgoIDMController(
                IDMConfig.from_dict(spec.idm_params),
                v_initial=spec.initial_ego_speed_mps,
            )
            leader_ctrl = LeaderIDMController(
                IDMConfig.from_dict(spec.leader_idm_params),
                rng_seed=hash(spec.scene_id) % (2**31),
                v_initial=spec.initial_leader_speed_mps,
            )

            # Heavy sensors are attached AFTER warmup (see the loop below):
            # rendering 6x1080p cameras and 2 LiDARs for 20 s of settling
            # would cost more than the rest of the scene.
            sensors = {}

            # Real bumper geometry — the 4.8 m constant underestimates the
            # actual Mustang + Model3 half-extents, which hid genuine contact.
            half_extent_sum = (ego.bounding_box.extent.x
                               + leader.bounding_box.extent.x)

            collision_sensor, collision_state = _attach_collision(world, ego)
            self._actors.append(collision_sensor)

            # CARLA recorder
            if self.record_log:
                self.client.start_recorder(
                    str(self.layout.log_path(spec.scene_id)), True)
                recorder_on = True

            RECORD_EVERY = spec.record_every_n_ticks   # =2 for 10 Hz

            total_ticks = spec.n_physics_ticks + WARMUP_TICKS
            for tick_i in range(total_ticks):
                world_frame = world.tick()
                sim_t = (tick_i - WARMUP_TICKS) / spec.physics_fps

                # ── Leader control ─────────────────────────────────────
                ldr_v  = leader.get_velocity()
                ldr_spd = math.sqrt(ldr_v.x**2 + ldr_v.y**2)

                # Check injection trigger
                if (tick_i >= WARMUP_TICKS
                        and not injection_fired
                        and sim_t >= spec.injection.trigger_time_s
                        and spec.injection.injection_type != InjectionType.NONE):
                    injection_fired = True
                    if spec.injection.injection_type == InjectionType.CUT_IN:
                        cut_in_actor, cut_in_speed = \
                            self._spawn_cut_in_vehicle(world, smap, ego, spec,
                                                       route, ego_cur)
                        if cut_in_actor is None:
                            # No usable adjacent lane — fall back to the old
                            # leader-brake proxy so the scene still yields a
                            # safety-critical event rather than nothing.
                            logger.warning(
                                "Scene %s: cut-in vehicle unavailable, "
                                "falling back to leader brake.", spec.scene_id)
                            self._fire_injection(spec, leader_ctrl, world,
                                                 leader, spec.injection)
                    else:
                        self._fire_injection(spec, leader_ctrl, world,
                                             leader, spec.injection)
                    logger.debug("Injection fired at t=%.1f s: %s",
                                 sim_t, spec.injection.injection_type)

                # Apply leader control
                ldr_ctrl = leader_ctrl.step(ldr_spd, 1.0 / spec.physics_fps)
                leader.apply_control(carla.VehicleControl(
                    throttle = ldr_ctrl["throttle"],
                    brake    = ldr_ctrl["brake"],
                    steer    = self._route_steer(route, ldr_cur, leader,
                                                ldr_spd, smap),
                ))
                # Derive from the leader's actual control phase. The old
                # `trigger_time + brake_duration_s` window was always empty
                # for cut_in / cut_out / speed_limit_change / grade_change,
                # which leave brake_duration_s at its 0.0 default.
                if not injection_fired:
                    injection_active = False
                elif spec.injection.injection_type in PASSIVE_INJECTIONS:
                    # No leader phase to inspect (CARLA physics do the work);
                    # the road geometry stays changed for the rest of the scene.
                    injection_active = True
                elif cut_in_actor is not None:
                    # A real cut-in car is on the road; the disturbance lasts
                    # as long as it does.
                    injection_active = True
                else:
                    injection_active = leader_ctrl.in_scripted_phase

                # ── Pedestrian spawning ────────────────────────────────
                ped_present = False
                if (spec.pedestrian.enabled
                        and tick_i >= WARMUP_TICKS
                        and not ped_spawned
                        and sim_t >= spec.pedestrian.trigger_time_s):
                    ped_actor  = self._spawn_pedestrian(world, leader, spec)
                    ped_spawned= True
                if ped_actor is not None:
                    try:
                        ped_present = ped_actor.is_alive
                    except Exception:
                        ped_present = False

                # ── Cut-in vehicle control ─────────────────────────────
                if cut_in_actor is not None:
                    try:
                        cut_in_ctrl = self._drive_cut_in(
                            smap, cut_in_actor, ego, cut_in_speed,
                            1.0 / spec.physics_fps)
                    except Exception as exc:
                        logger.warning("Cut-in control failed: %s", exc)
                        cut_in_actor = None

                # ── Ego control (IDM) ──────────────────────────────────
                # Whichever vehicle is actually directly ahead in the ego's
                # lane — after a cut-in that is the merging car, not the
                # original leader.
                active_leader = self._select_active_leader(
                    ego, (leader, cut_in_actor)) or leader
                is_cut_in_lead = (cut_in_actor is not None
                                  and active_leader is cut_in_actor)
                state = self._observe(ego, active_leader, half_extent_sum)
                ego_ctrl_out = ego_ctrl.step(CarFollowState(
                    ego_speed_mps    = state["ego_speed_mps"],
                    leader_speed_mps = state["leader_speed_mps"],
                    spacing_m        = state["spacing_m"],
                    dt_s             = 1.0 / spec.physics_fps,
                ))
                ego.apply_control(carla.VehicleControl(
                    throttle = ego_ctrl_out["throttle"],
                    brake    = ego_ctrl_out["brake"],
                    steer    = self._route_steer(
                        route, ego_cur, ego, state["ego_speed_mps"], smap),
                ))

                # ── Skip warmup ────────────────────────────────────────
                if tick_i < WARMUP_TICKS:
                    continue
                if tick_i == WARMUP_TICKS and not sensors:
                    # Settled: now pay for the sensors.
                    sensors = self._attach_sensors(world, ego)
                    if sensors:
                        world.tick()   # let them produce their first frame

                # ── Record every N ticks ───────────────────────────────
                physics_tick = tick_i - WARMUP_TICKS
                if physics_tick % RECORD_EVERY != 0:
                    self._drain_sensors(sensors)
                    continue

                # Collect sensors
                sensor_paths = self._collect_sensors(
                    sensors, spec.scene_id, frame_idx, world_frame
                )

                # Kinematic GT — describes the vehicle the ego is actually
                # following, so a cut-in car becomes the recorded "leader"
                # the moment it occupies the ego's lane.
                ego_loc  = ego.get_location()
                ldr_loc  = active_leader.get_location()
                ego_rot  = ego.get_transform().rotation
                ldr_rot  = active_leader.get_transform().rotation
                rel_spd  = state["ego_speed_mps"] - state["leader_speed_mps"]
                ttc      = (state["spacing_m"] / rel_spd
                            if rel_spd > 0.05 else 9999.0)
                ldr_acc  = ((cut_in_ctrl or ldr_ctrl)["raw_acceleration_mps2"]
                            if is_cut_in_lead
                            else ldr_ctrl["raw_acceleration_mps2"])

                gt_rec = {
                    "timestamp_s":              sim_t,
                    "spacing_m":                state["spacing_m"],
                    "ego_speed_mps":            state["ego_speed_mps"],
                    "leader_speed_mps":         state["leader_speed_mps"],
                    "relative_speed_mps":       rel_spd,
                    "ego_acceleration_mps2":    ego_ctrl_out["raw_acceleration_mps2"],
                    "leader_acceleration_mps2": ldr_acc,
                    "ttc_s":                    ttc,
                    "ego_x":    ego_loc.x,  "ego_y":    ego_loc.y,  "ego_z":    ego_loc.z,
                    "leader_x": ldr_loc.x,  "leader_y": ldr_loc.y,  "leader_z": ldr_loc.z,
                    "ego_yaw_rad":    math.radians(ego_rot.yaw),
                    "leader_yaw_rad": math.radians(ldr_rot.yaw),
                    "injection_active":   injection_active,
                    "driving_style":      spec.driving_style,
                    "scenario_category":  spec.scenario_category,
                    "pedestrian_present": ped_present,
                    # Sticky: once the ego has been hit, the scene is
                    # contaminated for every frame that follows.
                    "collision":          collision_state["hit"],
                }
                save_gt_kinematics(
                    self.layout.gt_det_path(spec.scene_id, frame_idx), gt_rec
                )

                frame_records.append({
                    "frame_idx":    frame_idx,
                    "timestamp_s":  sim_t,
                    "sensor_paths": sensor_paths,
                    "kinematics":   gt_rec,
                })
                frame_idx += 1

        finally:
            if recorder_on:
                try: self.client.stop_recorder()
                except Exception: pass
            self._cleanup(world, sensors)

        if collision_state.get("hit"):
            logger.warning("Scene %s COLLISION x%d with: %s",
                           spec.scene_id, collision_state["n"],
                           ", ".join(sorted(set(collision_state.get("with", [])))))
        logger.info("Scene %s done — %d frames recorded.", spec.scene_id, len(frame_records))
        return frame_records

    # ── CARLA setup ────────────────────────────────────────────────

    def _load_map(self, map_name: str):
        """Load a map, reusing the current world when it already matches.

        `load_world()` is the single most expensive and least stable call in
        the pipeline (each one costs seconds and, against a mismatched
        client, can segfault). Scenes are grouped by map in Stage 2, so this
        check collapses one load per scene into one load per map — roughly
        1350 loads down to 7 for a full run.
        """
        try:
            current = self.client.get_world().get_map().name.split("/")[-1]
        except Exception:
            current = None
        if current == map_name:
            logger.debug("Map %s already loaded — reusing world.", map_name)
            world = self.client.get_world()
            self._destroy_stale_actors(world)
            return world

        world = self.client.load_world(map_name)
        time.sleep(2.5)
        return world

    @staticmethod
    def _destroy_stale_actors(world) -> None:
        """Remove vehicles/sensors/walkers left by any previous scene."""
        try:
            actors = world.get_actors()
        except Exception:
            return
        for pattern in ("*vehicle*", "*sensor*", "*walker*"):
            for a in actors.filter(pattern):
                try:
                    if hasattr(a, "stop"):
                        a.stop()
                except Exception:
                    pass
                try:
                    a.destroy()
                except Exception:
                    pass

    @staticmethod
    def _set_weather(world, wd: Dict) -> None:
        w = carla.WeatherParameters()
        for attr, val in wd.items():
            if hasattr(w, attr): setattr(w, attr, val)
        world.set_weather(w)

    @staticmethod
    def _set_sync(world, fps: int) -> None:
        s = world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = 1.0 / fps
        world.apply_settings(s)

    def _pick_blueprint(self, bpl, pool, fallback):
        """One blueprint from the pool, skipping any absent in this build.

        CARLA builds differ in which vehicle assets ship, so a pool entry that
        does not resolve must not abort a scene; it is dropped and the choice
        made among what exists, falling back to the historical default only if
        the whole pool is missing.
        """
        available = []
        for name in pool:
            try:
                bpl.find(name)
                available.append(name)
            except (IndexError, RuntimeError):
                continue
        if not available:
            logger.warning("no pool blueprint available; using %s", fallback)
            return fallback
        return available[self._rng.randrange(len(available))]

    def _spawn_vehicles(self, world, smap, spec: SceneSpec):
        bpl = world.get_blueprint_library()
        pts = smap.get_spawn_points()
        if not pts:
            raise RuntimeError("No spawn points on this map.")
        self._rng.shuffle(pts)

        # Sample per scene from the pools, using the scene's own RNG so the
        # choice is reproducible from the seed and recorded in scene_info.
        ego_name = self._pick_blueprint(bpl, EGO_BLUEPRINT_POOL, EGO_BLUEPRINT)
        ldr_name = self._pick_blueprint(bpl, LEADER_BLUEPRINT_POOL,
                                        LEADER_BLUEPRINT)
        self._ego_blueprint, self._leader_blueprint = ego_name, ldr_name
        # Write back onto the spec so it reaches scene_info via to_dict().
        try:
            spec.ego_blueprint, spec.leader_blueprint = ego_name, ldr_name
        except AttributeError:
            pass
        ego_bp = bpl.find(ego_name)
        ego_bp.set_attribute("role_name", "hero")
        ldr_bp = bpl.find(ldr_name)
        ldr_bp.set_attribute("role_name", "leader")
        offset = spec.initial_spacing_m + VEHICLE_LENGTH

        # Rank spawn points by how much junction-free road lies ahead. The
        # scene needs roughly v * duration metres of continuous lane; ranking
        # (rather than requiring it outright) keeps this working on compact
        # urban maps where no such stretch exists.
        total_s = spec.scene_duration_s + WARMUP_TICKS / spec.physics_fps
        need_m  = max(spec.initial_leader_speed_mps * total_s,
                      MIN_STRAIGHT_RUN_M)
        need_m = min(need_m, JUNCTION_PROBE_LIMIT_M)
        # Scoring every candidate costs thousands of waypoint queries and
        # roughly doubled per-scene time. Stop at the first spawn that
        # already satisfies the requirement — on a motorway map that is
        # usually one of the first few — and only fall back to ranking the
        # rest when none does.
        ranked: List = []
        for sp in pts[:SPAWN_RANK_CANDIDATES]:
            wp = smap.get_waypoint(sp.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
                continue
            run = self._junction_free_distance(wp, need_m)
            ranked.append((run, sp))
            if run >= need_m:
                break
        if ranked:
            ranked.sort(key=lambda t: -t[0])
            pts = [sp for _, sp in ranked]
            logger.debug("Junction-free run at spawn: %.0f m (need %.0f m, "
                         "%d candidates scored)",
                         ranked[0][0], need_m, len(ranked))

        last_exc: Optional[Exception] = None
        for sp in pts[:SPAWN_ATTEMPTS]:
            # Place the LEADER by walking the lane, not by extrapolating the
            # ego's heading in a straight line: on any curve the straight-line
            # position lands off-lane or inside scenery, which is what used to
            # raise "collision at spawn position" and abort the scene.
            wp = smap.get_waypoint(sp.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            nxt = self._next_straight(wp, offset)
            if nxt is None:
                continue

            ego_tf = carla.Transform(wp.transform.location,
                                     wp.transform.rotation)
            ldr_tf = carla.Transform(nxt.transform.location,
                                     nxt.transform.rotation)
            ego_tf.location.z += SPAWN_Z_OFFSET_M
            ldr_tf.location.z += SPAWN_Z_OFFSET_M

            ego = leader = None
            try:
                ego    = world.spawn_actor(ego_bp, ego_tf)
                leader = world.spawn_actor(ldr_bp, ldr_tf)
            except RuntimeError as exc:
                # Partial success must not leak an actor into the next attempt.
                last_exc = exc
                for a in (leader, ego):
                    if a is not None:
                        try: a.destroy()
                        except Exception: pass
                continue

            # One path for both vehicles, long enough for the whole scene.
            total_s    = spec.scene_duration_s + WARMUP_TICKS / spec.physics_fps
            need_route = (spec.initial_leader_speed_mps * total_s
                          * ROUTE_LENGTH_MARGIN + offset + 50.0)
            route = self._build_route(smap, wp, need_route)
            span = sum(route[i].distance(route[i + 1])
                       for i in range(len(route) - 1)) if len(route) > 1 else 0.0
            logger.debug("Route: %d points spanning %.0f m (needed %.0f m)",
                         len(route), span, need_route)
            if span < need_route * 0.9:
                logger.warning("Route only spans %.0f m of %.0f m needed — the "
                               "pair may run out of road late in the scene.",
                               span, need_route)

            self._actors.extend([ego, leader])
            ego.set_autopilot(False)
            leader.set_autopilot(False)

            # Settle physics, then launch each vehicle along ITS OWN heading
            # (they differ on a curve).
            for _ in range(5):
                world.tick()
            e_fwd = ego_tf.get_forward_vector()
            l_fwd = ldr_tf.get_forward_vector()
            ego.set_target_velocity(
                carla.Vector3D(e_fwd.x, e_fwd.y, 0) * spec.initial_ego_speed_mps
            )
            leader.set_target_velocity(
                carla.Vector3D(l_fwd.x, l_fwd.y, 0) * spec.initial_leader_speed_mps
            )
            return ego, leader, route

        raise RuntimeError(
            f"Could not spawn ego+leader after {SPAWN_ATTEMPTS} attempts "
            f"on {spec.carla_map} (gap {offset:.1f} m): {last_exc}"
        )

    # ── Sensors ────────────────────────────────────────────────────

    def _attach_sensors(self, world, ego) -> Dict:
        sensors: Dict = {}
        if not self.collect_sensors:
            logger.info("Sensors disabled — kinematics-only run.")
            return sensors
        for cfg in CAMERA_CFGS:
            s, q = _attach_rgb(world, ego, cfg)
            sensors[cfg[0]] = (s, q); self._actors.append(s)

        s, q = _attach_lidar(world, ego, LIDAR_CFG)
        sensors["LIDAR"] = (s, q); self._actors.append(s)

        s, q = _attach_lidar(world, ego, SEG_LIDAR_CFG, semantic=True)
        sensors["SEG-LIDAR"] = (s, q); self._actors.append(s)

        for cfg in RADAR_CFGS:
            s, q = _attach_radar(world, ego, cfg)
            sensors[cfg[0]] = (s, q); self._actors.append(s)

        s, q = _attach_gnss(world, ego);  sensors["GNSS"] = (s, q); self._actors.append(s)
        s, q = _attach_imu(world, ego);   sensors["IMU"]  = (s, q); self._actors.append(s)
        return sensors

    @staticmethod
    def _drain_sensors(sensors: Dict) -> None:
        """Discard queued sensor data on non-recorded ticks.

        Queues are unbounded, so `listen(q.put)` never blocks the sensor
        callback thread (a bounded queue deadlocks world.tick() as soon as
        it fills). Draining keeps the backlog to at most one frame.
        """
        for _, (_, q) in sensors.items():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _collect_sensors(self, sensors: Dict, scene_id: str,
                          frame_idx: int, target_frame: int = 0) -> Dict[str, str]:
        paths = {}
        timeout = 2.0   # generous timeout per sensor
        for name, (_, q) in sensors.items():
            data = None
            deadline = time.time() + timeout
            # Skip stale frames; keep the one produced by the current tick.
            while time.time() < deadline:
                try:
                    data = q.get(timeout=max(deadline - time.time(), 0.01))
                except queue.Empty:
                    data = None
                    break
                if getattr(data, "frame", target_frame) >= target_frame:
                    break
            if data is None:
                logger.warning("Sensor %s produced no data for frame %d",
                               name, frame_idx)
                continue
            path = self.layout.sweep_path(name, scene_id, frame_idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith("RGB"):
                _save_rgb(data, path)
            elif name == "LIDAR":
                _save_lidar(data, path, False)
            elif name == "SEG-LIDAR":
                _save_lidar(data, path, True)
            elif name.startswith("RAD"):
                _save_radar(data, path)
            elif name == "GNSS":
                _save_gnss(data, path)
            elif name == "IMU":
                _save_imu(data, path)
            paths[name] = str(path.relative_to(self.layout.root))
        return paths

    # ── State observation ──────────────────────────────────────────

    @staticmethod
    def _observe(ego, leader, half_extent_sum: float = VEHICLE_LENGTH
                 ) -> Dict[str, float]:
        """Bumper-to-bumper gap between ego and leader.

        `half_extent_sum` is the sum of both vehicles' half-lengths taken
        from their actual CARLA bounding boxes, so the gap crosses zero at
        real bumper contact. It is deliberately NOT clamped at zero —
        negative values mean the bodies overlap, which is what the Stage 4
        collision audit looks for.
        """
        ev = ego.get_velocity();   es = math.sqrt(ev.x**2 + ev.y**2)
        lv = leader.get_velocity();ls = math.sqrt(lv.x**2 + lv.y**2)
        dist = ego.get_location().distance(leader.get_location())
        gap  = dist - half_extent_sum
        return {"ego_speed_mps": es, "leader_speed_mps": ls,
                "spacing_m": gap, "relative_speed_mps": es - ls}

    # ── Injection ──────────────────────────────────────────────────

    @staticmethod
    def _fire_injection(spec: SceneSpec,
                        leader_ctrl: LeaderIDMController,
                        world, leader, inj) -> None:
        itype = inj.injection_type
        if itype in (InjectionType.SUDDEN_BRAKE, InjectionType.PEDESTRIAN):
            leader_ctrl.set_phase_brake(
                inj.target_decel_mps2,
                inj.brake_duration_s,
                inj.resume_speed_mps,
            )
        elif itype == InjectionType.GRADUAL_BRAKE:
            leader_ctrl.set_phase_stop_and_go(
                inj.target_decel_mps2,
                inj.brake_duration_s,
                inj.resume_speed_mps,
            )
        elif itype == InjectionType.SPEED_LIMIT_CHANGE:
            leader_ctrl.set_phase_speed_limit(inj.new_speed_limit_mps)
        elif itype == InjectionType.CUT_IN:
            # FALLBACK ONLY. run_scene() spawns a real third vehicle that
            # merges into the ego's lane; this leader-brake approximation is
            # reached only when the ego is somewhere with no usable adjacent
            # lane, so the scene still produces a gap-closure event.
            leader_ctrl.set_phase_brake(
                decel_mps2=3.0, duration_s=2.0,
                resume_speed_mps=spec.initial_leader_speed_mps,
            )
        elif itype == InjectionType.CUT_OUT:
            leader_ctrl.set_phase_speed_limit(
                spec.initial_leader_speed_mps * 0.6
            )
        # GRADE_CHANGE is passive — CARLA physics handle it automatically

    # ── Lateral control ────────────────────────────────────────────

    def _junction_free_distance(self, wp, limit_m: float) -> float:
        """How far the lane runs ahead of `wp` before hitting a junction.

        Junction geometry is where lane keeping breaks down: waypoints inside
        an intersection curve away from the through-route, the follower
        corners hard, scrubs speed (27.8 -> 16 m/s at full throttle in a
        measured run) and the car-following gap runs away. Scenes are
        therefore spawned on stretches that stay junction-free.
        """
        travelled = 0.0
        cur = wp
        while travelled < limit_m:
            nxt = self._next_straight(cur, JUNCTION_PROBE_STEP_M)
            if nxt is None or nxt.is_junction:
                break
            cur = nxt
            travelled += JUNCTION_PROBE_STEP_M
        return travelled

    @staticmethod
    def _next_straight(wp, distance: float):
        """`waypoint.next()` but always continuing straight ahead.

        At a junction `next()` returns one candidate PER EXIT, and taking
        [0] picks an arbitrary branch. That made the ego turn off while the
        leader carried straight on — the follower scrubbed speed in the turn
        and the gap ran away (20 m to 59 m in a nominally steady-follow
        scene). Choosing the smallest heading change keeps both vehicles on
        one continuous path, which is what a car-following scene requires.
        """
        cands = wp.next(distance)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # Prefer continuations that stay OUT of a junction; junction lanes
        # curve away from the through-route and are where the pair used to
        # split up.
        through = [c for c in cands if not c.is_junction]
        cands = through or cands
        if len(cands) == 1:
            return cands[0]
        ref = wp.transform.rotation.yaw

        def heading_change(c) -> float:
            return abs((c.transform.rotation.yaw - ref + 180.0) % 360.0 - 180.0)

        return min(cands, key=heading_change)

    @staticmethod
    def _pursuit_steer(actor, target_loc, max_steer_deg: float) -> float:
        """Pure-pursuit steering toward a target point."""
        tf  = actor.get_transform()
        loc = tf.location
        fwd = tf.get_forward_vector()
        dx, dy = target_loc.x - loc.x, target_loc.y - loc.y
        # Signed angle between the heading and the vector to the target.
        ang = math.atan2(fwd.x * dy - fwd.y * dx,
                         fwd.x * dx + fwd.y * dy)
        return float(np.clip(ang / math.radians(max_steer_deg), -1.0, 1.0))

    def _build_route(self, smap, start_wp, length_m: float) -> List:
        """Pre-compute ONE path that both vehicles follow.

        Deciding the continuation independently at every tick lets the ego
        and the leader make different choices at a junction: a measured
        highway scene had the leader carry straight on while the ego turned
        90 degrees off-route, exploding the gap from 30 m to 149 m and
        hitting scenery. A car-following scene is only meaningful if both
        vehicles are on the same path, so the path is fixed at spawn time
        and shared.
        """
        route: List = []
        cur = start_wp
        travelled = 0.0
        while travelled < length_m:
            nxt = self._next_straight(cur, ROUTE_STEP_M)
            if nxt is None:
                break
            route.append(nxt.transform.location)
            cur = nxt
            travelled += ROUTE_STEP_M
        return route

    @staticmethod
    def _route_index_near(route: List, loc, cursor: Optional[Dict] = None) -> int:
        """Index of the route point closest to `loc`, searching forward."""
        start = (cursor or {}).get("idx", 0)
        best_i, best_d = start, float("inf")
        for j in range(start, min(start + ROUTE_SEARCH_WINDOW, len(route))):
            d = loc.distance(route[j])
            if d < best_d:
                best_d, best_i = d, j
        return best_i

    @staticmethod
    def _route_advance(route: List, start: int, distance_m: float) -> int:
        """Index roughly `distance_m` further along the route, by TRUE arc length.

        `waypoint.next(d)` returns the next waypoint at the lane's own
        sampling resolution, not exactly d metres away — measured spacing was
        ~8.4 m for a requested 2 m. Index arithmetic of the form
        `i + distance/ROUTE_STEP_M` therefore overshoots by whatever that
        ratio happens to be: it placed cut-in vehicles 234 m ahead when 56 m
        was asked for, and aimed the pure-pursuit lookahead ~4x too far.
        """
        acc = 0.0
        j   = start
        while j + 1 < len(route) and acc < distance_m:
            acc += route[j].distance(route[j + 1])
            j   += 1
        return j

    def _route_steer(self, route: List, cursor: Dict, actor,
                     speed_mps: float, smap=None) -> float:
        """Pure-pursuit along the shared route.

        `cursor` holds this vehicle's progress index so the search stays
        local instead of scanning the whole polyline each tick.
        """
        if not route:
            return self._lane_keep_steer(smap, actor, speed_mps)
        loc = actor.get_location()
        i   = cursor.get("idx", 0)
        best_i, best_d = i, float("inf")
        for j in range(i, min(i + ROUTE_SEARCH_WINDOW, len(route))):
            d = loc.distance(route[j])
            if d < best_d:
                best_d, best_i = d, j
        cursor["idx"] = best_i

        # Off the route entirely (pushed off by a collision) — fall back.
        if best_d > ROUTE_LOST_M:
            return self._lane_keep_steer(smap, actor, speed_mps)

        lookahead = max(speed_mps * LANE_KEEP_LOOKAHEAD_S,
                        LANE_KEEP_MIN_LOOKAHEAD_M)
        tgt = route[self._route_advance(route, best_i, lookahead)]
        return self._pursuit_steer(actor, tgt, LANE_KEEP_MAX_STEER_DEG)

    def _lane_keep_steer(self, smap, actor, speed_mps: float) -> float:
        """Steer to follow the actor's own lane.

        Both vehicles used to drive with `steer=0.0` hardcoded, which only
        works on perfectly straight road. Spawn points are picked at random,
        so on any curve the vehicles drifted out of lane and into scenery —
        corrupting the scene long before any injection fired.
        """
        try:
            wp = smap.get_waypoint(actor.get_location(), project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None:
                return 0.0
            lookahead = float(np.clip(speed_mps * LANE_KEEP_LOOKAHEAD_S,
                                      LANE_KEEP_MIN_LOOKAHEAD_M, 25.0))
            nxt = self._next_straight(wp, lookahead)
            if nxt is None:
                return 0.0
            return self._pursuit_steer(actor, nxt.transform.location,
                                       LANE_KEEP_MAX_STEER_DEG)
        except Exception:
            return 0.0

    # ── Cut-in vehicle ─────────────────────────────────────────────

    def _spawn_cut_in_vehicle(self, world, smap, ego, spec: SceneSpec,
                              route: Optional[List] = None,
                              cursor: Optional[Dict] = None):
        """Spawn a third vehicle in an adjacent lane, ahead of the ego.

        Placed so the initial time-to-collision matches the scenario's
        `cut_in_ttc_s`, and travelling slower than the ego so the gap
        genuinely closes once it merges.
        """
        ego_wp = smap.get_waypoint(ego.get_location(), project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            logger.warning("Cut-in: ego is not on a driving lane.")
            return None, 0.0

        # Candidate adjacent lanes, same travel direction only.
        cands = []
        for wp, is_left in ((ego_wp.get_left_lane(),  True),
                            (ego_wp.get_right_lane(), False)):
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                continue
            if wp.lane_id * ego_wp.lane_id < 0:      # opposing traffic
                continue
            cands.append((wp, is_left))
        if not cands:
            logger.warning("Cut-in: no adjacent same-direction lane here.")
            return None, 0.0

        want_left = bool(spec.injection.cut_in_from_left)
        side_wp   = next((w for w, l in cands if l == want_left), cands[0][0])

        ev        = ego.get_velocity()
        ego_speed = math.sqrt(ev.x ** 2 + ev.y ** 2)
        ttc       = max(spec.injection.cut_in_ttc_s, 0.5)
        # Gap that yields the requested TTC at the closing speed below.
        target_speed = max(ego_speed * CUT_IN_SPEED_RATIO, 2.0)
        closing      = max(ego_speed - target_speed, 0.5)
        bumper_gap   = max(closing * ttc, CUT_IN_MIN_AHEAD_M)
        # `next()` walks centre-to-centre along the lane, so the vehicle
        # half-lengths have to be added or the real bumper-to-bumper gap
        # comes out ~4.8 m shorter than the requested TTC implies — which
        # spawns the car practically on top of the ego.
        half_extents = (ego.bounding_box.extent.x
                        + CUT_IN_HALF_LENGTH_M)
        # The merge is not instantaneous: the car keeps closing on the ego
        # for the ~2 s it spends crossing the lane line. Budget for that, so
        # the requested TTC holds when it ARRIVES in the ego's lane rather
        # than when it spawns — otherwise every cut-in ends in a rear-end.
        merge_closing = closing * CUT_IN_MERGE_TIME_S

        # Guarantee the merge is SURVIVABLE. A cut-in is meant to be
        # safety-critical, not unavoidable: at motorway speed behind a short
        # headway the follower simply cannot stop, and 5 of 8 collisions in
        # the verification sweep were exactly this. Require the post-merge
        # gap to clear the Gipps safe distance for the ego's own parameters,
        # with both vehicles braking at the tyre limit.
        cfg      = IDMConfig.from_dict(spec.idm_params)
        gipps    = (ego_speed * cfg.reaction_time_s * CUT_IN_LAG_SAFETY
                    + ego_speed ** 2 / (2.0 * EMERGENCY_DECEL_MPS2)
                    - target_speed ** 2 / (2.0 * EMERGENCY_DECEL_MPS2)
                    + cfg.jam_spacing_m)
        if gipps > bumper_gap:
            logger.debug("Cut-in gap raised %.1f→%.1f m for survivability",
                         bumper_gap, gipps)
            bumper_gap = gipps

        ahead         = bumper_gap + merge_closing + half_extents

        # Place it by arc length along the EGO'S OWN ROUTE, then step
        # sideways. Walking `next()` from the adjacent-lane waypoint traverses
        # whatever lane topology it finds and was landing the car 234 m away
        # when 58 m was requested — so on motorway maps the "cut-in" often
        # never reached the ego at all, and when the topology happened to be
        # short it spawned on top of it instead.
        tf = None
        used_route = False
        if route:
            i0 = self._route_index_near(route, ego.get_location(), cursor)
            j  = self._route_advance(route, i0, ahead)
            rw = smap.get_waypoint(route[j], project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if rw is not None:
                side = (rw.get_left_lane() if want_left else rw.get_right_lane())
                if (side is not None
                        and side.lane_type == carla.LaneType.Driving
                        and side.lane_id * rw.lane_id > 0):
                    tf = side.transform
                    used_route = True
        if tf is None:
            nxt = self._next_straight(side_wp, ahead)
            if nxt is None:
                logger.warning("Cut-in: no waypoint %.1f m ahead.", ahead)
                return None, 0.0
            tf = nxt.transform
        tf = carla.Transform(tf.location, tf.rotation)
        tf.location.z += 0.3

        bp = world.get_blueprint_library().find(CUT_IN_BLUEPRINT)
        bp.set_attribute("role_name", "cut_in")
        try:
            veh = world.spawn_actor(bp, tf)
        except Exception as exc:
            logger.warning("Cut-in spawn failed: %s", exc)
            return None, 0.0

        self._actors.append(veh)
        fwd = tf.get_forward_vector()
        veh.set_target_velocity(
            carla.Vector3D(fwd.x, fwd.y, 0) * target_speed
        )
        # Measure against the spawn transform: an actor's get_location() is
        # not populated until the world ticks, so reading it here reports the
        # origin and makes a correct placement look 200 m off.
        actual = ego.get_location().distance(tf.location) - half_extents
        logger.info("Cut-in spawned: requested %.1f m, actual %.1f m bumper gap "
                    "(%s lane, %.1f m/s, %s placement)",
                    bumper_gap, actual, "left" if want_left else "right",
                    target_speed, "route" if used_route else "FALLBACK")
        return veh, target_speed

    def _drive_cut_in(self, smap, veh, ego, target_speed: float,
                      dt_s: float) -> Dict[str, float]:
        """Steer the cut-in vehicle from its lane into the ego's lane.

        Pure-pursuit toward a point in the EGO's lane level with the cut-in
        vehicle, plus a P controller on speed. Returns a control dict shaped
        like the IDM controllers' output so it can feed the GT record.
        """
        tf     = veh.get_transform()
        loc    = veh.get_location()
        ego_wp = smap.get_waypoint(ego.get_location(), project_to_road=True,
                                   lane_type=carla.LaneType.Driving)

        # Aim at the ego lane, level with the cut-in vehicle plus lookahead.
        lookahead = max(ego.get_location().distance(loc), 0.0) + CUT_IN_LOOKAHEAD_M
        target    = None
        if ego_wp is not None:
            nxt = self._next_straight(ego_wp, lookahead)
            if nxt is not None:
                target = nxt.transform.location

        steer = (self._pursuit_steer(veh, target, CUT_IN_MAX_STEER_DEG)
                 if target is not None else 0.0)

        v   = veh.get_velocity()
        spd = math.sqrt(v.x ** 2 + v.y ** 2)
        acc = float(np.clip((target_speed - spd) * 0.8, -4.0, 2.5))
        if acc >= 0:
            throttle, brake = min(acc / 2.5, 1.0), 0.0
        else:
            throttle, brake = 0.0, min(abs(acc) / 4.0, 1.0)

        veh.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer))
        return {"throttle": throttle, "brake": brake,
                "raw_acceleration_mps2": acc}

    @staticmethod
    def _select_active_leader(ego, candidates):
        """Nearest vehicle directly ahead of the ego in its own lane.

        Without this the ego's IDM keeps tracking the original leader and
        drives straight through a cut-in as if it were not there.
        """
        tf   = ego.get_transform()
        loc  = tf.location
        fwd  = tf.get_forward_vector()
        best, best_long = None, float("inf")
        for veh in candidates:
            if veh is None:
                continue
            try:
                if not veh.is_alive:
                    continue
                vl = veh.get_location()
            except Exception:
                continue
            dx, dy = vl.x - loc.x, vl.y - loc.y
            longitudinal = dx * fwd.x + dy * fwd.y        # along heading
            lateral      = abs(-dx * fwd.y + dy * fwd.x)  # perpendicular
            if longitudinal <= 0.0 or lateral > SAME_LANE_LATERAL_M:
                continue
            if longitudinal < best_long:
                best, best_long = veh, longitudinal
        return best

    def _spawn_pedestrian(self, world, leader, spec: SceneSpec):
        """Spawn a pedestrian ahead of the leader and set it walking."""
        bpl    = world.get_blueprint_library()
        ped_bp = bpl.find(random.choice(PED_BLUEPRINTS))
        ldr_t  = leader.get_transform()
        fwd    = ldr_t.get_forward_vector()
        d      = spec.pedestrian.cross_distance_m
        side   = spec.pedestrian.direction * 2.5   # 2.5 m to one side
        # Lateral offset (perpendicular to forward)
        right  = carla.Vector3D(-fwd.y, fwd.x, 0)  # 90° rotation

        spawn_loc = carla.Location(
            x = ldr_t.location.x + fwd.x * d + right.x * side,
            y = ldr_t.location.y + fwd.y * d + right.y * side,
            z = ldr_t.location.z + 0.5,
        )
        try:
            ped = world.spawn_actor(
                ped_bp,
                carla.Transform(spawn_loc, ldr_t.rotation),
            )
        except Exception as exc:
            logger.warning("Pedestrian spawn failed: %s", exc)
            return None
        self._actors.append(ped)

        # Walk across the road
        ped_ctrl = carla.WalkerControl()
        ped_ctrl.speed     = spec.pedestrian.cross_speed_mps
        ped_ctrl.direction = carla.Vector3D(
            right.x * spec.pedestrian.direction,
            right.y * spec.pedestrian.direction,
            0,
        )
        ped.apply_control(ped_ctrl)
        return ped

    # ── Config / cleanup ───────────────────────────────────────────

    def _save_config(self, spec: SceneSpec) -> None:
        p = self.layout.config_path(spec.scene_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(spec.to_dict(), f, default_flow_style=False, sort_keys=False)

    def _cleanup(self, world, sensors: Optional[Dict] = None) -> None:
        """Tear the scene down. Safe to call on partially-built state.

        Destroys every actor recorded in `self._actors` — vehicles, sensors,
        pedestrian, cut-in car — rather than only the handful passed in, so
        nothing survives into the next scene. Sensors are stopped first;
        listening sensors whose actor is destroyed can crash the client.
        """
        # Stop sensor callbacks before anything is destroyed.
        for actor in self._actors:
            try:
                if actor is not None and hasattr(actor, "stop") and actor.is_alive:
                    actor.stop()
            except Exception:
                pass
        if sensors:
            for _, (actor, _) in sensors.items():
                try: actor.stop()
                except Exception: pass

        for actor in self._actors:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self._actors.clear()

        # Always hand the world back in asynchronous mode. Leaving it
        # synchronous with no ticker makes every later client call hang.
        try:
            s = world.get_settings()
            s.synchronous_mode    = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception:
            pass
        gc.collect()
