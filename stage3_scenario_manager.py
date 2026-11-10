"""
stage3_scenario_manager.py
===========================
Converts LLM-generated ScenarioParams instances into concrete SceneSpec
objects ready for CARLA.

Key changes vs v1
-----------------
* Duration fixed to 30 s, sampling at 0.1 s (10 Hz recording)
* Leader has its own IDM params (human-like, independent driving)
* PedestrianCrossing injection type added
* SceneSpec carries `leader_idm_params` and `pedestrian_config`
* Stochastic variations per instance (received pre-perturbed from Stage 1)
"""

from __future__ import annotations

import logging
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from stage1_llm_generator import (
    ScenarioParams, DRIVING_STYLES, SCENARIO_CATEGORIES,
    LLMParameterGenerator, EMERGENCY_DECEL_MPS2,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weather presets
# ---------------------------------------------------------------------------

WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "clear_noon": {
        "cloudiness": 10.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
        "wind_intensity": 5.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 70.0,
        "fog_density": 0.0, "wetness": 0.0,
    },
    "clear_sunset": {
        "cloudiness": 15.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
        "wind_intensity": 5.0, "sun_azimuth_angle": 270.0, "sun_altitude_angle": 15.0,
        "fog_density": 0.0, "wetness": 0.0,
    },
    "overcast": {
        "cloudiness": 85.0, "precipitation": 0.0, "precipitation_deposits": 5.0,
        "wind_intensity": 15.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 45.0,
        "fog_density": 0.0, "wetness": 10.0,
    },
    "light_rain": {
        "cloudiness": 70.0, "precipitation": 35.0, "precipitation_deposits": 30.0,
        "wind_intensity": 25.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 50.0,
        "fog_density": 2.0, "wetness": 40.0,
    },
    "heavy_rain": {
        "cloudiness": 100.0, "precipitation": 80.0, "precipitation_deposits": 60.0,
        "wind_intensity": 40.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 50.0,
        "fog_density": 5.0, "wetness": 80.0,
    },
    "dense_fog": {
        "cloudiness": 80.0, "precipitation": 0.0, "precipitation_deposits": 10.0,
        "wind_intensity": 2.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 60.0,
        "fog_density": 90.0, "wetness": 0.0,
    },
    "night_clear": {
        "cloudiness": 5.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
        "wind_intensity": 3.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": -80.0,
        "fog_density": 0.0, "wetness": 0.0,
    },
    "night_rain": {
        "cloudiness": 100.0, "precipitation": 60.0, "precipitation_deposits": 50.0,
        "wind_intensity": 30.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": -80.0,
        "fog_density": 3.0, "wetness": 70.0,
    },
}

# Map scenario → eligible weather presets (multiple for diversity)
SCENARIO_WEATHER_MAP: Dict[str, List[str]] = {
    "steady_follow":      ["clear_noon", "clear_sunset", "overcast"],
    # Degraded visibility appears in EVERY category, not only in
    # adverse_weather.
    #
    # In the first release fog and night were confined to the adverse_weather
    # category. Two things followed. Coverage was thin -- 37 foggy scenes in
    # total, four of which reached the test split -- so no sensor comparison
    # under degraded visibility could be conclusive. Worse, weather was
    # confounded with scenario: the fog subset was behaviourally homogeneous,
    # its gap barely varied, and a per-group error therefore measured an
    # easier task rather than a harder sensing condition. Spreading the
    # degraded presets across categories decorrelates the two and multiplies
    # the adverse-visibility material several-fold.
    "steady_follow":      ["clear_noon", "clear_sunset", "overcast",
                           "light_rain", "night_clear", "dense_fog"],
    "stop_and_go":        ["clear_noon", "overcast", "light_rain",
                           "heavy_rain", "night_rain", "dense_fog"],
    "aggressive_leader":  ["clear_noon", "clear_sunset", "overcast",
                           "heavy_rain", "night_clear"],
    "cut_in":             ["clear_noon", "clear_sunset", "light_rain",
                           "night_clear", "dense_fog"],
    "cut_out":            ["clear_noon", "overcast", "heavy_rain",
                           "night_clear"],
    "speed_limit_change": ["clear_noon", "clear_sunset", "overcast",
                           "light_rain", "night_rain"],
    "grade_change":       ["clear_noon", "light_rain", "heavy_rain",
                           "dense_fog", "night_clear"],
    "adverse_weather":    ["heavy_rain", "dense_fog", "night_clear",
                           "night_rain"],
    "pedestrian_crossing":["clear_noon", "clear_sunset", "overcast",
                           "night_clear", "dense_fog"],
}

# CARLA maps — highway vs urban preference per scenario
HIGHWAY_MAPS = ["Town04", "Town06"]
URBAN_MAPS   = ["Town01", "Town02", "Town03", "Town05", "Town10HD"]
ALL_MAPS     = HIGHWAY_MAPS + URBAN_MAPS


# ---------------------------------------------------------------------------
# Traffic-regime calibration
# ---------------------------------------------------------------------------
# Stage 1's LLM describes a driving STYLE — relative aggressiveness, relative
# headway — but it has no way to know whether the scene will be placed on a
# congested urban arterial or a motorway. Left uncalibrated, every scene ran
# at one speed scale regardless of road type, which put the kinematic
# distributions far from both reference datasets (mean spacing ~41 m against
# NGSIM's 18 m and HighD's 22 m) and made the follower drift permanently into
# free-flow instead of car-following.
#
# This layer anchors the ABSOLUTE scale to the road context while preserving
# the LLM's relative style structure. Targets are the same empirical
# references Stage 4 validates against (NGSIM I-80, HighD).
REGIME_OF_MAP: Dict[str, str] = {
    "Town04": "highway", "Town06": "highway",
    "Town01": "urban",   "Town02": "urban",  "Town03": "urban",
    "Town05": "urban",   "Town07": "urban",  "Town10HD": "urban",
}

REGIME_TARGETS: Dict[str, Dict[str, float]] = {
    #                 cruising speed   equilibrium bumper gap
    "urban":   {"cruise_mps":  6.18, "spacing_m": 18.30},  # Waymo, city driving
    "highway": {"cruise_mps": 20.72, "spacing_m": 28.37},  # HighD, event means
}

# Both targets are the EVENT-level means of the calibration split of a corpus
# whose road context matches the regime. Two earlier mistakes are corrected
# here, and both mattered.
#
# First, the original defaults were fleet-level summary statistics quoted in
# the literature rather than the car-following EVENT population the dataset
# models. Measured on FollowNet's extractions the two differ materially: HighD
# car-following events average 20.7 m/s and 28.4 m, not 28 m/s and 22 m.
#
# Second, and less obvious, the urban target was taken from NGSIM I-80. That
# is a six-lane freeway at Emeryville, not city driving: its events average
# 8.77 m/s where genuine city driving (Waymo) averages 6.18 m/s. A first
# recalibration re-derived targets from calibration splits but kept NGSIM for
# urban, so it calibrated the urban regime to 1.42x the speed of the corpus it
# was then validated against. The urban target here is Waymo, so calibration
# and validation finally reference the same road context.
#
# CF_REGIME_TARGETS lets a calibration split supply the targets instead, so the
# constants can be re-derived from data rather than edited by hand. Format:
#   CF_REGIME_TARGETS='{"urban":{"cruise_mps":8.77,"spacing_m":21.71},...}'
_env_targets = os.environ.get("CF_REGIME_TARGETS")
if _env_targets:
    try:
        REGIME_TARGETS = {**REGIME_TARGETS, **json.loads(_env_targets)}
        logger.warning("REGIME_TARGETS overridden from environment: %s",
                       REGIME_TARGETS)
    except Exception as _exc:                       # noqa: BLE001
        logger.error("ignoring malformed CF_REGIME_TARGETS: %s", _exc)

# Style acts as a multiplier around the regime mean, so averaging over the
# three styles reproduces the reference mean.
STYLE_SPEED_MULT:   Dict[str, float] = {
    "cautious": 0.90, "normal": 1.00, "aggressive": 1.12,
}
STYLE_HEADWAY_MULT: Dict[str, float] = {
    "cautious": 1.30, "normal": 1.00, "aggressive": 0.72,
}

# The ego must WANT to go faster than the leader cruises, otherwise the IDM
# interaction term never binds and the gap grows without limit — which is
# exactly how the uncalibrated runs drifted to ~41 m.
#
# The margin is also what buys a STABLE time headway. Equilibrium is
#     s_eq = (s0 + v*T) / sqrt(1 - (v/v_des)^delta),
# so for a fixed target gap a bigger margin shrinks the denominator less and
# therefore admits a LARGER T. At 1.15 the highway solution was T = 0.44 s,
# which the aggressive style multiplier drove to the 0.30 s bound — 9 m at
# 31 m/s. IDM is string-unstable there: the measured follower surged to its
# desired speed, braked at -7 m/s^2, and diverged to a 216 m gap. At 1.5 the
# same target gap is met with T = 0.63 s, which is both realistic and stable.
EGO_DESIRED_SPEED_MARGIN = 1.5

# Hard floor on the time headway after the style multiplier. String stability
# — not the parameter bounds — is what sets this: sub-0.5 s headways at
# motorway speed oscillate and lose the leader.
MIN_TIME_HEADWAY_S = 0.5

# Absolute ceiling on the ego's DESIRED speed (~130 km/h).
#
# The margin above is a modelling device to keep the IDM interaction term
# engaged; it was never meant to describe a speed anyone drives. Unbounded,
# an aggressive highway ego wanted 31.4 x 1.5 = 47 m/s (169 km/h). Two
# consequences, both observed: whenever it lost sight of the leader it
# accelerated toward that and left the road (guardrail, sidewalk and traffic
# light impacts at 32-35 m/s), and it cruised faster than the speed the
# Gipps gap sizing assumed, turning a safe initial gap into a rear-end.
#
# The margin is recomputed against this cap so the equilibrium algebra stays
# self-consistent rather than being silently violated.
MAX_DESIRED_SPEED_MPS = 36.0

# Reduced-visibility scenarios: drive slower, leave more room.
ADVERSE_SPEED_FACTOR   = 0.75
ADVERSE_HEADWAY_FACTOR = 1.25

# Reaction lag must stay at or below the time headway (see calibrate_to_regime).
REACTION_HEADWAY_RATIO = 1.05
SAFE_SPACING_MARGIN_M  = 1.0   # slack above the Gipps minimum at t=0
LAG_SAFETY_FACTOR      = 1.6   # first-order lag vs pure-delay stopping distance

# Closed-loop gap correction (empirical).
#
# The equilibrium formula below describes the IDM alone. The system that
# actually runs is IDM + reaction lag + speed tracker + CARLA vehicle
# dynamics, and it settles into a limit cycle at a LARGER gap than the
# open-loop algebra predicts. Measured on settled highway scenes: a 22 m
# target produced ~26 m sustained (ratio 1.18) while urban landed on target.
# These factors invert that measured bias so the ACHIEVED distribution — the
# thing the paper compares against NGSIM/HighD — hits the reference, rather
# than the configured set-point hitting it in theory.
#
# Re-measure these if the controller gains, warmup, or dt change.
GAP_CALIBRATION: Dict[str, float] = {"urban": 1.00, "highway": 0.85}


def regime_of_map(carla_map: str) -> str:
    return REGIME_OF_MAP.get(carla_map, "urban")


def calibrate_to_regime(p: ScenarioParams, carla_map: str,
                        rng: random.Random) -> Dict[str, Any]:
    """Anchor a style's parameters to the road context.

    Returns the fields that Stage 3 should override on the SceneSpec. Only
    the speed scale, the ego time headway and the initial conditions are
    touched; accelerations, jam spacing and `delta` stay as Stage 1 produced
    them, so the LLM's behavioural signature survives.
    """
    regime = regime_of_map(carla_map)
    tgt    = REGIME_TARGETS[regime]
    style  = p.driving_style
    cat    = p.scenario_category

    speed_mult   = STYLE_SPEED_MULT.get(style, 1.0)
    headway_mult = STYLE_HEADWAY_MULT.get(style, 1.0)
    if cat == "adverse_weather":
        speed_mult   *= ADVERSE_SPEED_FACTOR
        headway_mult *= ADVERSE_HEADWAY_FACTOR

    # Leader cruises at the regime speed for this style.
    leader_cruise = tgt["cruise_mps"] * speed_mult
    # Margin, capped in absolute terms and then used CONSISTENTLY below so
    # the equilibrium relation still holds after capping.
    margin        = min(EGO_DESIRED_SPEED_MARGIN,
                        MAX_DESIRED_SPEED_MPS / max(leader_cruise, 1.0))
    margin        = max(margin, 1.02)
    ego_desired   = leader_cruise * margin

    # Choose the ego time headway so the IDM equilibrium gap lands on the
    # regime's target spacing.
    #
    # At steady state (v_ego = v_leader = v) the IDM balances its free-road
    # and interaction terms:
    #       1 - (v/v_desired)^delta = (s* / s)^2 ,   s* = s0 + v*T
    # so    s_eq = (s0 + v*T) / sqrt(1 - (v/v_desired)^delta).
    #
    # The naive s_eq = s0 + v*T only holds when the follower sits exactly AT
    # its desired speed. Here it deliberately does not — EGO_DESIRED_SPEED_
    # MARGIN keeps the ego wanting to go faster so the interaction term
    # stays engaged — so the free term never vanishes and the true
    # equilibrium gap is larger by 1/sqrt(1 - margin^-delta). Ignoring that
    # factor put highway scenes at ~34 m against a 22 m target.
    s0        = p.idm.jam_spacing_m
    delta     = max(p.idm.delta, 1.0)
    tgt_gap   = tgt["spacing_m"] * GAP_CALIBRATION.get(regime, 1.0)
    free_term = 1.0 - (1.0 / margin) ** delta
    shrink    = math.sqrt(max(free_term, 1e-3))
    base_T    = max((tgt_gap * shrink - s0) / max(leader_cruise, 1.0),
                    0.3)
    ego_T     = float(min(max(base_T * headway_mult, MIN_TIME_HEADWAY_S), 3.5))
    equilibrium = (s0 + leader_cruise * ego_T) / max(shrink, 1e-3)

    # Start at equilibrium so the scene spends its 30 s IN the car-following
    # regime rather than converging towards it.
    jitter  = lambda x, f: float(max(x * rng.uniform(1.0 - f, 1.0 + f), 0.1))
    # Never start BELOW the IDM equilibrium. Starting closer than equilibrium
    # makes the follower brake immediately, and that transient overshoots far
    # past the equilibrium gap (measured: 21 m start -> 36 m peak) and takes
    # ~25 s to decay — longer than the recording window, so the whole scene
    # records the transient rather than steady following. Opening from
    # slightly wide instead is gentle: the follower just accelerates in.
    # Bias ABOVE equilibrium. The realised gap comes out ~2 m short of the
    # configured value (lane arc-length vs straight-line measurement, plus
    # settling), and starting even slightly short is what triggers the
    # braking overshoot. Opening from wide costs nothing: the follower just
    # closes in gently during warmup.
    spacing = equilibrium * rng.uniform(1.10, 1.30)
    # Start the pair MATCHED. Equilibrium car-following means equal speeds;
    # jittering the two independently opened up to a 20 % relative
    # difference at t=0, and any initial speed deficit integrates straight
    # into a permanently larger gap over the 30 s scene.
    v0_pair = jitter(leader_cruise, 0.10)
    v0_ldr  = v0_pair
    v0_ego  = v0_pair

    # The jitter can leave the ego faster than the leader at t=0. Guarantee
    # the opening gap still satisfies the Gipps criterion, so no scene starts
    # in an already-unrecoverable state.
    # The reaction lag is a first-order filter, not a pure delay: after tau
    # only ~63% of the commanded deceleration has been reached, so the real
    # stopping distance exceeds the textbook Gipps figure. Inflate the
    # effective reaction time when sizing the safe gap.
    tau_eff = (min(p.idm.reaction_time_s, ego_T / REACTION_HEADWAY_RATIO)
               * LAG_SAFETY_FACTOR)
    b_l     = min(max(p.leader_idm.comfortable_decel_mps2,
                      p.edge.edge_decel_mps2, 0.1), EMERGENCY_DECEL_MPS2)
    safe    = (v0_ego * tau_eff + v0_ego ** 2 / (2.0 * EMERGENCY_DECEL_MPS2)
               - v0_ldr ** 2 / (2.0 * b_l) + s0)
    spacing = max(spacing, safe + SAFE_SPACING_MARGIN_M)

    idm        = p.idm.to_dict()
    leader_idm = p.leader_idm.to_dict()
    idm["desired_speed_mps"]        = ego_desired
    idm["time_headway_s"]           = ego_T
    leader_idm["desired_speed_mps"] = leader_cruise

    # Keep the reaction lag no longer than the time headway. With both
    # vehicles braking at tyre limit, the Gipps safe distance reduces to
    # v·tau + s0, while the IDM equilibrium gap is s0 + v·T — so T >= tau is
    # exactly the condition for the steady-following state to be collision
    # free. Style should express itself as "leaves more room", not "reacts
    # too late to survive its own chosen headway".
    idm["reaction_time_s"] = float(
        min(p.idm.reaction_time_s, ego_T / REACTION_HEADWAY_RATIO)
    )

    return {
        "regime":                   regime,
        "idm_params":               idm,
        "leader_idm_params":        leader_idm,
        "initial_spacing_m":        spacing,
        "initial_ego_speed_mps":    v0_ego,
        "initial_leader_speed_mps": v0_ldr,
    }

SCENARIO_MAP_PREF: Dict[str, List[str]] = {
    "steady_follow":      HIGHWAY_MAPS,
    "stop_and_go":        URBAN_MAPS,
    "aggressive_leader":  ALL_MAPS,
    "cut_in":             HIGHWAY_MAPS + ["Town05"],
    "cut_out":            HIGHWAY_MAPS,
    "speed_limit_change": URBAN_MAPS + ["Town04"],
    "grade_change":       ["Town06", "Town07"],   # hilly maps
    "adverse_weather":    ALL_MAPS,
    "pedestrian_crossing":URBAN_MAPS,             # pedestrians on urban maps
}


# ---------------------------------------------------------------------------
# Injection types
# ---------------------------------------------------------------------------

class InjectionType(str, Enum):
    NONE               = "none"
    SUDDEN_BRAKE       = "sudden_brake"       # leader brakes hard once
    GRADUAL_BRAKE      = "gradual_brake"      # stop-and-go repeated
    CUT_IN             = "cut_in"             # 3rd vehicle cuts in
    CUT_OUT            = "cut_out"            # leader leaves lane
    SPEED_LIMIT_CHANGE = "speed_limit_change"
    GRADE_CHANGE       = "grade_change"
    PEDESTRIAN         = "pedestrian"         # ped crosses in front of leader


@dataclass
class InjectionEvent:
    injection_type:     str   = InjectionType.NONE
    trigger_time_s:     float = 99999.0
    # Brake
    target_decel_mps2:  float = 0.0
    brake_duration_s:   float = 0.0
    resume_speed_mps:   float = 0.0    # speed after recovery
    # Cut-in
    cut_in_ttc_s:       float = 2.0
    cut_in_from_left:   bool  = True
    # Cut-out
    cut_out_direction:  int   = 1       # +1=right, -1=left
    # Speed limit
    new_speed_limit_mps:float = 0.0
    # Grade
    grade_deg:          float = 0.0
    # Pedestrian
    ped_cross_distance_m:  float = 30.0
    ped_cross_speed_mps:   float =  1.4  # normal walking ~5 km/h
    ped_cross_direction:   int   =  1    # +1=left-to-right, -1=right-to-left

    def to_dict(self) -> Dict: return asdict(self)


# ---------------------------------------------------------------------------
# Pedestrian config
# ---------------------------------------------------------------------------

@dataclass
class PedestrianConfig:
    enabled:           bool  = False
    cross_distance_m:  float = 30.0   # ahead of leader when crossing starts
    cross_speed_mps:   float =  1.4
    direction:         int   =  1     # +1 or -1
    trigger_time_s:    float = 10.0

    def to_dict(self) -> Dict: return asdict(self)


# ---------------------------------------------------------------------------
# Full scene specification
# ---------------------------------------------------------------------------

@dataclass
class SceneSpec:
    """Everything Stage 2 needs to execute one scene in CARLA."""

    scene_id:             str             = ""
    instance_idx:         int             = 0
    # Recorded so a user can split on vehicle identity. The first release
    # fixed both vehicles and did not record them, which made a held-out
    # vehicle evaluation impossible after the fact.
    ego_blueprint:        str             = ""
    leader_blueprint:     str             = ""
    driving_style:        str             = "normal"
    scenario_category:    str             = "steady_follow"
    carla_map:            str             = "Town04"
    regime:               str             = "urban"   # urban | highway
    weather:              Dict            = field(default_factory=dict)
    road_grade_deg:       float           = 0.0

    # Ego IDM controller params
    idm_params:           Dict            = field(default_factory=dict)
    # Leader IDM controller params (independent)
    leader_idm_params:    Dict            = field(default_factory=dict)
    mobil_params:         Dict            = field(default_factory=dict)

    # Scene initialisation
    initial_spacing_m:        float = 20.0
    initial_ego_speed_mps:    float = 15.0
    initial_leader_speed_mps: float = 15.0

    # Fixed timing
    scene_duration_s:     float = 30.0
    sampling_interval_s:  float =  0.1   # 10 Hz data recording
    physics_fps:          int   = 20     # CARLA ticks per second

    # Edge-case injection
    injection:            InjectionEvent   = field(default_factory=InjectionEvent)
    pedestrian:           PedestrianConfig = field(default_factory=PedestrianConfig)

    # Metadata
    llm_reasoning:     str = ""
    consistency_votes: int =  0

    def to_dict(self) -> Dict: return asdict(self)

    @property
    def n_physics_ticks(self) -> int:
        return int(self.scene_duration_s * self.physics_fps)

    @property
    def record_every_n_ticks(self) -> int:
        """Save one frame every N ticks to achieve sampling_interval_s."""
        return max(1, int(self.sampling_interval_s * self.physics_fps))

    @property
    def n_recorded_frames(self) -> int:
        return self.n_physics_ticks // self.record_every_n_ticks


# ---------------------------------------------------------------------------
# Injection builder
# ---------------------------------------------------------------------------

class InjectionBuilder:
    def build(self, p: ScenarioParams, rng: random.Random) -> Tuple[InjectionEvent, PedestrianConfig]:
        cat = p.scenario_category
        e   = p.edge
        ped = PedestrianConfig()   # default disabled

        # The leader cannot out-brake physics. EDGE_BOUNDS allows up to
        # 10 m/s^2, but the Gipps admission check that sizes the following
        # gap assumes the tyre limit (8 m/s^2) — so a leader braking at 10
        # invalidated the guarantee and rear-ended the follower regardless of
        # how correctly the gap was calculated. Clamp it at the same limit
        # the safety analysis uses.
        e.edge_decel_mps2 = min(e.edge_decel_mps2, EMERGENCY_DECEL_MPS2)

        if cat == "aggressive_leader":
            inj = InjectionEvent(
                injection_type    = InjectionType.SUDDEN_BRAKE,
                trigger_time_s    = e.edge_trigger_time_s,
                target_decel_mps2 = e.edge_decel_mps2,
                brake_duration_s  = rng.uniform(1.5, 4.0),
                resume_speed_mps  = p.scene.initial_leader_speed_mps,
            )

        elif cat == "cut_in":
            inj = InjectionEvent(
                injection_type   = InjectionType.CUT_IN,
                trigger_time_s   = e.edge_trigger_time_s,
                cut_in_ttc_s     = e.edge_cut_in_ttc_s,
                cut_in_from_left = rng.choice([True, False]),
            )

        elif cat == "cut_out":
            inj = InjectionEvent(
                injection_type      = InjectionType.CUT_OUT,
                trigger_time_s      = e.edge_trigger_time_s,
                cut_out_direction   = rng.choice([-1, 1]),
            )

        elif cat == "stop_and_go":
            inj = InjectionEvent(
                injection_type    = InjectionType.GRADUAL_BRAKE,
                trigger_time_s    = e.edge_trigger_time_s,
                target_decel_mps2 = min(e.edge_decel_mps2, 4.0),
                brake_duration_s  = rng.uniform(4.0, 9.0),
                resume_speed_mps  = p.scene.initial_leader_speed_mps,
            )

        elif cat == "speed_limit_change":
            inj = InjectionEvent(
                injection_type      = InjectionType.SPEED_LIMIT_CHANGE,
                trigger_time_s      = e.edge_trigger_time_s,
                new_speed_limit_mps = p.leader_idm.desired_speed_mps * rng.uniform(0.5, 0.75),
            )

        elif cat == "grade_change":
            inj = InjectionEvent(
                injection_type = InjectionType.GRADE_CHANGE,
                trigger_time_s = e.edge_trigger_time_s,
                grade_deg      = p.road_grade_deg,
            )

        elif cat == "pedestrian_crossing":
            inj = InjectionEvent(
                injection_type     = InjectionType.PEDESTRIAN,
                trigger_time_s     = e.edge_trigger_time_s,
                target_decel_mps2  = e.edge_decel_mps2,
                brake_duration_s   = 4.0,
                resume_speed_mps   = p.scene.initial_leader_speed_mps,
                ped_cross_distance_m = e.ped_cross_distance_m,
                ped_cross_speed_mps  = rng.uniform(1.0, 1.8),
                ped_cross_direction  = rng.choice([-1, 1]),
            )
            ped = PedestrianConfig(
                enabled           = True,
                cross_distance_m  = e.ped_cross_distance_m,
                cross_speed_mps   = inj.ped_cross_speed_mps,
                direction         = inj.ped_cross_direction,
                trigger_time_s    = e.edge_trigger_time_s,
            )

        else:  # steady_follow, adverse_weather
            inj = InjectionEvent(injection_type=InjectionType.NONE)

        return inj, ped


# ---------------------------------------------------------------------------
# Scenario Manager
# ---------------------------------------------------------------------------

class ScenarioManager:
    """
    Converts pre-perturbed ScenarioParams instances (from Stage 1's
    `sample_instances()`) into SceneSpec objects.

    Dataset size
    ------------
    n_instances=20 → 3 styles × 9 categories × 20 = 540 scenes
    Each scene: 30 s × 10 Hz = 300 frames
    Total frames: 540 × 300 = 162,000

    n_instances=50 → 1,350 scenes → 405,000 frames
    """

    def __init__(self, seed: int = 42):
        self._rng     = random.Random(seed)
        self._inj     = InjectionBuilder()
        self._counter = 0

    # ── public ─────────────────────────────────────────────────────

    def build_from_instances(
        self,
        instances: Dict[str, Dict[str, List[ScenarioParams]]],
    ) -> List[SceneSpec]:
        """
        `instances` is the output of LLMParameterGenerator.sample_instances():
            [style][category] → List[ScenarioParams]
        Returns a flat list of SceneSpec objects (one per instance).
        """
        specs: List[SceneSpec] = []
        for style in DRIVING_STYLES:
            for cat in SCENARIO_CATEGORIES:
                param_list = instances.get(style, {}).get(cat, [])
                for idx, p in enumerate(param_list):
                    spec = self._make_spec(p, idx)
                    specs.append(spec)
        self._rng.shuffle(specs)
        logger.info("ScenarioManager: %d scenes total "
                    "(%d frames each @ %.1f Hz → %d frames total)",
                    len(specs), specs[0].n_recorded_frames if specs else 0,
                    1.0 / specs[0].sampling_interval_s if specs else 10,
                    len(specs) * (specs[0].n_recorded_frames if specs else 0))
        return specs

    @staticmethod
    def split(
        specs: List[SceneSpec],
        train: float = 0.70,
        val:   float = 0.15,
    ) -> Tuple[List[SceneSpec], List[SceneSpec], List[SceneSpec]]:
        n    = len(specs)
        n_tr = int(n * train)
        n_va = int(n * val)
        return specs[:n_tr], specs[n_tr: n_tr + n_va], specs[n_tr + n_va:]

    # ── internal ───────────────────────────────────────────────────

    def _make_spec(self, p: ScenarioParams, idx: int) -> SceneSpec:
        self._counter += 1
        scene_id = f"CFD-scene-{self._counter:04d}"
        cat      = p.scenario_category
        weather  = self._rng.choice(SCENARIO_WEATHER_MAP.get(cat, ["clear_noon"]))
        cmap     = self._rng.choice(SCENARIO_MAP_PREF.get(cat, ALL_MAPS))
        inj, ped = self._inj.build(p, self._rng)

        # Anchor the style's absolute speed/spacing scale to the road context.
        cal = calibrate_to_regime(p, cmap, self._rng)

        return SceneSpec(
            scene_id                  = scene_id,
            instance_idx              = idx,
            driving_style             = p.driving_style,
            scenario_category         = cat,
            carla_map                 = cmap,
            regime                    = cal["regime"],
            weather                   = WEATHER_PRESETS[weather],
            road_grade_deg            = p.road_grade_deg,
            idm_params                = cal["idm_params"],
            leader_idm_params         = cal["leader_idm_params"],
            mobil_params              = p.mobil.to_dict(),
            initial_spacing_m         = cal["initial_spacing_m"],
            initial_ego_speed_mps     = cal["initial_ego_speed_mps"],
            initial_leader_speed_mps  = cal["initial_leader_speed_mps"],
            scene_duration_s          = 30.0,
            sampling_interval_s       =  0.1,
            physics_fps               = 20,
            injection                 = inj,
            pedestrian                = ped,
            llm_reasoning             = p.llm_reasoning,
            consistency_votes         = p.consistency_votes,
        )
