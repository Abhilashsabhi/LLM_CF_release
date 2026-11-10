"""
utils/idm_controller.py
========================
IDM controller for both the EGO and LEADER vehicles.

LeaderIDMController
-------------------
The leader has no vehicle in front of it (free-flow). It drives toward its
desired speed using the free-road portion of IDM, with:
  * Small Gaussian speed noise to model human-like speed fluctuation
  * Reaction to a virtual "phantom" slow vehicle injected at trigger time
    (models the leader's own braking / stop-and-go / ped reaction)

EgoIDMController
-----------------
Standard IDM with reaction-time lag filter. Inputs are the live gap and
relative speed to the leader (or to the cut-in vehicle when active).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class IDMConfig:
    desired_speed_mps:       float = 25.0
    time_headway_s:          float =  1.5
    max_acceleration_mps2:   float =  1.5
    comfortable_decel_mps2:  float =  2.0
    jam_spacing_m:           float =  2.0
    reaction_time_s:         float =  0.8
    delta:                   float =  4.0

    @classmethod
    def from_dict(cls, d: dict) -> "IDMConfig":
        return cls(**{k: float(v) for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class CarFollowState:
    """Car-following state at one timestep."""
    ego_speed_mps:    float
    leader_speed_mps: float
    spacing_m:        float   # bumper-to-bumper gap
    dt_s:             float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Tyre-limit deceleration on dry asphalt (mu ~0.8). Mirrors
# stage1_llm_generator.EMERGENCY_DECEL_MPS2 — the value the Gipps admission
# check assumes is available, so the controller must be able to deliver it.
EMERGENCY_DECEL_MPS2 = 8.0

# Below this commanded deceleration the driver is reacting to an emergency
# and the reaction-smoothing filter is bypassed (see EgoIDMController.step).
STARTLE_ACC_MPS2 = -4.0

def _idm_acceleration(v: float, v_leader: float, gap: float,
                       cfg: IDMConfig) -> float:
    """
    Full IDM formula.
        v       — ego speed
        v_leader— speed of the vehicle directly ahead
        gap     — bumper-to-bumper spacing (m)
    """
    v      = max(v, 0.0)
    gap    = max(gap, 0.01)
    delta_v = v - v_leader   # positive → approaching
    s_star  = (cfg.jam_spacing_m
               + max(0.0, v * cfg.time_headway_s
                     + v * delta_v / (2.0 * math.sqrt(
                         max(cfg.max_acceleration_mps2 *
                             cfg.comfortable_decel_mps2, 0.01)))))
    free   = 1.0 - (v / max(cfg.desired_speed_mps, 0.01)) ** cfg.delta
    inter  = (s_star / gap) ** 2
    a      = cfg.max_acceleration_mps2 * (free - inter)
    # Floor at the vehicle's BRAKING CAPABILITY, not at its comfort
    # preference. `comfortable_decel_mps2` says how hard this driver LIKES
    # to brake; clamping the output to 1.5x that meant a cautious profile
    # (b = 1.5) could never command more than -2.25 m/s^2 — physically
    # unable to avoid a leader braking hard in front of it, and inconsistent
    # with the Gipps admission check, which assumes tyre-limit braking is
    # available. Normal driving is unaffected: the IDM only reaches these
    # magnitudes when a collision is genuinely imminent.
    return max(a, -EMERGENCY_DECEL_MPS2)


def _acc_to_carla(acc: float, a_max: float, b: float) -> dict:
    """Open-loop acceleration → pedal map. Kept for reference/tests only.

    NOT used for control: see LongitudinalTracker. Mapping throttle linearly
    from the DESIRED acceleration ignores the fact that a pedal position has
    to overcome drag and rolling resistance before it produces any
    acceleration at all. In steady cruise the IDM asks for ~0 m/s², this map
    returns ~0 throttle, and the vehicle coasts down until drag balances
    whatever residual pedal it has — which pinned every vehicle near 21 m/s
    regardless of a 28-32 m/s desired speed.
    """
    if acc >= 0:
        throttle = min(acc / max(a_max, 0.01), 1.0)
        brake    = 0.0
    else:
        throttle = 0.0
        brake    = min(abs(acc) / max(b, 0.01), 1.0)
    return {
        "throttle": throttle, "brake": brake,
        "steer": 0.0, "hand_brake": False,
        "manual_gear_shift": False, "gear": 1,
        "raw_acceleration_mps2": acc,
    }


# Achievable longitudinal accelerations used to scale the feed-forward term.
FF_ACCEL_SCALE_MPS2 = 3.0   # full throttle, mid speed range
FF_BRAKE_SCALE_MPS2 = 8.0   # full brake, dry asphalt


class LongitudinalTracker:
    """Cascade controller: IDM acceleration → target speed → pedal.

    The IDM is a model of desired ACCELERATION; CARLA wants a pedal
    position. Converting between them open-loop leaves a standing error
    equal to whatever drag the vehicle happens to have at that speed. This
    tracker integrates the commanded acceleration into a reference speed and
    closes the loop on it, so the integral term absorbs drag automatically
    and the vehicle actually reaches its desired speed.

    Feed-forward keeps the response immediate (an emergency deceleration
    brakes on the same tick rather than waiting for error to build), while
    the PI terms remove the steady-state offset.
    """

    # The integral term is what cancels aerodynamic drag. Near cruise the
    # IDM asks for only a few tenths of a m/s², so the feed-forward pedal is
    # small and drag wins unless the integrator has enough authority to hold
    # the extra throttle. Under-gained values left both vehicles ~4 m/s below
    # their desired speed, and the follower's persistent ~1 m/s deficit made
    # the gap grow all scene.
    KP = 0.60
    KI = 0.60
    I_LIMIT      = 8.0    # anti-windup clamp on the integral term
    # How far the reference speed may run ahead of the actual speed. This is
    # the tracker's contribution to overshoot: at 4 m/s the follower kept
    # surging past the leader and then braking hard. Tightening it damps the
    # surge without costing the authority that cancels drag.
    REF_SLACK_MPS = 2.5

    def __init__(self, v_initial: float, v_max: float):
        self.v_ref = max(v_initial, 0.0)
        self.v_max = max(v_max, 0.1)
        self._i    = 0.0

    def step(self, acc_cmd: float, v_actual: float, dt_s: float) -> dict:
        v_actual = max(v_actual, 0.0)

        # 1. Integrate the IDM's desired acceleration into a reference speed.
        self.v_ref = float(np.clip(self.v_ref + acc_cmd * dt_s, 0.0, self.v_max))
        # Never let the reference run away from what the vehicle is doing,
        # or the controller saturates and loses authority.
        self.v_ref = float(np.clip(self.v_ref,
                                   v_actual - self.REF_SLACK_MPS,
                                   v_actual + self.REF_SLACK_MPS))

        # 2. Feed-forward from the commanded acceleration.
        u_ff = (acc_cmd / FF_ACCEL_SCALE_MPS2 if acc_cmd >= 0.0
                else acc_cmd / FF_BRAKE_SCALE_MPS2)

        # 3. PI feedback on speed error.
        err = self.v_ref - v_actual
        self._i = float(np.clip(self._i + err * dt_s,
                                -self.I_LIMIT, self.I_LIMIT))
        u = u_ff + self.KP * err + self.KI * self._i

        if u >= 0.0:
            throttle, brake = min(u, 1.0), 0.0
        else:
            throttle, brake = 0.0, min(-u, 1.0)
            # Bleed the integral while braking so it cannot fight the brake.
            self._i *= 0.9

        return {
            "throttle": throttle, "brake": brake,
            "steer": 0.0, "hand_brake": False,
            "manual_gear_shift": False, "gear": 1,
            "raw_acceleration_mps2": acc_cmd,
            "v_ref_mps": self.v_ref,
        }

    def emergency(self) -> dict:
        self.v_ref = 0.0
        self._i    = 0.0
        return {"throttle": 0.0, "brake": 1.0, "steer": 0.0,
                "hand_brake": False, "manual_gear_shift": False,
                "gear": 1, "raw_acceleration_mps2": -FF_BRAKE_SCALE_MPS2,
                "v_ref_mps": 0.0}


# ---------------------------------------------------------------------------
# Ego IDM controller
# ---------------------------------------------------------------------------

class EgoIDMController:
    """
    Standard IDM with first-order reaction-time lag.
    Safe-stop override if gap < MIN_SPACING.
    """
    MIN_SPACING = 0.5  # m — full emergency brake

    def __init__(self, config: IDMConfig, v_initial: float = 0.0):
        self.cfg       = config
        self._prev_acc = 0.0
        self._long     = LongitudinalTracker(v_initial, config.desired_speed_mps)

    def step(self, state: CarFollowState) -> dict:
        if state.spacing_m < self.MIN_SPACING:
            return self._long.emergency()

        acc = _idm_acceleration(
            state.ego_speed_mps,
            state.leader_speed_mps,
            state.spacing_m,
            self.cfg,
        )
        if acc <= STARTLE_ACC_MPS2:
            # Emergency: bypass the smoothing filter.
            #
            # The first-order lag models ordinary reaction smoothness, but it
            # reaches only ~63% of a commanded deceleration after tau. When
            # the IDM is calling for hard braking that shortfall is exactly
            # what turns a marginal gap into a rear-end — and it contradicts
            # the Gipps admission check, which assumes the deceleration is
            # actually delivered. A real driver stamps on the pedal instead
            # of easing into it.
            acc_lag = acc
        else:
            alpha   = state.dt_s / max(state.dt_s + self.cfg.reaction_time_s, 1e-6)
            acc_lag = alpha * acc + (1.0 - alpha) * self._prev_acc
        self._prev_acc = acc_lag

        return self._long.step(acc_lag, state.ego_speed_mps, state.dt_s)


# ---------------------------------------------------------------------------
# Leader IDM controller
# ---------------------------------------------------------------------------

class LeaderIDMController:
    """
    Controls the lead vehicle.

    In FREE-FLOW mode it accelerates/decelerates to reach `desired_speed_mps`
    with small Gaussian speed noise (human-like fluctuation).

    SCRIPTED PHASES override free-flow:
        phase = "brake"   — apply fixed deceleration for `duration` seconds
        phase = "stop"    — hold near-zero speed (stop-and-go stopped)
        phase = "resume"  — accelerate back toward desired speed
        phase = "speed_limit" — lower target speed permanently

    The executor calls `set_phase()` at the injection trigger time.
    """

    SPEED_NOISE_STD = 0.3   # m/s, human speed fluctuation

    def __init__(self, config: IDMConfig, rng_seed: int = 0,
                 v_initial: float = 0.0):
        self.cfg         = config
        self._rng        = np.random.default_rng(rng_seed)
        self._long       = LongitudinalTracker(v_initial,
                                               config.desired_speed_mps)
        self._phase      = "free"
        self._phase_t    = 0.0      # elapsed time in current phase
        self._brake_dur  = 0.0
        self._target_v   = config.desired_speed_mps
        self._prev_acc   = 0.0

    # ── public ─────────────────────────────────────────────────────

    @property
    def in_scripted_phase(self) -> bool:
        """True while an injected manoeuvre is still driving the leader.

        Covers the whole disturbance, not just the braking window: a
        sudden-brake injection stays active through "brake" → "resume"
        until the leader is back at its target speed, and a speed-limit
        change ("speed_limit") stays active for the rest of the scene
        because the lowered target never expires.
        """
        return self._phase != "free"

    def set_phase_brake(self, decel_mps2: float, duration_s: float,
                        resume_speed_mps: float) -> None:
        self._phase       = "brake"
        self._phase_t     = 0.0
        self._brake_decel = decel_mps2
        self._brake_dur   = duration_s
        self._resume_v    = resume_speed_mps
        self._prev_acc    = -decel_mps2   # reset lag — braking is immediate

    def set_phase_stop_and_go(self, decel_mps2: float, stop_dur: float,
                               resume_speed_mps: float) -> None:
        """Brake → stop → resume cycle."""
        self._phase       = "stop_and_go_brake"
        self._phase_t     = 0.0
        self._brake_decel = decel_mps2
        self._brake_dur   = stop_dur
        self._resume_v    = resume_speed_mps

    def set_phase_speed_limit(self, new_speed_mps: float) -> None:
        self._phase    = "speed_limit"
        self._target_v = new_speed_mps

    def step(self, current_speed_mps: float, dt_s: float) -> dict:
        """Returns CARLA control dict for the leader."""
        v = max(current_speed_mps, 0.0)

        if self._phase in ("brake", "ped_brake"):
            self._phase_t += dt_s
            acc = -self._brake_decel
            if self._phase_t >= self._brake_dur:
                self._phase   = "resume"
                self._phase_t = 0.0

        elif self._phase in ("stop_and_go_brake",):
            self._phase_t += dt_s
            if v > 0.5:
                acc = -self._brake_decel
            else:
                self._phase   = "stop_and_go_stopped"
                self._phase_t = 0.0
                acc           = 0.0

        elif self._phase == "stop_and_go_stopped":
            self._phase_t += dt_s
            acc = 0.0
            if self._phase_t > 3.0:   # wait ~3 s then resume
                self._phase   = "resume"
                self._phase_t = 0.0

        elif self._phase == "resume":
            # Proportional controller back to resume speed
            err = self._resume_v - v
            acc = float(np.clip(err * 0.8,
                                -self.cfg.comfortable_decel_mps2,
                                 self.cfg.max_acceleration_mps2))
            if abs(err) < 0.5:
                self._phase = "free"
                self._target_v = self._resume_v

        elif self._phase == "speed_limit":
            err = self._target_v - v
            acc = float(np.clip(err * 0.8,
                                -self.cfg.comfortable_decel_mps2,
                                 self.cfg.max_acceleration_mps2))

        else:  # "free"
            # IDM free-road term + small Gaussian noise
            free_acc = (self.cfg.max_acceleration_mps2
                        * (1.0 - (v / max(self._target_v, 0.01)) ** self.cfg.delta))
            noise    = float(self._rng.normal(0.0, self.SPEED_NOISE_STD))
            acc      = float(np.clip(
                free_acc + noise * 0.1,
                -self.cfg.comfortable_decel_mps2,
                 self.cfg.max_acceleration_mps2,
            ))

        # Reaction-time lag
        alpha          = dt_s / max(dt_s + self.cfg.reaction_time_s * 0.5, 1e-6)
        acc_lag        = alpha * acc + (1.0 - alpha) * self._prev_acc
        self._prev_acc = acc_lag

        # The reference speed must follow scripted phases too, otherwise a
        # speed-limit change or a resume would be fought by the tracker.
        self._long.v_max = max(self._target_v, 0.1)
        return self._long.step(acc_lag, v, dt_s)
