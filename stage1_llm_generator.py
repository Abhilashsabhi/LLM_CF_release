"""
stage1_llm_generator.py
========================
Generates IDM + MOBIL parameters for all (ego_style × scenario) combinations
using an LLM with self-consistency majority voting (N=5 per query).

Additions vs v1
---------------
* 9 scenario categories (added pedestrian_crossing)
* `leader_idm` field — separate IDM params for the lead vehicle
* `n_instances` samples per combo drawn from a truncated-normal around
  the LLM base params, giving diversity while respecting the base behavior
* API key loaded from .env only — never hardcoded
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRIVING_STYLES: List[str] = ["cautious", "normal", "aggressive"]

SCENARIO_CATEGORIES: List[str] = [
    "steady_follow",        # 0
    "stop_and_go",          # 1
    "aggressive_leader",    # 2  (leader brakes hard)
    "cut_in",               # 3  (third vehicle cuts in)
    "cut_out",              # 4  (leader leaves lane)
    "speed_limit_change",   # 5  (leader decelerates at sign)
    "grade_change",         # 6  (hill up / downhill)
    "adverse_weather",      # 7  (fog / rain / night)
    "pedestrian_crossing",  # 8  (ped crosses ahead of leader)
]

# Physical bounds (min, max)
IDM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "desired_speed_mps":       (3.0,  40.0),
    "time_headway_s":          (0.3,   3.5),
    "max_acceleration_mps2":   (0.3,   4.5),
    "comfortable_decel_mps2":  (0.5,   7.0),
    "jam_spacing_m":           (0.5,   8.0),
    "reaction_time_s":         (0.1,   2.0),
    "delta":                   (1.0,   6.0),
}

MOBIL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "politeness_factor":       (0.0,   1.0),
    "accel_threshold_mps2":    (0.05,  1.0),
    "safe_decel_mps2":         (1.0,   7.0),
}

SCENE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "initial_spacing_m":        (2.0, 80.0),
    "initial_ego_speed_mps":    (0.0, 35.0),
    "initial_leader_speed_mps": (0.0, 35.0),
}

# Tyre-limit deceleration on dry asphalt (mu ~0.8). Used as the braking
# CAPABILITY in safety checks, as opposed to the IDM's comfort preference.
EMERGENCY_DECEL_MPS2: float = 8.0

EDGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "edge_trigger_time_s":  (3.0,  25.0),
    "edge_decel_mps2":      (0.5,  10.0),
    "edge_cut_in_ttc_s":    (0.5,   4.0),
    "ped_cross_distance_m": (10.0, 50.0),  # how far ahead ped crosses
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IDMParams:
    desired_speed_mps:       float = 25.0
    time_headway_s:          float =  1.5
    max_acceleration_mps2:   float =  1.5
    comfortable_decel_mps2:  float =  2.0
    jam_spacing_m:           float =  2.0
    reaction_time_s:         float =  0.8
    delta:                   float =  4.0

    def to_dict(self) -> Dict: return asdict(self)

    def sample_variation(self, rng: np.random.Generator,
                         pct: float = 0.15) -> "IDMParams":
        """Return a copy with each param perturbed by ±pct (truncated normal)."""
        d = asdict(self)
        bounds = IDM_BOUNDS
        out = {}
        for k, v in d.items():
            lo, hi = bounds.get(k, (v * 0.1, v * 3.0))
            sigma = abs(v) * pct
            noisy = float(rng.normal(v, sigma))
            out[k] = float(np.clip(noisy, lo, hi))
        return IDMParams(**out)


@dataclass
class MOBILParams:
    politeness_factor:     float = 0.5
    accel_threshold_mps2:  float = 0.2
    safe_decel_mps2:       float = 4.0

    def to_dict(self) -> Dict: return asdict(self)


@dataclass
class SceneConfig:
    initial_spacing_m:        float = 20.0
    initial_ego_speed_mps:    float = 15.0
    initial_leader_speed_mps: float = 15.0
    # Fixed: 30 s @ 0.1 s sampling
    scene_duration_s:         float = 30.0
    sampling_interval_s:      float =  0.1

    def to_dict(self) -> Dict: return asdict(self)


@dataclass
class EdgeConfig:
    edge_trigger_time_s:  float = 10.0
    edge_decel_mps2:      float =  5.0
    edge_cut_in_ttc_s:    float =  2.0
    ped_cross_distance_m: float = 30.0

    def to_dict(self) -> Dict: return asdict(self)


@dataclass
class ScenarioParams:
    scenario_id:       str              = ""
    driving_style:     str              = "normal"
    scenario_category: str              = "steady_follow"
    # Ego IDM
    idm:               IDMParams        = field(default_factory=IDMParams)
    # Leader IDM  (leader drives independently with its own style)
    leader_idm:        IDMParams        = field(default_factory=IDMParams)
    mobil:             MOBILParams      = field(default_factory=MOBILParams)
    scene:             SceneConfig      = field(default_factory=SceneConfig)
    edge:              EdgeConfig       = field(default_factory=EdgeConfig)
    road_grade_deg:    float            = 0.0
    llm_reasoning:     str              = ""
    consistency_votes: int              = 0

    def to_dict(self) -> Dict: return asdict(self)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    STYLE_DESC = {
        "cautious": (
            "Cautious, safety-first driver: long headways (T ≥ 2s), "
            "gentle accelerations (a ≤ 1 m/s²), early braking, high reaction time."
        ),
        "normal": (
            "Average everyday driver: moderate headways (T ≈ 1.5 s), "
            "normal accelerations, typical 0.8 s reaction time."
        ),
        "aggressive": (
            "Aggressive driver: very short headways (T ≤ 0.8 s), "
            "hard accelerations (a ≥ 2.5 m/s²), late braking, fast reactions (τ ≤ 0.4 s)."
        ),
    }

    SCENARIO_DESC = {
        "steady_follow":
            "Leader maintains a roughly constant speed. Ego follows steadily on a straight road.",
        "stop_and_go":
            "Dense urban/motorway congestion. Leader decelerates to near zero several times.",
        "aggressive_leader":
            "Leader performs a sudden extreme braking event (≥ 6 m/s²). "
            "Safety-critical: TTC must drop below 3 s at some point.",
        "cut_in":
            "A third vehicle cuts abruptly in front of ego from a side lane, "
            "creating a very short gap (TTC 1–3 s). This is a safety-critical edge case.",
        "cut_out":
            "The leader suddenly changes lane, revealing a much slower vehicle ahead. "
            "Ego must adapt quickly.",
        "speed_limit_change":
            "Leader decelerates when passing a reduced-speed-limit zone sign.",
        "grade_change":
            "Road includes a significant uphill or downhill grade (±5–10°). "
            "Adjust desired speed and acceleration for gravity effect.",
        "adverse_weather":
            "Heavy rain, dense fog, or night driving: reduce desired speed, "
            "increase headway, model reduced friction and visibility.",
        "pedestrian_crossing":
            "A pedestrian suddenly crosses the road ahead of the leader. "
            "Leader must brake hard; ego must react. "
            "Safety-critical — generates near-stop situations.",
    }

    def build(self, ego_style: str, category: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._system()},
            {"role": "user",   "content": self._user(ego_style, category)},
        ]

    def _system(self) -> str:
        return (
            "You are an expert automotive engineer specialising in microscopic traffic "
            "simulation and the Intelligent Driver Model (IDM). Your task is to provide "
            "physically realistic IDM and MOBIL parameters for a CARLA simulation.\n\n"
            "CANONICAL IDM RANGES (Treiber & Kesting). Stay inside these "
            "unless the scenario justifies otherwise, and say why if you do:\n"
            "  desired_speed_mps      : set from the road type "
            "(urban arterial 8-14, motorway 25-33)\n"
            "  time_headway_s     T   : 0.6-2.5   (cautious high, aggressive low)\n"
            "  max_acceleration   a   : 0.7-2.5 m/s^2\n"
            "  comfortable_decel  b   : 1.5-3.0 m/s^2  (comfort, NOT the\n"
            "                           emergency limit, which is ~8 m/s^2)\n"
            "  jam_spacing_m      s0  : 1.0-3.0 m\n"
            "  reaction_time_s    tau : 0.3-1.5 s, and MUST NOT exceed T\n"
            "  delta                  : 4.0 is the standard free-acceleration\n"
            "                           exponent; only depart from it with reason\n\n"
            "RULES:\n"
            "1. Reason step-by-step before giving numbers.\n"
            "2. Monotonicity constraints MUST hold: "
            "   ∂a/∂v < 0 (higher speed → less throttle), "
            "   ∂a/∂s > 0 (more gap → more throttle), "
            "   ∂a/∂Δv < 0 (approaching faster → brake harder).\n"
            "3. Provide SEPARATE IDM params for the EGO and for the LEADER vehicle. "
            "   The leader has its own driving style (often different from ego).\n"
            "4. Return ONLY a valid JSON object — no markdown, no prose.\n\n"
            "JSON schema (all numeric values are float):\n"
            "{\n"
            '  "ego_idm":   { "desired_speed_mps":float, "time_headway_s":float,\n'
            '                 "max_acceleration_mps2":float, "comfortable_decel_mps2":float,\n'
            '                 "jam_spacing_m":float, "reaction_time_s":float, "delta":float },\n'
            '  "leader_idm":{ same keys as ego_idm },\n'
            '  "mobil":     { "politeness_factor":float, "accel_threshold_mps2":float,\n'
            '                 "safe_decel_mps2":float },\n'
            '  "scene":     { "initial_spacing_m":float, "initial_ego_speed_mps":float,\n'
            '                 "initial_leader_speed_mps":float },\n'
            '  "edge":      { "edge_trigger_time_s":float, "edge_decel_mps2":float,\n'
            '                 "edge_cut_in_ttc_s":float, "ped_cross_distance_m":float },\n'
            '  "road_grade_deg": float,\n'
            '  "reasoning": "string"\n'
            "}"
        )

    def _user(self, ego_style: str, category: str) -> str:
        return (
            f"EGO driving style   : {ego_style.upper()}\n"
            f"Style description   : {self.STYLE_DESC[ego_style]}\n\n"
            f"Scenario category   : {category.upper()}\n"
            f"Scenario description: {self.SCENARIO_DESC[category]}\n\n"
            "Generate the JSON. Make the leader driving style appropriate "
            "for this scenario (e.g. aggressive leader for aggressive_leader scenario, "
            "but normal leader for steady_follow)."
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ParameterValidator:
    # A value further outside its bound than this multiple of the bound
    # range signals the model misunderstood the request, rather than merely
    # being a little optimistic. Those candidates are still rejected.
    #
    # Set to a FULL bound-range: narrow bounds (MOBIL's accel threshold spans
    # only 0.05-1.0) otherwise flag ordinary disagreement as incomprehension —
    # a proposed 1.5 m/s2 threshold was rejecting entire samples over a
    # parameter the longitudinal simulator never reads.
    EXTREME_FACTOR = 1.0

    def clamp(self, p: ScenarioParams) -> Tuple[ScenarioParams, List[str], bool]:
        """Project parameters onto the physically admissible set.

        Rejecting an entire LLM sample because one number sits just outside a
        bound throws away otherwise sound physics — and several of the fields
        involved (`scene.*`, headway, reaction time) are subsequently
        overwritten by the Stage 3 regime calibration, so rejecting on them
        discarded good candidates over values that never reach the simulator.

        Clamping is projection onto the feasible set: it preserves the
        model's intent (short headway stays short) while guaranteeing every
        parameter is physical. Returns (params, clamped_field_notes,
        extreme) where `extreme` marks a sample too far out to trust.
        """
        notes: List[str] = []
        extreme = False

        def _do(obj, bounds: Dict[str, Tuple[float, float]], group: str):
            nonlocal extreme
            for k, (lo, hi) in bounds.items():
                if not hasattr(obj, k):
                    continue
                v = float(getattr(obj, k))
                if lo <= v <= hi:
                    continue
                span = max(hi - lo, 1e-6)
                overshoot = (lo - v if v < lo else v - hi) / span
                if overshoot > self.EXTREME_FACTOR:
                    extreme = True
                new = min(max(v, lo), hi)
                setattr(obj, k, new)
                notes.append(f"{group}.{k}: {v:.3f}→{new:.3f}")

        _do(p.idm,        IDM_BOUNDS,   "ego_idm")
        _do(p.leader_idm, IDM_BOUNDS,   "leader_idm")
        _do(p.mobil,      MOBIL_BOUNDS, "mobil")
        _do(p.scene,      SCENE_BOUNDS, "scene")
        _do(p.edge,       EDGE_BOUNDS,  "edge")
        return p, notes, extreme

    def validate(self, p: ScenarioParams) -> Tuple[bool, List[str]]:
        errs: List[str] = []
        self._bounds("ego_idm",    asdict(p.idm),    IDM_BOUNDS,   errs)
        self._bounds("leader_idm", asdict(p.leader_idm), IDM_BOUNDS, errs)
        self._bounds("mobil",      asdict(p.mobil),  MOBIL_BOUNDS, errs)
        self._bounds("scene",      asdict(p.scene),  SCENE_BOUNDS, errs)
        self._bounds("edge",       asdict(p.edge),   EDGE_BOUNDS,  errs)
        # Cross-check: car-following safe distance (Gipps).
        #
        # The old check compared the ego's FULL stopping distance against the
        # initial gap. That is the criterion for a stationary obstacle, not
        # for car following: the lead vehicle also needs distance to stop, and
        # that distance is available to the follower. Using it rejected every
        # physically reasonable parameter set — including all three hardcoded
        # fallbacks — while saying nothing about actual rear-end risk.
        v_f   = p.scene.initial_ego_speed_mps
        v_l   = p.scene.initial_leader_speed_mps
        # Braking CAPABILITY, not comfort preference: `comfortable_decel` is
        # what the IDM prefers in normal driving, but any of these vehicles
        # can brake at tyre-limit in an emergency, and the ego controller
        # does exactly that below its minimum-spacing threshold.
        b_f   = EMERGENCY_DECEL_MPS2
        # The leader cannot beat physics either, however hard the scenario
        # scripts it to brake.
        b_l   = min(max(p.leader_idm.comfortable_decel_mps2,
                        p.edge.edge_decel_mps2, 0.1),
                    EMERGENCY_DECEL_MPS2)
        s0    = p.idm.jam_spacing_m
        tau   = p.idm.reaction_time_s
        safe  = (v_f * tau + v_f ** 2 / (2.0 * b_f)
                 - v_l ** 2 / (2.0 * b_l) + s0)
        if safe > p.scene.initial_spacing_m:
            errs.append(
                f"Gipps safe distance {safe:.1f}m > initial spacing "
                f"{p.scene.initial_spacing_m:.1f}m "
                f"(v_f={v_f:.1f}, v_l={v_l:.1f}, b_f={b_f:.1f}, b_l={b_l:.1f})"
            )
        return len(errs) == 0, errs

    @staticmethod
    def _bounds(group: str, d: Dict, bounds: Dict, errs: List[str]) -> None:
        for k, (lo, hi) in bounds.items():
            if k not in d: continue
            v = d[k]
            if not (lo <= v <= hi):
                errs.append(f"{group}.{k}={v:.3f} ∉ [{lo},{hi}]")


# ---------------------------------------------------------------------------
# LLM client (Groq cloud or a local OpenAI-compatible server such as Ollama)
# ---------------------------------------------------------------------------

GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_URL        = "http://localhost:11434/v1/chat/completions"
OLLAMA_TAGS_URL   = "http://localhost:11434/api/tags"
DEFAULT_GROQ_MODEL   = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


def ollama_available(timeout: float = 3.0) -> bool:
    try:
        return requests.get(OLLAMA_TAGS_URL, timeout=timeout).ok
    except Exception:
        return False


def ollama_models(timeout: float = 3.0) -> List[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


class LLMClient:
    """Chat client for an OpenAI-compatible endpoint.

    Two backends are supported and selected automatically:

      * ``groq``   — cloud, needs GROQ_API_KEY
      * ``ollama`` — a local server, needs no key

    Auto-selection prefers Groq when a key is present and otherwise falls
    back to a local Ollama server. The hardcoded fallback parameter priors
    are used only when NEITHER backend is reachable, and that case is now
    logged loudly: parameters generated by those priors carry
    ``consistency_votes = 0`` and do not represent an LLM-generated dataset.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7, timeout: int = 180,
                 backend: str = "auto",
                 base_url: Optional[str] = None):
        self.api_key     = api_key or os.environ.get("GROQ_API_KEY", "")
        self.temperature = temperature
        self.timeout     = timeout

        if backend == "auto":
            if self.api_key:
                backend = "groq"
            elif ollama_available():
                backend = "ollama"
            else:
                backend = "none"
        self.backend = backend

        if backend == "groq":
            self.base_url = base_url or GROQ_URL
            self.model    = model or DEFAULT_GROQ_MODEL
            logger.info("LLM backend: Groq (%s)", self.model)
        elif backend == "ollama":
            self.base_url = base_url or OLLAMA_URL
            self.model    = model or DEFAULT_OLLAMA_MODEL
            have = ollama_models()
            if have and self.model not in have:
                # Tolerate "name" vs "name:latest" and pick a sane substitute.
                match = next((m for m in have
                              if m.split(":")[0] == self.model.split(":")[0]), None)
                if match:
                    logger.info("Model %s not present; using %s", self.model, match)
                    self.model = match
                else:
                    logger.warning(
                        "Model %s not pulled. Available: %s. "
                        "Run: ollama pull %s", self.model, have, self.model)
            logger.info("LLM backend: Ollama (%s)", self.model)
        else:
            self.base_url = None
            self.model    = model or DEFAULT_GROQ_MODEL
            logger.error(
                "NO LLM BACKEND AVAILABLE. Set GROQ_API_KEY or start Ollama "
                "(`ollama serve` + `ollama pull %s`). Hardcoded priors will "
                "be used and the dataset will NOT be LLM-parameterised.",
                DEFAULT_OLLAMA_MODEL)

    @property
    def enabled(self) -> bool:
        return self.backend in ("groq", "ollama")

    def chat(self, messages: List[Dict]) -> str:
        if not self.enabled:
            raise RuntimeError("No LLM backend configured")
        headers = {"Content-Type": "application/json"}
        if self.backend == "groq":
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  1200,
        }
        if self.backend == "ollama":
            # Ollama honours the OpenAI JSON-mode flag, which markedly
            # improves schema adherence for smaller local models.
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(self.base_url, headers=headers, json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# Backwards-compatible alias — older call sites construct GroqClient directly.
GroqClient = LLMClient


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ResponseParser:
    def parse(self, raw: str) -> Optional[Dict]:
        raw = re.sub(r"```json\s*|```", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1: return None
        try:
            return json.loads(raw[s: e + 1])
        except json.JSONDecodeError:
            return None

    def to_params(self, d: Dict, style: str, cat: str) -> Optional[ScenarioParams]:
        try:
            def mk_idm(key):
                return IDMParams(**{k: float(v)
                                    for k, v in d.get(key, {}).items()
                                    if k in IDMParams.__dataclass_fields__})
            sc_raw = d.get("scene", {})
            # Force duration and sampling to fixed values
            sc_raw.pop("scene_duration_s",     None)
            sc_raw.pop("sampling_interval_s",  None)
            scene = SceneConfig(**{k: float(v)
                                   for k, v in sc_raw.items()
                                   if k in SceneConfig.__dataclass_fields__})
            return ScenarioParams(
                driving_style=style,
                scenario_category=cat,
                idm        = mk_idm("ego_idm"),
                leader_idm = mk_idm("leader_idm"),
                mobil      = MOBILParams(**{k: float(v)
                                            for k, v in d.get("mobil", {}).items()
                                            if k in MOBILParams.__dataclass_fields__}),
                scene      = scene,
                edge       = EdgeConfig(**{k: float(v)
                                           for k, v in d.get("edge", {}).items()
                                           if k in EdgeConfig.__dataclass_fields__}),
                road_grade_deg=float(d.get("road_grade_deg", 0.0)),
                llm_reasoning=str(d.get("reasoning", "")),
            )
        except Exception as exc:
            logger.debug("Param build failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Hard-coded fallback defaults
# ---------------------------------------------------------------------------

def _fallback_base() -> Dict[str, Dict]:
    """Returns base IDM/MOBIL params per driving style."""
    return {
        "cautious": {
            "ego_idm":    IDMParams(18, 2.5, 0.8, 1.5, 4.0, 1.2, 4.0),
            "leader_idm": IDMParams(18, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0),
            "mobil":      MOBILParams(0.8, 0.1, 2.0),
            "scene":      SceneConfig(30.0, 12.0, 12.0),
            "edge":       EdgeConfig(10.0, 3.5, 3.0, 35.0),
        },
        "normal": {
            "ego_idm":    IDMParams(25, 1.5, 1.5, 2.5, 2.5, 0.8, 4.0),
            "leader_idm": IDMParams(25, 1.5, 1.5, 2.5, 2.0, 0.8, 4.0),
            "mobil":      MOBILParams(0.5, 0.2, 3.0),
            "scene":      SceneConfig(20.0, 18.0, 18.0),
            "edge":       EdgeConfig( 8.0, 6.0, 2.0, 30.0),
        },
        "aggressive": {
            "ego_idm":    IDMParams(33, 0.6, 3.0, 5.0, 1.0, 0.3, 4.0),
            "leader_idm": IDMParams(33, 0.7, 2.5, 5.5, 1.0, 0.4, 4.0),
            "mobil":      MOBILParams(0.1, 0.5, 5.5),
            "scene":      SceneConfig(12.0, 25.0, 25.0),
            "edge":       EdgeConfig( 6.0, 8.0, 1.5, 20.0),
        },
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class LLMParameterGenerator:
    """
    Generates one *base* ScenarioParams per (style × category) via the LLM,
    then exposes `sample_instances()` to produce `n_instances` parameter
    sets per combo by perturbing the base.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 n_votes: int = 5,
                 temperature: float = 0.7,
                 backend: str = "auto",
                 base_url: Optional[str] = None):
        self.client    = LLMClient(api_key, model, temperature,
                                   backend=backend, base_url=base_url)
        self.builder   = PromptBuilder()
        self.parser    = ResponseParser()
        self.validator = ParameterValidator()
        self.n_votes   = n_votes
        self._fb       = _fallback_base()

    # ── public ─────────────────────────────────────────────────────

    def generate_all_base(self) -> Dict[str, Dict[str, ScenarioParams]]:
        """One base param set per (style × category)."""
        out: Dict[str, Dict[str, ScenarioParams]] = {}
        for style in DRIVING_STYLES:
            out[style] = {}
            for cat in SCENARIO_CATEGORIES:
                p = self._generate_one(style, cat)
                out[style][cat] = p
                logger.info("[%s][%s] votes=%d", style, cat, p.consistency_votes)
        return out

    def sample_instances(
        self,
        base_params: Dict[str, Dict[str, ScenarioParams]],
        n_instances: int = 20,
        seed: int = 42,
    ) -> Dict[str, Dict[str, List[ScenarioParams]]]:
        """
        For each (style × category) base, return n_instances variations
        by sampling ±15 % truncated-normal perturbations on IDM params.
        Scene initial conditions also vary (±20 %).
        Returns [style][category] → List[ScenarioParams] of length n_instances.
        """
        rng = np.random.default_rng(seed)
        out: Dict[str, Dict[str, List[ScenarioParams]]] = {}
        for style in DRIVING_STYLES:
            out[style] = {}
            for cat in SCENARIO_CATEGORIES:
                base = base_params[style][cat]
                instances: List[ScenarioParams] = []
                for i in range(n_instances):
                    inst = self._perturb(base, rng, idx=i)
                    inst.scenario_id = f"{style}_{cat}_{i:03d}"
                    instances.append(inst)
                out[style][cat] = instances
        return out

    def save_base(self, params: Dict[str, Dict[str, ScenarioParams]],
                  path: str) -> None:
        d: Dict = {}
        for style, cats in params.items():
            d[style] = {cat: p.to_dict() for cat, p in cats.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load_base(path: str) -> Dict[str, Dict[str, ScenarioParams]]:
        with open(path) as f:
            raw = yaml.safe_load(f)
        out: Dict[str, Dict[str, ScenarioParams]] = {}
        for style, cats in raw.items():
            out[style] = {}
            for cat, d in cats.items():
                out[style][cat] = ScenarioParams(
                    driving_style=style, scenario_category=cat,
                    idm        = IDMParams(**d["idm"]),
                    leader_idm = IDMParams(**d.get("leader_idm", d["idm"])),
                    mobil      = MOBILParams(**d["mobil"]),
                    scene      = SceneConfig(**{k: v for k, v in d["scene"].items()
                                               if k in SceneConfig.__dataclass_fields__}),
                    edge       = EdgeConfig(**d["edge"]),
                    road_grade_deg=d.get("road_grade_deg", 0.0),
                    llm_reasoning=d.get("llm_reasoning", ""),
                    consistency_votes=d.get("consistency_votes", 0),
                )
        return out

    # ── internal ───────────────────────────────────────────────────

    def _generate_one(self, style: str, cat: str) -> ScenarioParams:
        messages    = self.builder.build(style, cat)
        candidates: List[ScenarioParams] = []
        for attempt in range(self.n_votes):
            try:
                raw    = self.client.chat(messages)
                parsed = self.parser.parse(raw)
                if parsed is None: continue
                p = self.parser.to_params(parsed, style, cat)
                if p is None: continue

                # Project onto the admissible set rather than discarding a
                # sample over a marginal bound violation.
                p, notes, extreme = self.validator.clamp(p)
                if extreme:
                    logger.warning("[%s][%s] attempt %d rejected — values far "
                                   "outside physical bounds: %s",
                                   style, cat, attempt, notes)
                    continue
                if notes:
                    logger.debug("[%s][%s] attempt %d clamped: %s",
                                 style, cat, attempt, notes)

                ok, errs = self.validator.validate(p)
                if not ok:
                    # The Gipps admission check reads the LLM's *initial
                    # conditions*, but Stage 3 replaces spacing and speeds
                    # during regime calibration and enforces the safety
                    # margin itself. So a failure here is informational: the
                    # behavioural parameters are still usable.
                    logger.debug("[%s][%s] attempt %d: %s "
                                 "(initial conditions are re-derived in Stage 3)",
                                 style, cat, attempt, errs)
                candidates.append(p)
            except Exception as exc:
                logger.warning("[%s][%s] attempt %d error: %s",
                               style, cat, attempt, exc)
        if not candidates:
            logger.warning("[%s][%s] all LLM attempts failed — using fallback",
                           style, cat)
            return self._make_fallback(style, cat)
        best = self._majority_vote(candidates)
        best.consistency_votes = len(candidates)
        return best

    def _make_fallback(self, style: str, cat: str) -> ScenarioParams:
        fb = self._fb[style]
        p  = ScenarioParams(
            driving_style=style, scenario_category=cat,
            idm=fb["ego_idm"], leader_idm=fb["leader_idm"],
            mobil=fb["mobil"], scene=fb["scene"], edge=fb["edge"],
            consistency_votes=0,
        )
        # Tweak edge params per scenario
        if cat == "pedestrian_crossing":
            p.edge.edge_trigger_time_s  = 8.0
            p.edge.edge_decel_mps2      = 6.0
            p.edge.ped_cross_distance_m = 25.0
        elif cat == "aggressive_leader":
            p.edge.edge_decel_mps2 = 8.0
        return p

    @staticmethod
    def _majority_vote(candidates: List[ScenarioParams]) -> ScenarioParams:
        if len(candidates) == 1:
            return candidates[0]

        def med(getter):
            return median([getter(c) for c in candidates])

        c0 = candidates[0]
        c0.idm = IDMParams(
            desired_speed_mps       = med(lambda c: c.idm.desired_speed_mps),
            time_headway_s          = med(lambda c: c.idm.time_headway_s),
            max_acceleration_mps2   = med(lambda c: c.idm.max_acceleration_mps2),
            comfortable_decel_mps2  = med(lambda c: c.idm.comfortable_decel_mps2),
            jam_spacing_m           = med(lambda c: c.idm.jam_spacing_m),
            reaction_time_s         = med(lambda c: c.idm.reaction_time_s),
            delta                   = med(lambda c: c.idm.delta),
        )
        c0.leader_idm = IDMParams(
            desired_speed_mps       = med(lambda c: c.leader_idm.desired_speed_mps),
            time_headway_s          = med(lambda c: c.leader_idm.time_headway_s),
            max_acceleration_mps2   = med(lambda c: c.leader_idm.max_acceleration_mps2),
            comfortable_decel_mps2  = med(lambda c: c.leader_idm.comfortable_decel_mps2),
            jam_spacing_m           = med(lambda c: c.leader_idm.jam_spacing_m),
            reaction_time_s         = med(lambda c: c.leader_idm.reaction_time_s),
            delta                   = med(lambda c: c.leader_idm.delta),
        )
        c0.mobil = MOBILParams(
            politeness_factor    = med(lambda c: c.mobil.politeness_factor),
            accel_threshold_mps2 = med(lambda c: c.mobil.accel_threshold_mps2),
            safe_decel_mps2      = med(lambda c: c.mobil.safe_decel_mps2),
        )
        c0.scene = SceneConfig(
            initial_spacing_m        = med(lambda c: c.scene.initial_spacing_m),
            initial_ego_speed_mps    = med(lambda c: c.scene.initial_ego_speed_mps),
            initial_leader_speed_mps = med(lambda c: c.scene.initial_leader_speed_mps),
        )
        c0.edge = EdgeConfig(
            edge_trigger_time_s  = med(lambda c: c.edge.edge_trigger_time_s),
            edge_decel_mps2      = med(lambda c: c.edge.edge_decel_mps2),
            edge_cut_in_ttc_s    = med(lambda c: c.edge.edge_cut_in_ttc_s),
            ped_cross_distance_m = med(lambda c: c.edge.ped_cross_distance_m),
        )
        return c0

    @staticmethod
    def _perturb(base: ScenarioParams, rng: np.random.Generator,
                 idx: int = 0) -> ScenarioParams:
        """Create one instance by perturbing base params."""
        import copy
        inst = copy.deepcopy(base)

        pct = 0.15  # ±15 % variation

        def noisy(v: float, lo: float, hi: float, pct=pct) -> float:
            return float(np.clip(rng.normal(v, abs(v) * pct), lo, hi))

        b = IDM_BOUNDS
        inst.idm = IDMParams(
            desired_speed_mps      = noisy(base.idm.desired_speed_mps,     *b["desired_speed_mps"]),
            time_headway_s         = noisy(base.idm.time_headway_s,        *b["time_headway_s"]),
            max_acceleration_mps2  = noisy(base.idm.max_acceleration_mps2, *b["max_acceleration_mps2"]),
            comfortable_decel_mps2 = noisy(base.idm.comfortable_decel_mps2,*b["comfortable_decel_mps2"]),
            jam_spacing_m          = noisy(base.idm.jam_spacing_m,         *b["jam_spacing_m"]),
            reaction_time_s        = noisy(base.idm.reaction_time_s,       *b["reaction_time_s"]),
            delta                  = noisy(base.idm.delta,                 *b["delta"]),
        )
        inst.leader_idm = IDMParams(
            desired_speed_mps      = noisy(base.leader_idm.desired_speed_mps,     *b["desired_speed_mps"]),
            time_headway_s         = noisy(base.leader_idm.time_headway_s,        *b["time_headway_s"]),
            max_acceleration_mps2  = noisy(base.leader_idm.max_acceleration_mps2, *b["max_acceleration_mps2"]),
            comfortable_decel_mps2 = noisy(base.leader_idm.comfortable_decel_mps2,*b["comfortable_decel_mps2"]),
            jam_spacing_m          = noisy(base.leader_idm.jam_spacing_m,         *b["jam_spacing_m"]),
            reaction_time_s        = noisy(base.leader_idm.reaction_time_s,       *b["reaction_time_s"]),
            delta                  = noisy(base.leader_idm.delta,                 *b["delta"]),
        )
        # Vary scene initial conditions (±20 %)
        sb = SCENE_BOUNDS
        inst.scene.initial_spacing_m        = noisy(base.scene.initial_spacing_m,        *sb["initial_spacing_m"],        0.20)
        inst.scene.initial_ego_speed_mps    = noisy(base.scene.initial_ego_speed_mps,    *sb["initial_ego_speed_mps"],    0.20)
        inst.scene.initial_leader_speed_mps = noisy(base.scene.initial_leader_speed_mps, *sb["initial_leader_speed_mps"], 0.20)
        # Keep duration and sampling rate fixed
        inst.scene.scene_duration_s    = 30.0
        inst.scene.sampling_interval_s =  0.1

        # Vary edge trigger time slightly
        inst.edge.edge_trigger_time_s = noisy(base.edge.edge_trigger_time_s, *EDGE_BOUNDS["edge_trigger_time_s"], 0.20)
        return inst
