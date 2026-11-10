"""
main.py — Car-Following Synthetic Dataset Generator v2
=======================================================

Dataset size summary
--------------------
  n_instances=20  →  3 × 9 × 20 =  540 scenes  ×  300 frames  = 162,000 frames
  n_instances=50  →  3 × 9 × 50 = 1,350 scenes  × 300 frames  = 405,000 frames
  n_instances=100 →  3 × 9 × 100 = 2,700 scenes  × 300 frames  = 810,000 frames

Each frame = 30 s / 300 frames = 0.1 s (10 Hz), 30 s per scene.

Usage examples
--------------
  # Generate params only (no CARLA):
  python main.py --stage 1 --output /dataset --n-instances 20

  # Full pipeline:
  python main.py --stage all --output /dataset \\
      --n-instances 20 --n-votes 5 \\
      --carla-host localhost --carla-port 2000

  # Stage 1+3 only (build scene list, save configs — no CARLA):
  python main.py --stage 13 --output /dataset --n-instances 20

  # Stage 2 only (run CARLA, params already generated):
  python main.py --stage 2 --output /dataset \\
      --carla-host localhost --carla-port 2000

  # Stage 4 only (validate existing collected dataset):
  python main.py --stage 4 --output /dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Car-Following Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stage",       default="all",
                   help="Stages: 1|2|3|4|13|all  (default: all)")
    p.add_argument("--output",      default="/dataset",
                   help="Dataset root directory  (default: /dataset)")
    p.add_argument("--params",      default=None,
                   help="Path to pre-generated base params YAML (skips Stage 1)")
    p.add_argument("--instances",   default=None,
                   help="Path to pre-generated instances JSON (skips Stage 1+3)")
    p.add_argument("--n-instances", type=int, default=20,
                   help="Scenes per (style × category) combo  (default: 20)")
    p.add_argument("--n-votes",     type=int, default=5,
                   help="LLM self-consistency votes per base scenario (default: 5)")
    p.add_argument("--model",       default=None,
                   help="LLM model name (default: backend-specific)")
    p.add_argument("--llm-backend", default="auto",
                   choices=["auto", "groq", "ollama"],
                   help="LLM backend. auto = Groq if GROQ_API_KEY is set, "
                        "else a local Ollama server (default: auto)")
    p.add_argument("--llm-base-url", default=None,
                   help="Override the chat-completions URL")
    p.add_argument("--api-key",     default=None,
                   help="Groq API key  (falls back to GROQ_API_KEY env var)")
    p.add_argument("--carla-host",  default="localhost")
    p.add_argument("--carla-port",  type=int, default=2000)
    p.add_argument("--carla-timeout", type=float, default=120.0,
                   help="CARLA client timeout in seconds (default: 120)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-record",   action="store_true",
                   help="Disable CARLA recorder (skip .log files)")
    p.add_argument("--no-sensors",  action="store_true",
                   help="Kinematics-only: skip all 14 sensors. ~2.6 GB -> a few "
                        "hundred kB per scene; use to validate distributions "
                        "over the whole scene list before a full collection.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print scene list without running CARLA")
    p.add_argument("--no-group-by-map", action="store_true",
                   help="Collect scenes in list order instead of grouping by "
                        "map (grouping avoids one load_world() per scene)")
    p.add_argument("--only-map",    default=None,
                   help="Collect only scenes on this CARLA map. Used by the "
                        "per-map subprocess runner; also useful manually.")
    p.add_argument("--min-free-gb", type=float, default=0.0,
                   help="Pause between scenes while free space on the output "
                        "filesystem is below this. Used with local staging: "
                        "generation (~39 s/scene) outruns the transfer to the "
                        "destination drive (~108 s/scene), so without "
                        "backpressure the staging disk fills and the run dies.")
    p.add_argument("--skip-existing", default=None,
                   help="Dataset root already holding collected scenes; any "
                        "scene with a complete ground-truth shard there is "
                        "skipped. Makes the collection resumable across "
                        "restarts of the CARLA server.")
    p.add_argument("--exclude-categories", nargs="*", default=None,
                   help="Scenario categories to leave out of the scene list "
                        "(e.g. --exclude-categories cut_in). Used to ship "
                        "validated categories while a broken one is fixed; "
                        "the excluded ones can be collected later into the "
                        "same dataset root.")
    p.add_argument("--no-map-subprocess", action="store_true",
                   help="Run every map in this process. Faster to start, but "
                        "a load_world() crash then kills the whole run.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage1(args) -> Dict:
    from stage1_llm_generator import LLMParameterGenerator

    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  Stage 1 — LLM Parameter Generation  ║")
    logger.info("╚═══════════════════════════════════════╝")

    gen     = LLMParameterGenerator(api_key=args.api_key,
                                    model=args.model, n_votes=args.n_votes,
                                    backend=args.llm_backend,
                                    base_url=args.llm_base_url)
    if not gen.client.enabled:
        raise SystemExit(
            "Refusing to generate a 'LLM-parameterised' dataset with no LLM.\n"
            "  Either  export GROQ_API_KEY=...\n"
            "  or      ollama serve && ollama pull qwen2.5:7b-instruct\n"
            "  (use --llm-backend to force a choice)")
    base    = gen.generate_all_base()
    outpath = str(Path(args.output) / "configs" / "llm_base_params.yaml")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    gen.save_base(base, outpath)

    # Record HOW the parameters were generated. Without this the dataset's
    # central claim — that the behavioural parameters are LLM-generated with
    # self-consistency voting — is unverifiable by anyone who downloads it.
    import datetime as _dt
    meta = {
        "backend":        gen.client.backend,
        "model":          gen.client.model,
        "base_url":       gen.client.base_url,
        "n_votes":        args.n_votes,
        "temperature":    gen.client.temperature,
        "seed":           args.seed,
        "n_instances":    args.n_instances,
        "generated_utc":  _dt.datetime.now(_dt.timezone.utc).isoformat(
                              timespec="seconds"),
        "prompt_system":  gen.builder._system(),
    }
    meta_path = Path(args.output) / "configs" / "llm_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("LLM provenance → %s (%s / %s, %d votes)",
                meta_path, meta["backend"], meta["model"], args.n_votes)

    instances = gen.sample_instances(base,
                                      n_instances=args.n_instances,
                                      seed=args.seed)
    inst_path = str(Path(args.output) / "configs" / "llm_instances.json")
    _save_instances(instances, inst_path)
    logger.info("Stage 1 done. %d base params × %d instances = %d total.",
                9 * 3, args.n_instances, 9 * 3 * args.n_instances)
    return {"base": base, "instances": instances}


def run_stage3(args, instances=None) -> List:
    from stage1_llm_generator import LLMParameterGenerator, ScenarioParams
    from stage3_scenario_manager import ScenarioManager

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  Stage 3 — Scene List Construction  ║")
    logger.info("╚══════════════════════════════════════╝")

    if instances is None:
        # Load from disk
        inst_path = args.instances or str(
            Path(args.output) / "configs" / "llm_instances.json"
        )
        if not Path(inst_path).exists():
            raise FileNotFoundError(
                f"Instances file not found at {inst_path}. Run Stage 1 first."
            )
        instances = _load_instances(inst_path)

    mgr    = ScenarioManager(seed=args.seed)
    scenes = mgr.build_from_instances(instances)

    if args.exclude_categories:
        drop   = set(args.exclude_categories)
        before = len(scenes)
        scenes = [s for s in scenes if s.scenario_category not in drop]
        logger.warning("Excluded %s: %d -> %d scenes.",
                       ", ".join(sorted(drop)), before, len(scenes))

    scene_list_path = Path(args.output) / "configs" / "scene_list.json"
    scene_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scene_list_path, "w") as f:
        json.dump([s.to_dict() for s in scenes], f, indent=2)

    n_frames = len(scenes) * (scenes[0].n_recorded_frames if scenes else 300)
    logger.info("Stage 3 done. %d scenes → %d total frames → %s",
                len(scenes), n_frames, scene_list_path)

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"  {'Driving style':<18}  {'Scenario category':<25}  Count")
    print(f"{'─'*55}")
    from collections import Counter
    cnt = Counter((s.driving_style, s.scenario_category) for s in scenes)
    for (style, cat), n in sorted(cnt.items()):
        print(f"  {style:<18}  {cat:<25}  {n}")
    print(f"{'─'*55}")
    print(f"  TOTAL: {len(scenes)} scenes × {scenes[0].n_recorded_frames} frames"
          f" = {len(scenes) * scenes[0].n_recorded_frames:,} frames @ 10 Hz")
    print(f"  Scene duration: {scenes[0].scene_duration_s:.0f} s "
          f"| Sampling: {scenes[0].sampling_interval_s:.1f} s\n")

    return scenes


def run_stage2(args, scenes: Optional[List] = None) -> List[Dict]:
    from stage2_carla_executor import CarlaExecutor

    logger.info("╔═════════════════════════════════════╗")
    logger.info("║  Stage 2 — CARLA Data Collection   ║")
    logger.info("╚═════════════════════════════════════╝")

    if scenes is None:
        from stage3_scenario_manager import SceneSpec
        scene_list_path = Path(args.output) / "configs" / "scene_list.json"
        if not scene_list_path.exists():
            raise FileNotFoundError(
                "scene_list.json not found. Run Stage 3 first."
            )
        with open(scene_list_path) as f:
            raw = json.load(f)
        scenes = [_dict_to_spec(d) for d in raw]

    if args.skip_existing:
        done_root = Path(args.skip_existing) / "ground-truth" / "det"
        done = set()
        if done_root.exists():
            for d in done_root.iterdir():
                # A shard is complete when it holds one file per frame.
                if d.is_dir() and sum(1 for _ in d.glob("*.npy")) >= 300:
                    done.add(d.name)
        if done:
            before = len(scenes)
            scenes = [s for s in scenes if s.scene_id not in done]
            logger.info("Resume: %d scene(s) already collected, %d -> %d to do.",
                        len(done), before, len(scenes))

    if args.only_map:
        scenes = [s for s in scenes if s.carla_map == args.only_map]
        logger.info("Restricted to map %s: %d scene(s).",
                    args.only_map, len(scenes))

    if args.dry_run:
        logger.info("[DRY RUN] Would execute %d scenes.", len(scenes))
        return []

    # Group scenes by map. load_world() is the slowest and least reliable
    # CARLA call in the pipeline; grouping turns one load per scene into one
    # per map. Split assignment is stratified by (style, category, regime),
    # so collection order carries no information and reordering is free.
    if not args.no_group_by_map:
        scenes = sorted(scenes, key=lambda s: s.carla_map)

    maps = sorted({s.carla_map for s in scenes})
    if (not args.only_map and not args.no_map_subprocess and len(maps) > 1):
        return _run_stage2_per_map(args, scenes, maps)

    return _collect_scenes(args, scenes)


def _run_stage2_per_map(args, scenes: List, maps: List[str]) -> List[Dict]:
    """Collect each map in its own subprocess.

    Repeated `client.load_world()` calls in a single long-lived process
    segfault the CARLA client — reproducibly, on the first map CHANGE, while
    scenes that reuse an already-loaded map run fine. A fresh process loads a
    map reliably, so each map gets its own child: one load_world per child,
    at startup, in clean state.

    Children append to configs/stage2_records.json, so a child that dies
    costs only its own map and the run continues.
    """
    import subprocess

    rec_path = Path(args.output) / "configs" / "stage2_records.json"
    logger.info("Collecting %d scenes across %d maps, one subprocess per map.",
                len(scenes), len(maps))

    passthrough: List[str] = [
        "--output", str(args.output),
        "--carla-host", str(args.carla_host),
        "--carla-port", str(args.carla_port),
        "--carla-timeout", str(args.carla_timeout),
    ]
    for flag, on in (("--no-record", args.no_record),
                     ("--no-sensors", args.no_sensors)):
        if on:
            passthrough.append(flag)

    failed: List[str] = []
    for i, cmap in enumerate(maps, 1):
        n = sum(1 for s in scenes if s.carla_map == cmap)
        logger.info("── [map %d/%d] %s — %d scene(s)", i, len(maps), cmap, n)
        cmd = ([sys.executable, str(Path(__file__).resolve()),
                "--stage", "2", "--only-map", cmap] + passthrough)
        # One retry: a child killed by a signal usually leaves the world in
        # synchronous mode, and the next process now recovers it at connect
        # time, so an immediate retry generally succeeds.
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            logger.warning("Map %s exited with code %d — retrying once.",
                           cmap, rc)
            rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append(cmap)
            logger.error("Map %s failed twice (code %d) — continuing with "
                         "the next map; rerun to fill the gap.", cmap, rc)

    if failed:
        logger.warning("Maps that did not finish cleanly: %s", ", ".join(failed))

    if rec_path.exists():
        with open(rec_path) as f:
            all_records = json.load(f)
    else:
        all_records = []
    logger.info("Stage 2 done. %d scene(s) collected → %s",
                len(all_records), rec_path)
    return all_records


def _collect_scenes(args, scenes: List) -> List[Dict]:
    from stage2_carla_executor import CarlaExecutor

    executor = CarlaExecutor(
        dataset_root = args.output,
        host         = args.carla_host,
        port         = args.carla_port,
        timeout      = args.carla_timeout,
        record_log   = not args.no_record,
        collect_sensors = not args.no_sensors,
    )

    # Checkpointed after every scene: a CARLA crash mid-run can kill the
    # process outright (uncatchable C++ terminate), so nothing may be lost.
    rec_path = Path(args.output) / "configs" / "stage2_records.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)

    # Carry over scenes collected by an earlier (interrupted) run, so
    # resuming with a filtered scene list extends the dataset instead of
    # replacing it. Scenes about to be re-run are dropped and re-added.
    def _await_space() -> None:
        """Block while the staging filesystem is nearly full."""
        if args.min_free_gb <= 0:
            return
        import shutil as _sh
        waited = 0
        while True:
            free_gb = _sh.disk_usage(str(args.output)).free / 1e9
            if free_gb >= args.min_free_gb:
                if waited:
                    logger.info("Staging drained (%.0f GB free) after %d s.",
                                free_gb, waited)
                return
            if waited % 300 == 0:
                logger.warning("Waiting for the mover: %.0f GB free, need %.0f. "
                               "Generation outruns the destination drive; this "
                               "is backpressure, not a fault.",
                               free_gb, args.min_free_gb)
            time.sleep(30)
            waited += 30

    all_records: List[Dict] = []
    if rec_path.exists():
        try:
            with open(rec_path) as f:
                previous = json.load(f)
            rerun = {s.scene_id for s in scenes}
            all_records = [r for r in previous if r.get("scene_id") not in rerun]
            if all_records:
                logger.info("Resuming: kept %d scene(s) from a previous run.",
                            len(all_records))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s) — starting fresh.",
                           rec_path, exc)
    for i, spec in enumerate(scenes):
        _await_space()
        logger.info("[%d / %d] %s", i + 1, len(scenes), spec.scene_id)
        try:
            fr = executor.run_scene(spec)
            all_records.append({
                "scene_id":   spec.scene_id,
                "scene_info": spec.to_dict(),
                "scene_data": fr,
            })
            with open(rec_path, "w") as f:
                json.dump(all_records, f, indent=2)
        except Exception as exc:
            logger.error("Scene %s failed: %s", spec.scene_id, exc, exc_info=True)

    logger.info("Stage 2 done. %d / %d scenes collected → %s",
                len(all_records), len(scenes), rec_path)
    return all_records


def run_stage4(args, scene_records: Optional[List] = None) -> Dict:
    from stage4_validator import DatasetValidator, DatasetAssembler

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Stage 4 — Validation & Info Assembly   ║")
    logger.info("╚══════════════════════════════════════════╝")

    if scene_records is None:
        # Standalone run — reload what Stage 2 persisted.
        rec_path = Path(args.output) / "configs" / "stage2_records.json"
        if rec_path.exists():
            with open(rec_path) as f:
                scene_records = json.load(f)
            logger.info("Loaded %d scene records from %s",
                        len(scene_records), rec_path)
        else:
            logger.warning("%s not found — infos/ files will not be written "
                           "(Stage 5 VQA needs them).", rec_path)

    assembler = DatasetAssembler(args.output)
    if scene_records:
        # Town-disjoint splits are opt-in via CF_TOWN_DISJOINT=1: they change
        # what a benchmark measures, so the choice is explicit and recorded in
        # split_policy rather than silently applied.
        assembler.write_info_files(
            scene_records,
            town_disjoint=os.environ.get("CF_TOWN_DISJOINT", "") == "1")

    report = DatasetValidator(args.output).validate()
    logger.info("Stage 4 done.")
    return report


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _save_instances(instances, path: str) -> None:
    """Save [style][category] → List[ScenarioParams.to_dict()] as JSON."""
    from dataclasses import asdict
    d: Dict = {}
    for style, cats in instances.items():
        d[style] = {}
        for cat, lst in cats.items():
            d[style][cat] = [p.to_dict() for p in lst]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    logger.info("Instances saved to %s", path)


def _load_instances(path: str) -> Dict:
    from stage1_llm_generator import (
        ScenarioParams, IDMParams, MOBILParams, SceneConfig, EdgeConfig,
    )
    with open(path) as f:
        raw = json.load(f)
    out: Dict = {}
    for style, cats in raw.items():
        out[style] = {}
        for cat, lst in cats.items():
            params_list = []
            for d in lst:
                params_list.append(ScenarioParams(
                    scenario_id      = d.get("scenario_id", ""),
                    driving_style    = style,
                    scenario_category= cat,
                    idm              = IDMParams(**d["idm"]),
                    leader_idm       = IDMParams(**d.get("leader_idm", d["idm"])),
                    mobil            = MOBILParams(**d["mobil"]),
                    scene            = SceneConfig(**{
                        k: v for k, v in d["scene"].items()
                        if k in SceneConfig.__dataclass_fields__
                    }),
                    edge             = EdgeConfig(**d["edge"]),
                    road_grade_deg   = d.get("road_grade_deg", 0.0),
                    llm_reasoning    = d.get("llm_reasoning", ""),
                    consistency_votes= d.get("consistency_votes", 0),
                ))
            out[style][cat] = params_list
    return out


def _dict_to_spec(d: Dict):
    from stage3_scenario_manager import SceneSpec, InjectionEvent, PedestrianConfig
    inj_d = d.get("injection", {})
    ped_d = d.get("pedestrian", {})
    spec  = SceneSpec(
        scene_id                  = d["scene_id"],
        instance_idx              = d.get("instance_idx", 0),
        driving_style             = d["driving_style"],
        scenario_category         = d["scenario_category"],
        carla_map                 = d["carla_map"],
        regime                    = d.get("regime", "urban"),
        weather                   = d.get("weather", {}),
        road_grade_deg            = d.get("road_grade_deg", 0.0),
        idm_params                = d.get("idm_params", {}),
        leader_idm_params         = d.get("leader_idm_params", {}),
        mobil_params              = d.get("mobil_params", {}),
        initial_spacing_m         = d.get("initial_spacing_m", 20.0),
        initial_ego_speed_mps     = d.get("initial_ego_speed_mps", 15.0),
        initial_leader_speed_mps  = d.get("initial_leader_speed_mps", 15.0),
        scene_duration_s          = d.get("scene_duration_s", 30.0),
        sampling_interval_s       = d.get("sampling_interval_s", 0.1),
        physics_fps               = d.get("physics_fps", 20),
        llm_reasoning             = d.get("llm_reasoning", ""),
        consistency_votes         = d.get("consistency_votes", 0),
    )
    if inj_d:
        spec.injection = InjectionEvent(**{
            k: v for k, v in inj_d.items()
            if k in InjectionEvent.__dataclass_fields__
        })
    if ped_d:
        spec.pedestrian = PedestrianConfig(**{
            k: v for k, v in ped_d.items()
            if k in PedestrianConfig.__dataclass_fields__
        })
    return spec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    stages = args.stage.lower().strip()

    base_params   = None
    instances     = None
    scenes        = None
    scene_records = None

    if stages in ("1", "13", "all"):
        r1 = run_stage1(args)
        base_params = r1["base"]
        instances   = r1["instances"]

    if stages in ("3", "13", "all"):
        scenes = run_stage3(args, instances=instances)

    if stages in ("2", "all"):
        # scenes is None here only for a standalone `--stage 2`; in that case
        # run_stage2 loads configs/scene_list.json as-is, so a hand-filtered
        # or resumed scene list is honoured instead of being regenerated.
        scene_records = run_stage2(args, scenes)

    if stages in ("4", "all"):
        run_stage4(args, scene_records=scene_records)

    logger.info("All requested stages complete.")


if __name__ == "__main__":
    main()
