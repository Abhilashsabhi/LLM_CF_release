#!/usr/bin/env bash
# =============================================================================
# Car-Following Dataset Generator — Complete Run Guide
# =============================================================================
#
# PREREQUISITES
# -------------
#   1. CARLA server running (see bottom of this file)
#   2. Python environment with: numpy scipy pyyaml requests python-dotenv
#   3. CARLA Python API installed:
#        pip install /path/to/carla/PythonAPI/carla/dist/*.whl
#   4. An LLM backend, either:
#        (a) Groq cloud:  echo "GROQ_API_KEY=your_key_here" > .env
#        (b) Local Ollama: ollama serve && ollama pull qwen2.5:14b-instruct
#      Stage 1 REFUSES to run without one — a dataset generated from the
#      hardcoded priors is not an LLM-parameterised dataset.
#
# DATASET SIZE OPTIONS
# --------------------
#   --n-instances 20  →  540 scenes  × 300 frames = 162,000 frames  (quick test)
#   --n-instances 50  → 1350 scenes  × 300 frames = 405,000 frames  (paper-ready)
#   --n-instances 100 → 2700 scenes  × 300 frames = 810,000 frames  (large-scale)
#
#   Each frame = 0.1 s, each scene = 30 s, 10 Hz recording.
#
# =============================================================================

set -e   # stop on first error

PY="${PY:-python3}"                          # interpreter (no bare `python` on this box)
DATASET_ROOT="${DATASET_ROOT:-$PWD/cf_dataset}"   # change to your output path
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
N_INSTANCES="${N_INSTANCES:-50}"                    # scenes per (style × category) combo
N_VOTES=5                         # LLM self-consistency votes
LLM_BACKEND="${LLM_BACKEND:-auto}"          # auto | groq | ollama
MODEL="${MODEL:-qwen2.5:14b-instruct}"      # Groq or Ollama model name
SEED=42

echo "============================================================"
echo " Car-Following Dataset Generator"
echo " Output : $DATASET_ROOT"
echo " Scenes : $(( 3 * 9 * N_INSTANCES )) total"
echo " Frames : $(( 3 * 9 * N_INSTANCES * 300 )) total"
echo "============================================================"

# =============================================================================
# STEP 0 — Verify environment
# =============================================================================

echo ""
echo "[Step 0] Verifying environment ..."

$PY -c "import carla; print('  CARLA Python API OK:', carla.__version__)" || {
    echo "ERROR: CARLA Python API not found."
    echo "Install: pip install /path/to/CARLA/PythonAPI/carla/dist/*.whl"
    exit 1
}

$PY -c "import numpy, scipy, yaml, requests, dotenv; print('  Python deps OK')" || {
    echo "ERROR: Missing Python dependencies."
    echo "Install: pip install numpy scipy pyyaml requests python-dotenv"
    exit 1
}

$PY -c "
from dotenv import load_dotenv; import os; load_dotenv()
from stage1_llm_generator import LLMClient
c = LLMClient(backend='$LLM_BACKEND', model='$MODEL')
if not c.enabled:
    raise SystemExit('  ERROR: no LLM backend reachable (see prerequisites)')
print(f'  LLM OK: {c.backend} / {c.model}')
"

echo "[Step 0] Done."

# =============================================================================
# STEP 1 — Generate LLM behavioral parameters (Stage 1)
# No CARLA needed. Takes ~5-10 min (N_VOTES × 27 LLM calls).
# Output: $DATASET_ROOT/configs/llm_base_params.yaml
#         $DATASET_ROOT/configs/llm_instances.json
# =============================================================================

echo ""
echo "[Step 1] Generating LLM behavioral parameters ..."
echo "  Querying LLM $N_VOTES times × 27 combinations (3 styles × 9 scenarios)"
echo "  Estimated time: 5-15 minutes depending on API latency"

$PY main.py \
    --stage 1 \
    --output "$DATASET_ROOT" \
    --n-instances "$N_INSTANCES" \
    --n-votes "$N_VOTES" \
    --llm-backend "$LLM_BACKEND" \
    --model "$MODEL" \
    --seed "$SEED"

echo "[Step 1] Done. Check: $DATASET_ROOT/configs/llm_base_params.yaml"

# Sanity check — verify file exists and has content
$PY -c "
import yaml
with open('$DATASET_ROOT/configs/llm_base_params.yaml') as f:
    d = yaml.safe_load(f)
styles = list(d.keys())
cats   = list(d[styles[0]].keys())
print(f'  Styles  : {styles}')
print(f'  Scenarios: {len(cats)} — {cats}')
"

# =============================================================================
# STEP 2 — Build scene list (Stage 3)
# No CARLA needed. Instantaneous.
# Output: $DATASET_ROOT/configs/scene_list.json
# =============================================================================

echo ""
echo "[Step 2] Building scene list ..."

$PY main.py \
    --stage 3 \
    --output "$DATASET_ROOT" \
    --n-instances "$N_INSTANCES" \
    --seed "$SEED"

TOTAL_SCENES=$($PY -c "
import json
with open('$DATASET_ROOT/configs/scene_list.json') as f:
    d = json.load(f)
print(len(d))
")
echo "[Step 2] Done. $TOTAL_SCENES scenes in scene_list.json"

# =============================================================================
# STEP 3 — Run CARLA data collection (Stage 2)
# CARLA SERVER MUST BE RUNNING before this step.
#
# For large datasets (n-instances=50+), this step takes many hours.
# You can safely interrupt and resume — already-collected scenes are
# saved to disk. To resume after a crash, filter the scene list
# to skip already-completed scene IDs (see resume script below).
#
# Output: $DATASET_ROOT/sweeps/       (sensor data)
#         $DATASET_ROOT/ground-truth/ (kinematic GT)
#         $DATASET_ROOT/configs/      (per-scene YAML)
#         $DATASET_ROOT/logs/         (CARLA recorder logs)
# =============================================================================

# STRONGLY RECOMMENDED before a full collection: a kinematics-only sweep.
# It runs the identical scene list without the 14 sensors, so a scene costs
# a few hundred kB instead of ~2.6 GB, and Stage 4 can confirm the kinematic
# distributions PASS before terabytes are committed:
#
#   $PY main.py --stage 2 --output "$DATASET_ROOT" --no-sensors --no-record
#   $PY main.py --stage 4 --output "$DATASET_ROOT"
#
# Stage 2 collects one map per subprocess and checkpoints after every scene,
# so an interrupted run resumes by re-invoking the same command.

echo ""
echo "[Step 3] Starting CARLA data collection ..."
echo "  Make sure CARLA server is running at $CARLA_HOST:$CARLA_PORT"
echo "  Estimated time: ~$(( TOTAL_SCENES * 35 / 3600 )) hours"
echo "  (30 s/scene + ~5 s overhead = 35 s/scene)"
echo ""
echo "  To verify CARLA is reachable:"
echo "  $PY -c \"import carla; c=carla.Client('$CARLA_HOST',$CARLA_PORT); print(c.get_server_version())\""
echo ""

# Optional: verify CARLA connection before starting
$PY -c "
import carla, sys
try:
    c = carla.Client('$CARLA_HOST', $CARLA_PORT)
    c.set_timeout(10.0)
    v = c.get_server_version()
    print(f'  CARLA server OK: version {v}')
except Exception as e:
    print(f'  ERROR: Cannot connect to CARLA — {e}')
    sys.exit(1)
"

$PY main.py \
    --stage 2 \
    --output "$DATASET_ROOT" \
    --carla-host "$CARLA_HOST" \
    --carla-port "$CARLA_PORT"

echo "[Step 3] Done. Sensor data saved to $DATASET_ROOT/sweeps/"

# =============================================================================
# STEP 4 — Statistical validation + info assembly (Stage 4)
# No CARLA needed. Validates GT distributions vs NGSIM / HighD.
#
# NOTE: this MUST run before Step 5 — Stage 5 reads the scene/frame structure
#       from infos/cf_infos_{train,val,test}.json, which Stage 4 writes.
#
# Output: $DATASET_ROOT/infos/cf_infos_{train,val,test}.json
#         $DATASET_ROOT/infos/validation_report.json
# =============================================================================

echo ""
echo "[Step 4] Running dataset validation and info assembly ..."

$PY main.py \
    --stage 4 \
    --output "$DATASET_ROOT"

echo "[Step 4] Done. Validation report: $DATASET_ROOT/infos/validation_report.json"

# =============================================================================
# STEP 5 — Generate VQA pairs (Stage 5)
# No CARLA needed. Reads infos/ + saved GT + sensor paths.
# Run template mode first (free, fast) then optionally LLM mode.
#
# Output: $DATASET_ROOT/vqa/{scene_id}/frame-NNNN-vqa.json
# =============================================================================

echo ""
echo "[Step 5a] Generating template VQA pairs (all frames) ..."

$PY stage5_vqa_generator.py \
    --dataset "$DATASET_ROOT" \
    --mode template \
    --qa-types A B C D E \
    --workers 8

TEMPLATE_COUNT=$(find "$DATASET_ROOT/vqa" -name "*-vqa.json" | wc -l)
echo "[Step 5a] Done. $TEMPLATE_COUNT VQA records written."

# Optional: LLM-enriched answers for 1-in-10 frames — needs a Groq API key.
echo ""
if [ -z "${GROQ_API_KEY:-}" ] && ! grep -qs GROQ_API_KEY .env; then
    echo "[Step 5b] Skipped — GROQ_API_KEY not set (template VQA already written)."
else
    echo "[Step 5b] Generating LLM VQA pairs (sampled 1 in 10 frames) ..."
    echo "  ~$(( TOTAL_SCENES * 30 )) API calls."
    $PY stage5_vqa_generator.py \
        --dataset "$DATASET_ROOT" \
        --mode llm \
        --qa-types A B D \
        --sample-rate 10 \
        --workers 4
    echo "[Step 5b] Done."
fi

# Print validation summary
$PY -c "
import json
with open('$DATASET_ROOT/infos/validation_report.json') as f:
    r = json.load(f)
print()
print('  Validation result:', 'PASS' if r.get('overall_pass') else 'FAIL')
print(f'  Total frames: {r[\"total_frames\"]:,}')
print(f'  Safety-critical frames (TTC<3s): {r[\"ttc\"][\"n_critical_frames\"]:,}')
print(f'  Collision rate: {100*r[\"collisions\"][\"rate\"]:.3f}%')
"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

# =============================================================================
# STEP 6 — Datasheet + reproducibility metadata (for the paper)
# Writes infos/provenance.json (versions, seeds, calibration constants) and
# DATASHEET.md (composition, collection process, splits, limitations).
# =============================================================================

echo ""
echo "[Step 6] Writing datasheet and provenance metadata ..."

$PY dataset_card.py \
    --dataset "$DATASET_ROOT" \
    --carla-host "$CARLA_HOST" \
    --carla-port "$CARLA_PORT"

echo "[Step 6] Done."

echo ""
echo "============================================================"
echo " Dataset generation complete."
echo ""
echo " Output structure:"
echo "   $DATASET_ROOT/"
echo "   ├── configs/          per-scene YAML + scene_list.json"
echo "   ├── ground-truth/det/ kinematic GT (.npy, 20 fields)"
echo "   ├── infos/            split JSONs + validation report"
echo "   ├── logs/             CARLA recorder logs"
echo "   ├── sweeps/           sensor data (RGB × 6, LiDAR, Radar × 4, IMU, GNSS)"
echo "   └── vqa/              VQA JSON pairs (5 types per frame)"
echo ""
echo " Next steps:"
echo "   1. Train a VQA model on infos/cf_infos_train.json + vqa/"
echo "   2. Evaluate on real NGSIM/HighD test set for cross-dataset generalisation"
echo "   3. Compare against GenFollower and KIDL baselines"
echo "============================================================"

# =============================================================================
# HOW TO START CARLA SERVER
# =============================================================================
# In a separate terminal, before running Step 3:
#
#   cd /path/to/CARLA
#   ./CarlaUE4.sh -RenderOffScreen -nosound -fps=20 -carla-port=2000
#
# For Docker (SimBEV style):
#   docker run --runtime=nvidia --gpus all --network=host \
#       -v /path/to/CARLA:/home/carla \
#       -v /path/to/dataset:/dataset \
#       simbev:develop /home/carla/CarlaUE4.sh \
#       -RenderOffScreen -nosound -fps=20 -carla-port=2000
#
# Verify CARLA is ready:
#   python -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
#
# =============================================================================
# RESUME AFTER CRASH (Stage 2)
# =============================================================================
# If CARLA crashes mid-collection, resume with only the remaining scenes:
#
#   python -c "
#   import json
#   from pathlib import Path
#   completed = {p.stem.split('-frame')[0]
#                for p in Path('/dataset/ground-truth/det').rglob('*.npy')}
#   with open('/dataset/configs/scene_list.json') as f:
#       scenes = json.load(f)
#   remaining = [s for s in scenes if s['scene_id'] not in completed]
#   with open('/dataset/configs/scene_list_remaining.json','w') as f:
#       json.dump(remaining, f, indent=2)
#   print(f'Completed: {len(completed)}, Remaining: {len(remaining)}')
#   "
#
#   Then edit main.py's run_stage2() to load scene_list_remaining.json,
#   or pass --instances /dataset/configs/scene_list_remaining.json
#
# =============================================================================
