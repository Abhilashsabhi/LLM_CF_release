#!/usr/bin/env bash
# =============================================================================
# LLM_CF_v2 — Master Orchestration Script
# =============================================================================
#
# Runs the complete LLM_CF_v2 pipeline end-to-end:
#   Phase A : Data collection    (run_pipeline.sh)
#   Phase B : Post-processing    (validate, export, VQA v2)
#   Phase C : Model training     (spacing CNN, VLM fine-tune)
#   Phase D : Evaluation         (CF benchmark, VQA, zero-shot, multimodal)
#   Phase E : Adaptation study   (run_adaptation_v3b.sh)
#   Phase F : Paper tables/figs  (make_*.py)
#
# USAGE
# -----
#   bash run_all.sh
#   DATASET_ROOT=/mnt/data/llm_cf_v2 N_INSTANCES=50 bash run_all.sh
#
# Each phase can be skipped by setting the corresponding SKIP_* variable:
#   SKIP_COLLECTION=1 SKIP_TRAINING=1 bash run_all.sh
#
# PREREQUISITES
# -------------
#   1. CARLA 0.9.15 server running (see Phase A comment below)
#   2. Python 3.8+ environment with requirements.txt installed
#   3. CARLA Python API: pip install /path/to/CARLA/PythonAPI/carla/dist/*.whl
#   4. LLM backend: export GROQ_API_KEY=... OR ollama serve && ollama pull qwen2.5:14b-instruct
#   5. GPU with >= 8 GB VRAM (for fine-tuning and VLM eval)
#   6. ~2 TB free disk space for the full 1350-scene dataset
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURABLE VARIABLES
# =============================================================================

PY="${PY:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset paths
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/cf_dataset}"
VQA_V2_DIR="${VQA_V2_DIR:-$DATASET_ROOT/vqa_v2}"
CF_ARRAYS_DIR="${CF_ARRAYS_DIR:-$DATASET_ROOT/cf_arrays}"
SPACING_CACHE_DIR="${SPACING_CACHE_DIR:-$DATASET_ROOT/spacing_cache}"
VLM_FT_DATA_DIR="${VLM_FT_DATA_DIR:-$DATASET_ROOT/vlm_ft_data_v2}"

# Experiment output directories
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
ADAPT_OUT_DIR="${ADAPT_OUT_DIR:-$RESULTS_DIR/adapt_v3b}"
SPACING_MODEL_DIR="${SPACING_MODEL_DIR:-$RESULTS_DIR/spacing_cnn}"
VLM_ADAPTER_DIR="${VLM_ADAPTER_DIR:-$RESULTS_DIR/vlm_adapter}"

# CARLA connection
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"

# Dataset scale
N_INSTANCES="${N_INSTANCES:-50}"     # 50 → 1350 scenes (paper scale)

# LLM backend
LLM_BACKEND="${LLM_BACKEND:-auto}"   # auto | groq | ollama
MODEL="${MODEL:-qwen2.5:14b-instruct}"

# Skip flags (set to 1 to skip a phase)
SKIP_COLLECTION="${SKIP_COLLECTION:-0}"
SKIP_POSTPROC="${SKIP_POSTPROC:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ADAPTATION="${SKIP_ADAPTATION:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

# =============================================================================
# HELPERS
# =============================================================================

log()  { echo "[$(date '+%F %T')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }
skip() { log "Skipping: $*"; }

require_file() {
    [ -f "$1" ] || die "Required file not found: $1"
}

require_dir() {
    [ -d "$1" ] || die "Required directory not found: $1"
}

check_python_deps() {
    log "Checking Python dependencies ..."
    $PY -c "import numpy, scipy, yaml, requests, dotenv, PIL, tqdm, torch, transformers" \
        || die "Missing Python dependencies. Run: pip install -r requirements.txt"
    log "Python dependencies OK."
}

check_carla_api() {
    $PY -c "import carla; print('  CARLA Python API:', carla.__version__)" \
        || die "CARLA Python API not found. Install: pip install /path/to/CARLA/PythonAPI/carla/dist/*.whl"
}

check_carla_server() {
    log "Checking CARLA server at $CARLA_HOST:$CARLA_PORT ..."
    $PY -c "
import carla, sys
try:
    c = carla.Client('$CARLA_HOST', $CARLA_PORT)
    c.set_timeout(10.0)
    print('  CARLA server version:', c.get_server_version())
except Exception as e:
    print(f'  ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" || die "Cannot reach CARLA server. Start it before running Phase A."
}

mkdir -p "$RESULTS_DIR" "$ADAPT_OUT_DIR"

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================

log "=== LLM_CF_v2 Master Pipeline ==="
log "Dataset root : $DATASET_ROOT"
log "Results dir  : $RESULTS_DIR"
log ""

check_python_deps

# =============================================================================
# PHASE A — DATA COLLECTION
# Delegates to run_pipeline.sh which handles Stages 1-5 (LLM params, scene
# list, CARLA collection, validation, VQA generation).
# CARLA 0.9.15 server must be running before this phase.
# =============================================================================

if [ "$SKIP_COLLECTION" = "1" ]; then
    skip "Phase A — data collection (SKIP_COLLECTION=1)"
else
    log ""
    log "=== PHASE A: Data Collection ==="
    log "This calls run_pipeline.sh which runs all 5 pipeline stages."
    log "Estimated time: ~13 hours for N_INSTANCES=50 (1350 scenes)."
    log ""
    log "CARLA server must be running at $CARLA_HOST:$CARLA_PORT"
    log "Start it with:  ./CarlaUE4.sh -RenderOffScreen -nosound -fps=20 -carla-port=$CARLA_PORT"
    log ""

    check_carla_api
    check_carla_server

    # Export variables so run_pipeline.sh picks them up
    export PY DATASET_ROOT CARLA_HOST CARLA_PORT N_INSTANCES LLM_BACKEND MODEL

    bash "$SCRIPT_DIR/run_pipeline.sh"
    log "Phase A complete."
fi

# =============================================================================
# PHASE B — POST-PROCESSING
# Validates the collected dataset against real car-following data (Waymo/HighD),
# exports car-following arrays for the benchmark, and generates VQA v2 pairs.
# No CARLA needed.
# =============================================================================

if [ "$SKIP_POSTPROC" = "1" ]; then
    skip "Phase B — post-processing (SKIP_POSTPROC=1)"
else
    log ""
    log "=== PHASE B: Post-Processing ==="

    require_dir "$DATASET_ROOT"

    # B1: Validate against real car-following data (Waymo / HighD)
    log "[B1] Validating against real-world car-following distributions ..."
    $PY "$SCRIPT_DIR/validate_against_real.py" \
        --dataset "$DATASET_ROOT" \
        --out "$RESULTS_DIR/fidelity_report.json"
    log "[B1] Done → $RESULTS_DIR/fidelity_report.json"

    # B2: Export car-following arrays (train / val / test splits)
    log "[B2] Exporting car-following arrays ..."
    $PY "$SCRIPT_DIR/export_cf_arrays.py" \
        --dataset "$DATASET_ROOT" \
        --out "$CF_ARRAYS_DIR"
    log "[B2] Done → $CF_ARRAYS_DIR"

    # B3: Build VQA v2 fine-tuning corpus
    log "[B3] Building VLM fine-tuning corpus (VQA v2) ..."
    $PY "$SCRIPT_DIR/build_vlm_ft_data_v2.py" \
        --vqa "$VQA_V2_DIR" \
        --out "$VLM_FT_DATA_DIR"
    log "[B3] Done → $VLM_FT_DATA_DIR"

    # B4: Build spacing perception cache
    log "[B4] Building spacing perception cache ..."
    $PY "$SCRIPT_DIR/build_spacing_cache.py" \
        --dataset "$DATASET_ROOT" \
        --out "$SPACING_CACHE_DIR"
    log "[B4] Done → $SPACING_CACHE_DIR"

    # B5: Build multimodal alignment cache
    log "[B5] Building multimodal alignment cache ..."
    $PY "$SCRIPT_DIR/build_multimodal_cache.py" \
        --dataset "$DATASET_ROOT" \
        --out "$DATASET_ROOT/multimodal_cache"
    log "[B5] Done → $DATASET_ROOT/multimodal_cache"

    log "Phase B complete."
fi

# =============================================================================
# PHASE C — MODEL TRAINING
# Trains the supervised spacing CNN and LoRA-adapts Qwen2.5-VL.
# Requires GPU with >= 8 GB VRAM.
# =============================================================================

if [ "$SKIP_TRAINING" = "1" ]; then
    skip "Phase C — model training (SKIP_TRAINING=1)"
else
    log ""
    log "=== PHASE C: Model Training ==="

    require_dir "$SPACING_CACHE_DIR"
    require_dir "$VLM_FT_DATA_DIR"

    # C1: Train spacing perception CNN
    log "[C1] Training supervised spacing CNN ..."
    $PY "$SCRIPT_DIR/train_spacing_cnn.py" \
        --cache "$SPACING_CACHE_DIR" \
        --out "$SPACING_MODEL_DIR"
    log "[C1] Done → $SPACING_MODEL_DIR"

    log "Phase C complete."
    log "(VLM fine-tuning is run separately via run_adaptation_v3b.sh in Phase E.)"
fi

# =============================================================================
# PHASE D — EVALUATION
# Evaluates CF benchmark, VQA quality, zero-shot VLM, and multimodal alignment.
# =============================================================================

if [ "$SKIP_EVAL" = "1" ]; then
    skip "Phase D — evaluation (SKIP_EVAL=1)"
else
    log ""
    log "=== PHASE D: Evaluation ==="

    require_dir "$CF_ARRAYS_DIR"
    require_dir "$VQA_V2_DIR"

    # D1: Car-following benchmark
    log "[D1] Running car-following benchmark ..."
    $PY "$SCRIPT_DIR/eval_carfollowing_v2.py" \
        --arrays "$CF_ARRAYS_DIR" \
        --out "$RESULTS_DIR/cf_bench.json"
    log "[D1] Done → $RESULTS_DIR/cf_bench.json"

    # D2: VQA benchmark (unadapted baseline)
    log "[D2] Running VQA benchmark (zero-shot) ..."
    $PY "$SCRIPT_DIR/eval_qwenvl_v2.py" \
        --qa "$VQA_V2_DIR" \
        --root "$DATASET_ROOT" \
        --out "$RESULTS_DIR/vqa_zeroshot.json" \
        --split test
    log "[D2] Done → $RESULTS_DIR/vqa_zeroshot.json"

    # D3: Zero-shot VLM evaluation
    log "[D3] Running zero-shot VLM evaluation ..."
    $PY "$SCRIPT_DIR/eval_vlm_zeroshot.py" \
        --qa "$VQA_V2_DIR" \
        --root "$DATASET_ROOT" \
        --out "$RESULTS_DIR/vlm_zeroshot.json"
    log "[D3] Done → $RESULTS_DIR/vlm_zeroshot.json"

    # D4: Multimodal alignment evaluation
    log "[D4] Running multimodal alignment evaluation ..."
    $PY "$SCRIPT_DIR/eval_multimodal.py" \
        --dataset "$DATASET_ROOT" \
        --out "$RESULTS_DIR/multimodal.json"
    log "[D4] Done → $RESULTS_DIR/multimodal.json"

    # D5: Spacing CNN generalization
    log "[D5] Evaluating spacing CNN generalization ..."
    $PY "$SCRIPT_DIR/eval_spacing_generalization.py" \
        --model "$SPACING_MODEL_DIR" \
        --cache "$SPACING_CACHE_DIR" \
        --out "$RESULTS_DIR/spacing_generalization.json"
    log "[D5] Done → $RESULTS_DIR/spacing_generalization.json"

    # D6: Spacing CNN cross-town evaluation
    log "[D6] Running cross-town spacing evaluation ..."
    $PY "$SCRIPT_DIR/eval_spacing_crosstown.py" \
        --model "$SPACING_MODEL_DIR" \
        --cache "$SPACING_CACHE_DIR" \
        --out "$RESULTS_DIR/spacing_crosstown.json"
    log "[D6] Done → $RESULTS_DIR/spacing_crosstown.json"

    # D7: Ablation over parameter sources
    log "[D7] Running parameter-source ablation ..."
    $PY "$SCRIPT_DIR/ablation_param_sources.py" \
        --dataset "$DATASET_ROOT" \
        --out "$RESULTS_DIR/ablation_param_sources.json"
    log "[D7] Done → $RESULTS_DIR/ablation_param_sources.json"

    # D8: Compare ablation arms
    log "[D8] Comparing ablation arms ..."
    $PY "$SCRIPT_DIR/compare_ablation_arms.py" \
        --results "$RESULTS_DIR/ablation_param_sources.json" \
        --out "$RESULTS_DIR/ablation_comparison.json"
    log "[D8] Done → $RESULTS_DIR/ablation_comparison.json"

    log "Phase D complete."
fi

# =============================================================================
# PHASE E — ADAPTATION STUDY
# LoRA fine-tunes Qwen2.5-VL at checkpoints 0.25, 0.5, and 1.0 epochs and
# evaluates each. Produces results consumed by make_adapt_table.py.
# Requires GPU with >= 24 GB VRAM (bf16 fine-tuning).
# =============================================================================

if [ "$SKIP_ADAPTATION" = "1" ]; then
    skip "Phase E — adaptation study (SKIP_ADAPTATION=1)"
else
    log ""
    log "=== PHASE E: Adaptation Study ==="

    require_dir "$VLM_FT_DATA_DIR"
    require_dir "$VQA_V2_DIR"

    # Export variables expected by run_adaptation_v3b.sh
    export PY
    export DATA="$VLM_FT_DATA_DIR"
    export QA="$VQA_V2_DIR"
    export ROOT="$DATASET_ROOT"
    export OUT="$ADAPT_OUT_DIR"
    export CODE="$SCRIPT_DIR"

    bash "$SCRIPT_DIR/run_adaptation_v3b.sh"
    log "Phase E complete → $ADAPT_OUT_DIR"
fi

# =============================================================================
# PHASE F — PAPER TABLES AND FIGURES
# Generates all LaTeX tables and PDF figures for the paper.
# =============================================================================

if [ "$SKIP_FIGURES" = "1" ]; then
    skip "Phase F — paper tables/figures (SKIP_FIGURES=1)"
else
    log ""
    log "=== PHASE F: Paper Tables and Figures ==="

    FIGS_DIR="$RESULTS_DIR/paper_figs"
    mkdir -p "$FIGS_DIR"

    log "[F1] Adaptation study table ..."
    $PY "$SCRIPT_DIR/make_adapt_table.py" \
        --results "$ADAPT_OUT_DIR" \
        --out "$FIGS_DIR/tab_adapt_v2.tex"

    log "[F2] Adaptation study figure ..."
    $PY "$SCRIPT_DIR/make_adapt_figure.py" \
        --results "$ADAPT_OUT_DIR" \
        --out "$FIGS_DIR/fig_adapt_v2.pdf"

    log "[F3] CF benchmark table ..."
    $PY "$SCRIPT_DIR/make_cfbench_table.py" \
        --results "$RESULTS_DIR/cf_bench.json" \
        --out "$FIGS_DIR/tab_cfbench.tex"

    log "[F4] Generalization table ..."
    $PY "$SCRIPT_DIR/make_generalization_table.py" \
        --results "$RESULTS_DIR/spacing_generalization.json" \
        --out "$FIGS_DIR/tab_generalization.tex"

    log "[F5] Multimodal table ..."
    $PY "$SCRIPT_DIR/make_multimodal_table.py" \
        --results "$RESULTS_DIR/multimodal.json" \
        --out "$FIGS_DIR/tab_multimodal.tex"

    log "[F6] VQA v2 benchmark table ..."
    $PY "$SCRIPT_DIR/make_v2_bench_table.py" \
        --results "$RESULTS_DIR/vqa_zeroshot.json" \
        --out "$FIGS_DIR/tab_v2_bench.tex"

    log "[F7] Sensor positions figure ..."
    $PY "$SCRIPT_DIR/make_sensor_positions_fig.py" \
        --out "$FIGS_DIR/fig_sensor_positions.pdf"

    log "[F8] All paper figures ..."
    $PY "$SCRIPT_DIR/make_paper_figures.py" \
        --dataset "$DATASET_ROOT" \
        --results "$RESULTS_DIR" \
        --out "$FIGS_DIR"

    log "[F9] Updating paper fidelity numbers ..."
    $PY "$SCRIPT_DIR/update_paper_fidelity.py" \
        --report "$RESULTS_DIR/fidelity_report.json"

    log "Phase F complete → $FIGS_DIR"
fi

# =============================================================================
# DONE
# =============================================================================

log ""
log "============================================================"
log " LLM_CF_v2 pipeline complete."
log ""
log " Key outputs:"
log "   Dataset        : $DATASET_ROOT"
log "   CF arrays      : $CF_ARRAYS_DIR"
log "   VQA v2         : $VQA_V2_DIR"
log "   Spacing CNN    : $SPACING_MODEL_DIR"
log "   Eval results   : $RESULTS_DIR"
log "   Paper figs     : $RESULTS_DIR/paper_figs"
log "============================================================"
