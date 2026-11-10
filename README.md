# LLM_CF_v2: A Large-Scale Synthetic Car-Following Dataset

LLM_CF_v2 is a synthetic car-following dataset of **1329 admitted scenes** (11.1 hours, 10 Hz) collected in CARLA 0.9.15, where behavioural parameters for an Intelligent Driver Model (IDM) are proposed by a large language model using self-consistency voting rather than hand-tuned priors. The dataset covers 9 scenario categories, 3 driving styles, and 2 traffic regimes, and ships 14 sensor streams per frame together with **338,895 VQA pairs** across 8 question types. This repository contains the full generation, training, and evaluation code accompanying the paper submission to IEEE Transactions on Intelligent Transportation Systems.

---

## Dataset Overview

| Property | Value |
|---|---|
| Admitted scenes | 1329 |
| Total duration | ~11.1 hours |
| Recording rate | 10 Hz (0.1 s / frame) |
| Scene duration | 30 s (300 frames) |
| Scenario categories | 9 |
| Driving styles | 3 (aggressive, normal, cautious) |
| Traffic regimes | 2 (urban, motorway) |
| Sensor streams | 14 |
| VQA pairs | 338,895 |

### Scenario Categories

`steady_follow`, `adverse_weather`, `aggressive_leader`, `stop_and_go`, `cut_in`, `cut_out`, `speed_limit_change`, `grade_change`, `pedestrian_crossing`

### Sensor Streams (14 total)

| Stream | Format | Resolution / Shape |
|---|---|---|
| RGB-CAM\_FRONT | .jpg | 1280 × 720 |
| RGB-CAM\_FRONT\_LEFT | .jpg | 1280 × 720 |
| RGB-CAM\_FRONT\_RIGHT | .jpg | 1280 × 720 |
| RGB-CAM\_BACK | .jpg | 1280 × 720 |
| RGB-CAM\_BACK\_LEFT | .jpg | 1280 × 720 |
| RGB-CAM\_BACK\_RIGHT | .jpg | 1280 × 720 |
| LIDAR | .npy | (N, 3) — FLU |
| SEG-LIDAR | .npy | (N, 6) — semantic |
| RAD\_FRONT | .npy | (N, 4) |
| RAD\_LEFT | .npy | (N, 4) |
| RAD\_RIGHT | .npy | (N, 4) |
| RAD\_BACK | .npy | (N, 4) |
| GNSS | .npy | (3,) |
| IMU | .npy | (7,) |

All point clouds use the FLU (Forward-Left-Up) coordinate frame. The folder layout mirrors SimBEV exactly.

### VQA Pairs

338,895 pairs covering 8 question types:

| Type | Level | Question |
|---|---|---|
| Spacing estimation | Frame | Bumper-to-bumper gap to lead vehicle |
| Lead presence | Frame | Is a lead vehicle within 60 m? |
| Visual conditions | Frame | Weather / lighting conditions |
| Gap trend | Clip | CLOSING / OPENING / STEADY over 2 s |
| Relative motion | Clip | Lead vehicle faster or slower than ego? |
| TTC safety | Clip | Is time-to-collision below 3 s? |
| Ego action | Clip | Ego driving behaviour over the clip |
| Scenario recognition | Clip | Which of the 9 scenario categories? |

---

## Pipeline Overview

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     LLM_CF_v2 Pipeline                          │
  └─────────────────────────────────────────────────────────────────┘

  Stage 1               Stage 3               Stage 2
  LLM Parameter    →    Scene List       →    CARLA Data
  Generation            Construction          Collection
  (stage1_llm_          (stage3_scenario_     (stage2_carla_
   generator.py)         manager.py)           executor.py)
        │                                           │
        │  Self-consistency                         │  14 sensor streams
        │  voting (N=5)                             │  per frame
        │                                           ▼
        └──────────────────────────────►  Stage 4: Validate
                                          (stage4_validator.py)
                                                    │
                                                    ▼
                                          Stage 5: VQA Generation
                                          (stage5b_vqa_v2.py)
                                          8 types, 338 895 pairs
```

---

## Requirements

### Hardware

- GPU with **>= 8 GB VRAM** (required for VLM inference and CNN training)
- GPU with **>= 24 GB VRAM** recommended for LoRA fine-tuning (bf16)
- **~2 TB** free disk space for the full 1350-scene dataset (sensor data)
- ~10 GB for kinematics-only dry run

### Software

- **CARLA 0.9.15** (build `c93d684fa3` used in the paper)
- **Python 3.8+**
- LLM backend: Groq API key **or** local Ollama with `qwen2.5:14b-instruct`

---

## Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/<org>/LLM_CF_v2.git
cd LLM_CF_v2
```

**Step 2 — Install Python dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Install the CARLA Python API**

```bash
# Adjust path to your CARLA installation
pip install /path/to/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl
```

**Step 4 — Set up LLM backend**

Option A — Groq cloud (recommended; free tier sufficient for N_INSTANCES=50):

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Option B — Local Ollama:

```bash
ollama serve
ollama pull qwen2.5:14b-instruct
```

> **Note:** Stage 1 refuses to run without a reachable LLM backend. A dataset generated from hardcoded priors is not an LLM-parameterised dataset, and the central claim would be unverifiable.

---

## Quick Start (12-Scene Smoke Test)

Run 12 scenes (4 per style, 1 per scenario subset) to verify the full pipeline works before committing terabytes of disk:

```bash
# Start CARLA in a separate terminal
/path/to/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -fps=20 -carla-port=2000

# In this repo directory
DATASET_ROOT=/tmp/cf_smoke N_INSTANCES=4 python main.py \
    --stage all \
    --output /tmp/cf_smoke \
    --n-instances 4 \
    --n-votes 3 \
    --carla-host localhost \
    --carla-port 2000

# Check validation report
python -c "
import json
with open('/tmp/cf_smoke/infos/validation_report.json') as f:
    r = json.load(f)
print('PASS' if r.get('overall_pass') else 'FAIL', r.get('total_frames'), 'frames')
"
```

**Kinematics-only dry run** (no sensor data, much faster; use to validate distributions before the full collection):

```bash
python main.py --stage 2 --output /tmp/cf_dry \
    --carla-host localhost --carla-port 2000 \
    --no-sensors --no-record
python main.py --stage 4 --output /tmp/cf_dry
```

---

## Full Dataset Generation

Configure the key variables and run `run_pipeline.sh`:

```bash
# Key variables (pass as environment or edit run_pipeline.sh)
export DATASET_ROOT=/mnt/data/LLM_CF_v2
export CARLA_HOST=localhost
export CARLA_PORT=2000
export N_INSTANCES=50        # 50 → 1350 scenes (paper scale)
export LLM_BACKEND=groq      # or ollama
export MODEL=qwen2.5:14b-instruct

bash run_pipeline.sh
```

Expected time: ~13 hours for N_INSTANCES=50 on a mid-range GPU workstation (35 s/scene including CARLA overhead).

The script is **resumable**: Stage 2 checkpoints after every scene and skips already-collected scenes on restart.

---

## Downstream Experiments

All downstream scripts are run in sequence by `run_all.sh`. They can also be called individually:

| Script | Purpose | Primary Input | Output |
|---|---|---|---|
| `validate_against_real.py` | Fidelity vs. Waymo / HighD empirical distributions | dataset root | `fidelity_report.json` |
| `export_cf_arrays.py` | Export car-following arrays (train / val / test) | dataset root | `cf_arrays/*.npy` |
| `build_vlm_ft_data_v2.py` | Build VLM fine-tuning corpus from VQA v2 pairs | `vqa_v2/` | `vlm_ft_data_v2/` |
| `build_spacing_cache.py` | Build spacing perception cache for CNN training | dataset root | `spacing_cache/` |
| `build_multimodal_cache.py` | Build multimodal alignment cache | dataset root | `multimodal_cache/` |
| `train_spacing_cnn.py` | Train supervised spacing perception CNN | `spacing_cache/` | model + `results.json` |
| `finetune_qwenvl_v2.py` | LoRA-adapt Qwen2.5-VL on VQA pairs | `vlm_ft_data_v2/` | `adapter/` |
| `eval_carfollowing_v2.py` | Car-following benchmark (IDM fit, TTC, jerk) | `cf_arrays/` | `cf_bench.json` |
| `eval_qwenvl_v2.py` | VQA benchmark (adapted and unadapted) | `vqa_v2/`, `adapter/` | `vqa_results.json` |
| `eval_vlm_zeroshot.py` | Zero-shot VLM evaluation across 8 QA types | `vqa_v2/` | `vlm_zeroshot.json` |
| `eval_multimodal.py` | Multimodal alignment evaluation | dataset root | `multimodal.json` |
| `eval_spacing_generalization.py` | Spacing CNN cross-scenario generalization | model, cache | `spacing_gen.json` |
| `eval_spacing_crosstown.py` | Spacing CNN cross-town transfer | model, cache | `spacing_xtown.json` |
| `ablation_param_sources.py` | Parameter source ablation (LLM vs. priors vs. random) | — | `ablation.json` |
| `compare_ablation_arms.py` | Statistical comparison of ablation arms | `ablation.json` | `ablation_cmp.json` |
| `check_settling.py` | Verify scene settling / warm-up period | dataset root | console report |
| `validate_styles.py` | Validate driving style distributions | dataset root | console report |
| `audit_qa_exact.py` | Audit QA pairs for exact-match leakage | `vqa_v2/` | console report |
| `audit_qa_leakage.py` | Deeper train/test leakage audit | `vqa_v2/` | console report |
| `make_adapt_table.py` | Generate LaTeX adaptation study table | `adapt_v3b/` | `tab_adapt_v2.tex` |
| `make_adapt_figure.py` | Generate adaptation learning curve figure | `adapt_v3b/` | `fig_adapt_v2.pdf` |
| `make_cfbench_table.py` | Generate LaTeX CF benchmark table | `cf_bench.json` | `tab_cfbench.tex` |
| `make_generalization_table.py` | Generate LaTeX generalization table | results | `tab_generalization.tex` |
| `make_multimodal_table.py` | Generate LaTeX multimodal table | results | `tab_multimodal.tex` |
| `make_v2_bench_table.py` | Generate LaTeX VQA v2 benchmark table | results | `tab_v2_bench.tex` |
| `make_sensor_positions_fig.py` | Sensor rig diagram | — | `fig_sensor_positions.pdf` |
| `make_paper_figures.py` | All remaining paper figures | dataset, results | `paper_figs/` |
| `update_paper_fidelity.py` | Patch fidelity numbers into paper LaTeX | `fidelity_report.json` | updated .tex |

---

## One-Command Reproduction

```bash
# Runs Phases A–F: collection, post-processing, training, eval, adaptation, figures
bash run_all.sh
```

Individual phases can be skipped:

```bash
# Skip collection (dataset already on disk) and training
SKIP_COLLECTION=1 SKIP_TRAINING=1 DATASET_ROOT=/mnt/data/LLM_CF_v2 bash run_all.sh
```

The adaptation study (Phase E) is also available standalone:

```bash
# Fine-tunes Qwen2.5-VL at epoch checkpoints 0.25 / 0.5 / 1.0
DATA=/path/to/vlm_ft_data_v2 QA=/path/to/vqa_v2 ROOT=/path/to/dataset \
    OUT=/path/to/adapt_out bash run_adaptation_v3b.sh
```

---

## Folder Structure (Generated Dataset)

```
cf_dataset/
├── configs/
│   ├── llm_base_params.yaml    LLM-proposed IDM parameters (verifiable provenance)
│   ├── llm_instances.json      Per-scene parameter samples
│   ├── llm_metadata.json       Backend, model, n_votes, seed, prompt
│   ├── scene_list.json         Full scene specification list
│   └── stage2_records.json     Per-scene collection records
├── ground-truth/
│   └── det/<scene_id>/         Per-frame kinematic GT (.npy, 20 fields)
├── infos/
│   ├── cf_infos_train.json
│   ├── cf_infos_val.json
│   ├── cf_infos_test.json
│   └── validation_report.json
├── logs/                       CARLA recorder logs (.log)
├── sweeps/
│   ├── RGB-CAM_FRONT/<scene_id>/  Per-frame .jpg
│   ├── ...                        (6 cameras)
│   ├── LIDAR/<scene_id>/          Per-frame .npy
│   ├── SEG-LIDAR/<scene_id>/
│   ├── RAD_FRONT/<scene_id>/      (4 radars)
│   ├── ...
│   ├── GNSS/<scene_id>/
│   └── IMU/<scene_id>/
└── vqa_v2/<scene_id>/          Per-frame VQA JSON files (8 types)
```

---

## Known Issues and Platform Notes

**AMD iGPU / NVIDIA discrete GPU conflict (Vulkan)**
On systems with both an AMD integrated GPU and an NVIDIA discrete GPU, CARLA may bind Vulkan to the iGPU and fail to render. Force the discrete GPU with:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./CarlaUE4.sh -RenderOffScreen ...
```

**Ollama VRAM contention with CARLA**
Running Ollama (Stage 1) and CARLA (Stage 2) simultaneously on the same GPU causes OOM on <= 16 GB VRAM cards. The pipeline naturally avoids this — Stage 1 (LLM calls) completes before Stage 2 (CARLA) begins — but avoid running them concurrently on separate terminals.

**Radar velocity channel (NaN)**
The radial-velocity channel in the four radar streams (`RAD_*`, column index 3) is NaN on CARLA build `c93d684fa3`. This is a simulator defect. The channel is present in the `.npy` arrays for format compatibility but should not be used for evaluation. All results in the paper use only the (x, y, z) position columns.

**Stage 2 per-map subprocesses**
Stage 2 spawns one child process per CARLA map to work around a segfault in `client.load_world()` when called repeatedly within a single long-lived process. Each child exits cleanly after collecting its map; a failing child is retried once and then skipped, so the run continues. Re-invoke Stage 2 to fill any gap.

---

## Citation

```bibtex
@article{llm_cf_v2_2025,
  title   = {{LLM\_CF\_v2}: A Large-Scale Synthetic Car-Following Dataset
             with {LLM}-Proposed {IDM} Parameters and Multimodal {VQA} Pairs},
  author  = {<authors>},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.
