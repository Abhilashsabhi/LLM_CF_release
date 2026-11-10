#!/usr/bin/env bash
# V2 adaptation study for tab_adapt_v2.tex + fig_adapt_v2.pdf
# Uses released QA pairs format (all 8 types) for fine-tuning and evaluation.
# Output format compatible with make_adapt_table.py and make_adapt_figure.py.
set -euo pipefail

PY=/var/tmp/vlmft310/bin/python
DATA=/var/tmp/vlm_ft_data_v2          # released QA pairs training data
QA=/var/tmp/LLM_CF_v2_local/vqa_v2    # released QA pairs for evaluation
ROOT=/var/tmp/LLM_CF_v2_local         # dataset root for eval
OUT=/var/tmp/adapt_v3                  # where make_adapt_table.py expects results
CODE=/data/ABHILASH/Downloads/CARLA_0.9.15/CFDATA
export HF_HOME=/var/tmp/hf

mkdir -p "$OUT"
cd "$CODE"

echo "[$(date '+%F %T')] === unadapted bf16 baseline (all 8 types) ==="
$PY eval_qwenvl_v2.py --qa "$QA" --root "$ROOT" --out "$OUT/before.json" \
    --n 100 --split test

echo "[$(date '+%F %T')] === LoRA fine-tune on released QA pairs (1 epoch with checkpoints) ==="
$PY finetune_qwenvl_v2.py --data "$DATA" --out "$OUT" --epochs 1.0 \
    --ckpt-epochs 0.25 0.5 1.0

for E in 0p25 0p5 1; do
    echo "[$(date '+%F %T')] === evaluating checkpoint epoch$E ==="
    $PY eval_qwenvl_v2.py --qa "$QA" --root "$ROOT" \
        --out "$OUT/after_epoch${E}.json" \
        --adapter "$OUT/adapter_epoch${E}" \
        --n 100 --split test
done

echo "[$(date '+%F %T')] ADAPTATION V3B COMPLETE"
