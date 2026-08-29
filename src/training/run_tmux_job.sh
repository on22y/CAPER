#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user20250805/gayeon/HCLT2026/work/hcx_dpo501_train_v3
source /home/user20250805/anaconda3/etc/profile.d/conda.sh
conda activate hclt
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python "$ROOT/scripts/run_dpo501_train.py" --mode train >> "$ROOT/logs/train.log" 2>&1
