#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hclt
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python "$ROOT/scripts/run_base_vs_dpo501_final51.py" >> "$ROOT/logs/generation.log" 2>&1
