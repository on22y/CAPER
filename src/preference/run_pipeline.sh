#!/usr/bin/env bash
set -euo pipefail
source /home/user20250805/anaconda3/etc/profile.d/conda.sh
conda activate hclt
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

ROOT=/home/user20250805/gayeon/HCLT2026/work/hcx_dpo900_mode_jury_v1
RUNNER="$ROOT/scripts/run_dpo900_mode_jury_pipeline.py"

python "$RUNNER" pipeline
