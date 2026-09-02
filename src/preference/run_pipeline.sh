#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hclt
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNNER="$ROOT/scripts/run_dpo900_mode_jury_pipeline.py"

python "$RUNNER" pipeline
