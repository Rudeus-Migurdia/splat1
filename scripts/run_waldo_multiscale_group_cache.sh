#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
LEVEL=${LEVEL:?Set LEVEL to a multiscale SAM level}
OUTPUT=${OUTPUT:?Set OUTPUT to an isolated cache directory}
LOG_FILE=${LOG_FILE:?Set LOG_FILE}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG_FILE")"

if [[ -f "$OUTPUT/manifest.json" ]]; then
  echo "cache already complete: $OUTPUT"
  exit 0
fi

.venv/bin/python -u prepare_semantic_field.py \
  -s drsplat_data/lerf_ovs/waldo_kitchen \
  -m "$OUTPUT" \
  --geometry_checkpoint runs/3dgs/waldo_kitchen/chkpnt30000.pth \
  --feature_dir drsplat_data/lerf_ovs/waldo_kitchen/language_features_multiscale \
  --feature_level "$LEVEL" \
  --semantic_dim 512 \
  --identity_codec \
  --topk 45 \
  --max_pixels_per_view 32768 \
  --raw_contribution_weights \
  > "$LOG_FILE" 2>&1
