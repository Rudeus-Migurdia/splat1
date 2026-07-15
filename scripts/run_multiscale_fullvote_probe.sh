#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
FEATURE_LEVELS=${FEATURE_LEVELS:-"0 1 2 3"}

cd "$ROOT"
for level in $FEATURE_LEVELS; do
  FEATURE_DIR_NAME=language_features_multiscale \
  FEATURE_LEVEL="$level" \
  RUN_SUFFIX="_multiscale_l${level}" \
  ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_baseline_voting_consensus_probe.sh
done

echo "multiscale full-view probe complete: levels=$FEATURE_LEVELS"
