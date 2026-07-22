#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A46 run directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SCENE=${SCENE:-ramen}
SEED=${SEED:-20260719}

MEMORY_DIR=${MEMORY_DIR:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/equal_four_token_memory}
CACHE_DIR=${CACHE_DIR:-$ROOT/runs/a6_responsibility_multiscene_20260715/$SCENE/cache_l2_raw}
FEATURE_DIR=${FEATURE_DIR:-$ROOT/drsplat_data/lerf_ovs/$SCENE/language_features_multiscale}
RELATION_GRAPH_DIR=${RELATION_GRAPH_DIR:-$ROOT/runs/a39_ramen_relation_graph_20260719_155527/$SCENE/local_relation_graph}
OUTPUT_DIR=$RUN_ROOT/$SCENE/multiscale_set_relation_audit

mkdir -p "$OUTPUT_DIR"
export PYTHONHASHSEED=$SEED

"$PYTHON_BIN" "$SOURCE_DIR/build_multiscale_set_relation_diagnostic.py" \
  --memory_dir "$MEMORY_DIR" \
  --cache_dir "$CACHE_DIR" \
  --feature_dir "$FEATURE_DIR" \
  --relation_graph_dir "$RELATION_GRAPH_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --minimum_dominant_fraction 0.55 \
  --minimum_split_views 3 \
  --beta_prior 1.0 \
  --positive_threshold 0.65 \
  --negative_threshold 0.35 \
  --minimum_nll_improvement 0.10 \
  --minimum_split_stability 0.80 \
  --expected_memory_seed "$SEED"

"$PYTHON_BIN" - "$RUN_ROOT" "$OUTPUT_DIR/gate.json" <<'PY'
import json
import os
import sys

run_root, gate_path = sys.argv[1:]
with open(gate_path) as source:
    gate = json.load(source)
marker = "A46_1_READY" if gate["pass"] else "STOP_BEFORE_CODEBOOK_TRAINING"
with open(os.path.join(run_root, marker), "w") as output:
    output.write(json.dumps(gate, indent=2))
with open(os.path.join(run_root, "PROBE_COMPLETE"), "w") as output:
    output.write(marker + "\n")
print(json.dumps(gate, indent=2))
PY
