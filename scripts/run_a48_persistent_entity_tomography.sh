#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A48 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A48 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A47_RUN=${A47_RUN:-$ROOT/runs/a47_raw_entity_tomography_20260721_181309}
A47_AUDIT_DIR=$A47_RUN/ramen/entity_identifiability_audit
OUTPUT_DIR=$RUN_ROOT/ramen/persistent_entity_tomography
CACHE_ROOT=$RUN_ROOT/.cache

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$OUTPUT_DIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

for required in \
  "$SOURCE_DIR/build_persistent_entity_tomography.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$A47_RUN/PROBE_COMPLETE" "$A47_AUDIT_DIR/manifest.json" \
  "$A47_AUDIT_DIR/incidence_views"; do
  [[ -e "$required" ]] || { echo "Missing A48 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$A47_RUN/PROBE_COMPLETE" "$A47_AUDIT_DIR/manifest.json" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import json
import sys

complete_path, manifest_path, seed = sys.argv[1:]
assert open(complete_path).read().strip() == "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS"
manifest = json.load(open(manifest_path))
assert manifest["experiment"] == "A47.0_raw_proposal_identifiability_audit"
assert manifest["gate"]["pass"] is False
assert manifest["source_contract"]["raw_overlapping_proposals"] is True
assert manifest["source_contract"]["raw_top45_talpha"] is True
assert manifest["source_contract"]["evaluation_queries_or_labels_used"] is False
assert manifest["source_contract"]["codebooks_trained"] is False
assert int(manifest["source_contract"]["fixed_seed"]) == int(seed)
print("A48_PERSISTENT_ENTITY_INPUT_CONTRACT_OK")
PY

"$PYTHON_BIN" -u "$SOURCE_DIR/build_persistent_entity_tomography.py" \
  --a47_audit_dir "$A47_AUDIT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --coverage_threshold 0.30 \
  --minimum_spatial_jaccard 0.35 \
  --minimum_semantic_cosine 0.75 \
  --minimum_association 0.40 \
  --spatial_weight 0.85 \
  --temporal_neighbors 2 \
  --minimum_persistence_views 3 \
  --merge_jaccard 0.85 \
  --merge_semantic_cosine 0.90 \
  --maximum_slots 256 \
  --evaluation_candidates 6 \
  --union_relative_nll_penalty 0.05 \
  --minimum_nll_improvement 0.10 \
  --minimum_split_stability 0.80 \
  --minimum_stable_slots 8 \
  --minimum_slot_count_agreement 0.80 \
  --minimum_union_mass_fraction 0.01 \
  --maximum_union_mass_fraction 0.50 \
  > "$LOG_DIR/persistent_entity_audit.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$OUTPUT_DIR/gate.json" <<'PY'
import json
import os
import sys

run_root, gate_path = sys.argv[1:]
gate = json.load(open(gate_path))
marker = "A48_1_READY" if gate["pass"] else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS"
with open(os.path.join(run_root, marker), "w") as output:
    output.write(json.dumps(gate, indent=2))
with open(os.path.join(run_root, "PROBE_COMPLETE"), "w") as output:
    output.write(marker + "\n")
print(json.dumps(gate, indent=2))
PY

echo "A48.0 persistent entity tomography complete: $RUN_ROOT"
