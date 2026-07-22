#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A50 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A50 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A47_RUN=${A47_RUN:-$ROOT/runs/a47_raw_entity_tomography_20260721_181309}
A47_AUDIT_DIR=$A47_RUN/ramen/entity_identifiability_audit
GEOMETRY_CHECKPOINT=${GEOMETRY_CHECKPOINT:-$ROOT/runs/3dgs/ramen/chkpnt30000.pth}
OUTPUT_DIR=$RUN_ROOT/ramen/group_addressed_spatial_audit
CACHE_ROOT=$RUN_ROOT/.cache

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch HF_HOME=$CACHE_ROOT/huggingface
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$OUTPUT_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

for required in \
  "$SOURCE_DIR/build_group_addressed_spatial_memory_audit.py" \
  "$SOURCE_DIR/build_geometry_conditioned_tracklet_partition.py" \
  "$SOURCE_DIR/build_persistent_entity_tomography.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$A47_RUN/PROBE_COMPLETE" "$A47_AUDIT_DIR/manifest.json" \
  "$A47_AUDIT_DIR/gaussian_atom_ids.npy" "$GEOMETRY_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "Missing A50 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$A47_AUDIT_DIR/manifest.json" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import json
import sys

manifest_path, seed = sys.argv[1:]
manifest = json.load(open(manifest_path))
assert manifest["experiment"] == "A47.0_raw_proposal_identifiability_audit"
assert manifest["source_contract"]["raw_overlapping_proposals"] is True
assert manifest["source_contract"]["evaluation_queries_or_labels_used"] is False
assert manifest["source_contract"]["codebooks_trained"] is False
assert int(manifest["source_contract"]["fixed_seed"]) == int(seed)
print("A50_GROUP_ADDRESS_INPUT_CONTRACT_OK")
PY

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u \
  "$SOURCE_DIR/build_group_addressed_spatial_memory_audit.py" \
  --a47_audit_dir "$A47_AUDIT_DIR" \
  --geometry_checkpoint "$GEOMETRY_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" --seed "$SEED" \
  > "$LOG_DIR/group_addressed_spatial_audit.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$OUTPUT_DIR/gate.json" <<'PY'
import json
import os
import sys

run_root, gate_path = sys.argv[1:]
gate = json.load(open(gate_path))
marker = "A50_1_READY" if gate["pass"] else "STOP_BEFORE_CONTINUOUS_RETRIEVAL_AND_CODEBOOKS"
with open(os.path.join(run_root, marker), "w") as output:
    output.write(json.dumps(gate, indent=2))
with open(os.path.join(run_root, "PROBE_COMPLETE"), "w") as output:
    output.write(marker + "\n")
print(json.dumps(gate, indent=2))
PY

echo "A50.0 Group-addressed spatial memory audit complete: $RUN_ROOT"
