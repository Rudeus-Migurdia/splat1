#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A49 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A49 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A47_RUN=${A47_RUN:-$ROOT/runs/a47_raw_entity_tomography_20260721_181309}
A48_RUN=${A48_RUN:-$ROOT/runs/a48_persistent_birth_mdl_20260721_191530}
A47_AUDIT_DIR=$A47_RUN/ramen/entity_identifiability_audit
A48_AUDIT_DIR=$A48_RUN/ramen/persistent_entity_tomography
GEOMETRY_CHECKPOINT=${GEOMETRY_CHECKPOINT:-$ROOT/runs/3dgs/ramen/chkpnt30000.pth}
OUTPUT_DIR=$RUN_ROOT/ramen/geometry_tracklet_partition
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
  "$SOURCE_DIR/build_geometry_conditioned_tracklet_partition.py" \
  "$SOURCE_DIR/build_persistent_entity_tomography.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$A47_RUN/PROBE_COMPLETE" "$A48_RUN/PROBE_COMPLETE" \
  "$A47_AUDIT_DIR/manifest.json" "$A48_AUDIT_DIR/manifest.json" \
  "$A47_AUDIT_DIR/gaussian_atom_ids.npy" "$GEOMETRY_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "Missing A49 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$A47_AUDIT_DIR/manifest.json" "$A48_AUDIT_DIR/manifest.json" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import json
import sys

a47_path, a48_path, seed = sys.argv[1:]
a47 = json.load(open(a47_path))
a48 = json.load(open(a48_path))
assert a47["experiment"] == "A47.0_raw_proposal_identifiability_audit"
assert a48["experiment"] == "A48.0_persistent_birth_mdl_entity_tomography"
assert a47["source_contract"]["evaluation_queries_or_labels_used"] is False
assert a47["source_contract"]["codebooks_trained"] is False
assert int(a47["source_contract"]["fixed_seed"]) == int(seed)
assert a48["gate"]["pass"] is False
print("A49_GEOMETRY_PARTITION_INPUT_CONTRACT_OK")
PY

"$PYTHON_BIN" -u "$SOURCE_DIR/build_geometry_conditioned_tracklet_partition.py" \
  --a47_audit_dir "$A47_AUDIT_DIR" \
  --a48_audit_dir "$A48_AUDIT_DIR" \
  --geometry_checkpoint "$GEOMETRY_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  > "$LOG_DIR/geometry_partition_audit.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$OUTPUT_DIR/gate.json" <<'PY'
import json
import os
import sys

run_root, gate_path = sys.argv[1:]
gate = json.load(open(gate_path))
marker = "A49_1_READY" if gate["pass"] else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS"
with open(os.path.join(run_root, marker), "w") as output:
    output.write(json.dumps(gate, indent=2))
with open(os.path.join(run_root, "PROBE_COMPLETE"), "w") as output:
    output.write(marker + "\n")
print(json.dumps(gate, indent=2))
PY

echo "A49.0 geometry-conditioned tracklet partition complete: $RUN_ROOT"
