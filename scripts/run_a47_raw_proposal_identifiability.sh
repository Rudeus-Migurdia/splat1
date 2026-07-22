#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A47 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A47 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
SEED=${SEED:-20260719}
SCENE=${SCENE:-ramen}
DATASET=${DATASET:-$ROOT/drsplat_data/lerf_ovs/$SCENE}
CACHE_DIR=${CACHE_DIR:-$ROOT/runs/a6_responsibility_multiscene_20260715/$SCENE/cache_l2_raw}
GEOMETRY_CHECKPOINT=${GEOMETRY_CHECKPOINT:-$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth}
SAM_CHECKPOINT=${SAM_CHECKPOINT:-$ROOT/ckpts/sam_vit_h_4b8939.pth}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
PROPOSAL_DIR=$RUN_ROOT/$SCENE/raw_overlapping_proposals
AUDIT_DIR=$RUN_ROOT/$SCENE/entity_identifiability_audit
CACHE_ROOT=$RUN_ROOT/.cache

export ROOT RUN_ROOT LOG_DIR SOURCE_DIR PYTHON_BIN GPU_GUARD GPU_LIST SEED SCENE
export DATASET CACHE_DIR GEOMETRY_CHECKPOINT SAM_CHECKPOINT CLIP_CHECKPOINT

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=$CLIP_CHECKPOINT
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$PROPOSAL_DIR" "$AUDIT_DIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

for required in \
  "$SOURCE_DIR/export_raw_sam_proposals.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$GPU_GUARD" "$DATASET/images" "$CACHE_DIR/manifest.json" \
  "$GEOMETRY_CHECKPOINT" "$SAM_CHECKPOINT" "$CLIP_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "Missing A47 input: $required" >&2; exit 2; }
done

read -r -a GPUS <<< "$GPU_LIST"
NUM_SHARDS=${#GPUS[@]}
if (( NUM_SHARDS < 1 || NUM_SHARDS > 4 )); then
  echo "A47 requires between one and four guarded GPU workers" >&2
  exit 2
fi

if [[ "${1:-}" == "--proposal-worker" ]]; then
  SHARD_INDEX=${2:?missing shard index}
  "$PYTHON_BIN" -u "$SOURCE_DIR/export_raw_sam_proposals.py" \
    --dataset_path "$DATASET" \
    --output_dir "$PROPOSAL_DIR" \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --clip_checkpoint "$CLIP_CHECKPOINT" \
    --view_manifest "$CACHE_DIR/manifest.json" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --seed "$SEED" \
    --clip_batch_size 64
  exit 0
fi

"$PYTHON_BIN" - "$CACHE_DIR/manifest.json" "$DATASET/images" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import json
import os
import sys

manifest_path, image_dir, seed = sys.argv[1:]
manifest = json.load(open(manifest_path))
assert manifest["raw_contribution_weights"] is True
assert int(manifest["topk"]) >= 45
view_names = [str(item["image_name"]) for item in manifest["views"]]
image_names = {
    os.path.splitext(name)[0]
    for name in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, name))
}
assert view_names and len(view_names) == len(set(view_names))
assert set(view_names).issubset(image_names)
assert int(seed) == 20260719
print("A47_RAW_PROPOSAL_INPUT_CONTRACT_OK", len(view_names))
PY

worker_pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  "$PYTHON_BIN" "$GPU_GUARD" \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 21600 --poll-interval 5 -- \
    bash "$SOURCE_DIR/scripts/run_a47_raw_proposal_identifiability.sh" \
      --proposal-worker "$shard_index" \
      > "$LOG_DIR/proposal_gpu_${gpu}.log" 2>&1 &
  worker_pids+=("$!")
  echo "$!" > "$RUN_ROOT/proposal_gpu_${gpu}.pid"
done

worker_failure=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    worker_failure=1
  fi
done
if (( worker_failure != 0 )); then
  echo "At least one guarded raw-proposal worker failed" >&2
  exit 1
fi

"$PYTHON_BIN" "$SOURCE_DIR/export_raw_sam_proposals.py" \
  --dataset_path "$DATASET" \
  --output_dir "$PROPOSAL_DIR" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --clip_checkpoint "$CLIP_CHECKPOINT" \
  --view_manifest "$CACHE_DIR/manifest.json" \
  --seed "$SEED" \
  --finalize > "$LOG_DIR/proposal_finalize.log" 2>&1

"$PYTHON_BIN" -u "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  --geometry_checkpoint "$GEOMETRY_CHECKPOINT" \
  --cache_dir "$CACHE_DIR" \
  --proposal_dir "$PROPOSAL_DIR" \
  --output_dir "$AUDIT_DIR" \
  --seed "$SEED" \
  --target_atoms 8192 \
  --maximum_proposals_per_level 24 \
  --minimum_area_fraction 0.001 \
  --maximum_area_fraction 0.90 \
  --maximum_slots 192 \
  --minimum_nll_improvement 0.10 \
  --minimum_split_stability 0.80 \
  --minimum_nontrivial_mass_fraction 0.01 \
  > "$LOG_DIR/entity_audit.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$AUDIT_DIR/gate.json" <<'PY'
import json
import os
import sys

run_root, gate_path = sys.argv[1:]
gate = json.load(open(gate_path))
marker = "A47_1_READY" if gate["pass"] else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS"
with open(os.path.join(run_root, marker), "w") as output:
    output.write(json.dumps(gate, indent=2))
with open(os.path.join(run_root, "PROBE_COMPLETE"), "w") as output:
    output.write(marker + "\n")
print(json.dumps(gate, indent=2))
PY

echo "A47.0 raw proposal identifiability audit complete: $RUN_ROOT"
