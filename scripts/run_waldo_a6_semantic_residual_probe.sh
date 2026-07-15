#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_semantic_residual_waldo_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_semantic_residual_waldo_20260715}
GPU_ONE=${GPU_ONE:-0}
GPU_TWO=${GPU_TWO:-1}
ITERATIONS=${ITERATIONS:-3000}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

mkdir -p "$RUN_ROOT" "$LOG_DIR"

cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l2_raw
base=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt
split=$ROOT/runs/multiscale_split_consistency/waldo_l2_split2/consensus.pt
dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth
baseline=$ROOT/runs/paper_selection_20260714/waldo_kitchen/a6/metrics.json

for required in "$cache/manifest.json" "$base" "$split" "$dataset" "$labels" "$geometry" "$baseline"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

run_variant() {
  local variant=$1
  local output=$RUN_ROOT/$variant
  mkdir -p "$output"

  if [[ ! -f "$output/consensus.pt" ]]; then
    local train_args=(
      --cache_dir "$cache"
      --base_consensus "$base"
      --output_dir "$output"
      --iterations "$ITERATIONS"
      --batch_pixels 2048
      --topk 8
      --rank 8
      --code_lr 0.02
      --basis_lr 0.002
      --direct_weight 1.0
      --anchor_weight 0.2
      --code_regularization 0.0001
      --seed 20260715
    )
    if [[ "$variant" == "m2b_split_ccl_opacity" ]]; then
      train_args+=(
        --split_consensus "$split"
        --lovo_weight 0.5
        --contrastive_weight 0.02
        --contrastive_temperature 0.07
        --agreement_floor 0.65
        --direct_confidence_floor 0.25
        --train_semantic_opacity
        --opacity_lr 0.01
        --opacity_regularization 0.01
      )
    fi
    "$PYTHON_BIN" -u train_a6_semantic_residual.py "${train_args[@]}" \
      > "$LOG_DIR/train_${variant}.log" 2>&1
  fi

  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" \
      -m "$ROOT/runs/3dgs/waldo_kitchen" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$output/consensus.pt" \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds 0.55 \
      --occupancy_threshold 0.7 \
      --output "$output/eval" \
      > "$LOG_DIR/eval_${variant}.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_variant "$2"
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_ONE" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_a6_semantic_residual_probe.sh" \
    --worker m2a_residual_only \
  > "$LOG_DIR/worker_m2a.log" 2>&1 &
worker_one=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_TWO" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_a6_semantic_residual_probe.sh" \
    --worker m2b_split_ccl_opacity \
  > "$LOG_DIR/worker_m2b.log" 2>&1 &
worker_two=$!

status=0
wait "$worker_one" || status=$?
wait "$worker_two" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "One or more A6 semantic-residual workers failed with status=$status" >&2
  exit "$status"
fi

"$PYTHON_BIN" - "$baseline" "$RUN_ROOT" <<'PY'
import json
import os
import sys

baseline_path, run_root = sys.argv[1:]
sources = {
    "a6": baseline_path,
    "m2a_residual_only": os.path.join(run_root, "m2a_residual_only", "eval", "metrics.json"),
    "m2b_split_ccl_opacity": os.path.join(run_root, "m2b_split_ccl_opacity", "eval", "metrics.json"),
}
rows = {}
for name, path in sources.items():
    payload = json.load(open(path))
    grid = {float(row["selection_threshold"]): row for row in payload["threshold_summary"]}
    row = grid[0.55]
    rows[name] = {
        "mIoU": row["mIoU"],
        "mAcc@0.25": row["mAcc@0.25"],
        "mAcc@0.5": row["mAcc@0.5"],
    }
base_miou = rows["a6"]["mIoU"]
for row in rows.values():
    row["delta_mIoU_vs_a6"] = row["mIoU"] - base_miou
summary = {
    "selection_threshold": 0.55,
    "occupancy_threshold": 0.7,
    "results": rows,
}
with open(os.path.join(run_root, "comparison.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "Waldo A6 semantic-residual probe complete: $RUN_ROOT"
