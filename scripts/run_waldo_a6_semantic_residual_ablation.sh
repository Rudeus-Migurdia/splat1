#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_semantic_residual_waldo_20260715/ablations}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_semantic_residual_waldo_20260715/ablations}
GPU_ONE=${GPU_ONE:-0}
GPU_TWO=${GPU_TWO:-1}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l2_raw
base=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt
split=$ROOT/runs/multiscale_split_consistency/waldo_l2_split2/consensus.pt
dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth

run_variant() {
  local variant=$1
  local contrastive_weight=$2
  local output=$RUN_ROOT/$variant
  mkdir -p "$output"
  if [[ ! -f "$output/consensus.pt" ]]; then
    "$PYTHON_BIN" -u train_a6_semantic_residual.py \
      --cache_dir "$cache" \
      --base_consensus "$base" \
      --split_consensus "$split" \
      --output_dir "$output" \
      --iterations 3000 \
      --batch_pixels 2048 \
      --topk 8 \
      --rank 8 \
      --code_lr 0.02 \
      --basis_lr 0.002 \
      --direct_weight 1.0 \
      --lovo_weight 0.5 \
      --contrastive_weight "$contrastive_weight" \
      --contrastive_temperature 0.07 \
      --agreement_floor 0.65 \
      --direct_confidence_floor 0.25 \
      --anchor_weight 0.2 \
      --code_regularization 0.0001 \
      --seed 20260715 \
      > "$LOG_DIR/train_${variant}.log" 2>&1
  fi
  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
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
  run_variant "$2" "$3"
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_ONE" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_a6_semantic_residual_ablation.sh" \
    --worker m2c_split_lovo 0.0 \
  > "$LOG_DIR/worker_m2c.log" 2>&1 &
p0=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_TWO" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_a6_semantic_residual_ablation.sh" \
    --worker m2d_split_ccl_no_opacity 0.02 \
  > "$LOG_DIR/worker_m2d.log" 2>&1 &
p1=$!

status=0
wait "$p0" || status=$?
wait "$p1" || status=$?
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$ROOT" "$RUN_ROOT" <<'PY'
import json
import os
import sys

root, run_root = sys.argv[1:]
paths = {
    "a6": os.path.join(root, "runs/paper_selection_20260714/waldo_kitchen/a6/metrics.json"),
    "m2b_split_ccl_trained_with_opacity": os.path.join(
        root, "runs/a6_semantic_residual_waldo_20260715/m2b_eval_no_opacity/metrics.json"
    ),
    "m2c_split_lovo": os.path.join(run_root, "m2c_split_lovo/eval/metrics.json"),
    "m2d_split_ccl_no_opacity": os.path.join(
        run_root, "m2d_split_ccl_no_opacity/eval/metrics.json"
    ),
}
rows = {}
for name, path in paths.items():
    payload = json.load(open(path))
    row = next(
        item
        for item in payload["threshold_summary"]
        if abs(item["selection_threshold"] - 0.55) < 1e-9
    )
    rows[name] = {key: row[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}
base = rows["a6"]["mIoU"]
for row in rows.values():
    row["delta_mIoU_vs_a6"] = row["mIoU"] - base
with open(os.path.join(run_root, "comparison.json"), "w") as output:
    json.dump(rows, output, indent=2)
print(json.dumps(rows, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
