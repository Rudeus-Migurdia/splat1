#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_continuous_paper_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_continuous_paper_20260715}
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

mkdir -p "$RUN_ROOT" "$LOG_DIR"

continuous_artifact() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt"
  else
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/$scene/fused_w1p5_t005.pt"
  fi
}

run_scene() {
  local scene=$1
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local consensus
  consensus=$(continuous_artifact "$scene")
  local output=$RUN_ROOT/$scene

  for required in "$dataset" "$labels" "$geometry" "$consensus"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; return 2; }
  done
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene reuse=$output/metrics.json"
    return
  fi

  mkdir -p "$output"
  echo "[$(date +%FT%T)] scene=$scene continuous start"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$geometry" \
    --consensus_path "$consensus" \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --output "$output" \
    > "$LOG_DIR/${scene}.log" 2>&1
  echo "[$(date +%FT%T)] scene=$scene continuous done"
}

run_worker() {
  local worker=$1
  shift
  echo "[$(date +%FT%T)] worker=$worker scenes=$*"
  for scene in "$@"; do
    run_scene "$scene"
  done
  echo "[$(date +%FT%T)] worker=$worker complete"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_ONE" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_a6_continuous_paper_audit.sh" \
    --worker gpu-one figurines teatime \
  > "$LOG_DIR/worker_gpu_one.log" 2>&1 &
worker_one=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_TWO" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_a6_continuous_paper_audit.sh" \
    --worker gpu-two ramen waldo_kitchen \
  > "$LOG_DIR/worker_gpu_two.log" 2>&1 &
worker_two=$!

status=0
wait "$worker_one" || status=$?
wait "$worker_two" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "One or more continuous-audit workers failed with status=$status" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
  "$RUN_ROOT/figurines/metrics.json" \
  "$RUN_ROOT/ramen/metrics.json" \
  "$RUN_ROOT/teatime/metrics.json" \
  "$RUN_ROOT/waldo_kitchen/metrics.json" \
  --output "$RUN_ROOT/four_scene_metrics.json" \
  > "$RUN_ROOT/four_scene_table.md"

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 continuous paper audit complete: $RUN_ROOT"
