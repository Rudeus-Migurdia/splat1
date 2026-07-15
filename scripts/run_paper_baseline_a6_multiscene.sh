#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/paper_selection_20260714}
LOG_DIR=${LOG_DIR:-$ROOT/logs/paper_selection_20260714}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

mkdir -p "$RUN_ROOT" "$LOG_DIR"

a6_artifact() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2"
  else
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/$scene/fused_w1p5_t005_codebook_k4096x2"
  fi
}

run_scene() {
  local scene=$1
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local baseline_model=$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128
  local codebook
  codebook=$(a6_artifact "$scene")
  local scene_out=$RUN_ROOT/$scene
  mkdir -p "$scene_out"

  for required in "$dataset" "$labels" "$geometry" "$baseline_model/chkpnt0.pth" "$codebook/manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; return 2; }
  done

  echo "[$(date +%FT%T)] scene=$scene method=baseline start"
  "$PYTHON_BIN" -u eval_lerf_ovs_miou.py \
    -s "$dataset" \
    -m "$baseline_model" \
    --checkpoint "$baseline_model/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --output "$scene_out/baseline" \
    > "$LOG_DIR/${scene}_baseline.log" 2>&1
  echo "[$(date +%FT%T)] scene=$scene method=baseline done"

  echo "[$(date +%FT%T)] scene=$scene method=a6 start"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$codebook" \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --output "$scene_out/a6" \
    > "$LOG_DIR/${scene}_a6.log" 2>&1
  echo "[$(date +%FT%T)] scene=$scene method=a6 done"
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

summarize_all() {
  local method=$1
  "$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
    "$RUN_ROOT/figurines/$method/metrics.json" \
    "$RUN_ROOT/ramen/$method/metrics.json" \
    "$RUN_ROOT/teatime/$method/metrics.json" \
    "$RUN_ROOT/waldo_kitchen/$method/metrics.json" \
    --output "$RUN_ROOT/${method}_four_scene_metrics.json" \
    > "$RUN_ROOT/${method}_four_scene_table.md"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu 1 --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_paper_baseline_a6_multiscene.sh" \
    --worker gpu1 figurines teatime \
  > "$LOG_DIR/worker_gpu1.log" 2>&1 &
worker_one=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu 2 --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_paper_baseline_a6_multiscene.sh" \
    --worker gpu2 ramen waldo_kitchen \
  > "$LOG_DIR/worker_gpu2.log" 2>&1 &
worker_two=$!

status=0
wait "$worker_one" || status=$?
wait "$worker_two" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "One or more workers failed with status=$status" >&2
  exit "$status"
fi

summarize_all baseline
summarize_all a6
date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "Paper baseline/A6 evaluation complete: $RUN_ROOT"
