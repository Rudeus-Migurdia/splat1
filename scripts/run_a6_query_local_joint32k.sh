#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/runs/a6_query_margin_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_query_local_joint32k_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_query_local_joint32k_20260715}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene/eval
  mkdir -p "$output"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$SOURCE_ROOT/$scene/candidate_ids" \
    --query_route_base_codebook_dir "$SOURCE_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --occupancy_threshold 0.7 \
    --output "$output" \
    > "$LOG_DIR/${scene}_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

scenes=(figurines ramen teatime waldo_kitchen)
pids=()
for gpu in 0 1 2 3; do
  scene=${scenes[$gpu]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_a6_query_local_joint32k.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
  "$RUN_ROOT/figurines/eval/metrics.json" \
  "$RUN_ROOT/ramen/eval/metrics.json" \
  "$RUN_ROOT/teatime/eval/metrics.json" \
  "$RUN_ROOT/waldo_kitchen/eval/metrics.json" \
  --output "$RUN_ROOT/four_scene_metrics.json" \
  > "$RUN_ROOT/four_scene_table.md"

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 query-local joint-32k evaluation complete: $RUN_ROOT"
