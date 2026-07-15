#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/visibility_assignment/waldo_mass095_rel001_min2}
LOG_DIR=${LOG_DIR:-$ROOT/logs/visibility_assignment_waldo_mass095_rel001_min2}
GPU_OLD=${GPU_OLD:-2}
GPU_L2=${GPU_L2:-3}

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

dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth
old_cache=$RUN_ROOT/old_split2
l2_cache=$RUN_ROOT/l2_split2
fused=$RUN_ROOT/fused_w1p5_t005.pt
evaluation=$RUN_ROOT/eval_continuous_paper
mkdir -p "$RUN_ROOT" "$LOG_DIR"

prepare_source() {
  local tag=$1
  local feature_dir=$2
  local feature_level=$3
  local output=$4
  if [[ -f "$output/manifest.json" ]]; then
    echo "[$(date +%FT%T)] source=$tag reuse=$output/manifest.json"
    return
  fi
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$output" \
    --geometry_checkpoint "$geometry" \
    --feature_dir "$feature_dir" --feature_level "$feature_level" \
    --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --consensus_only --consensus_chunk_pixels 1024 --consensus_splits 2 \
    --visibility_mass_fraction 0.95 \
    --visibility_relative_floor 0.01 \
    --visibility_min_contributors 2 \
    > "$LOG_DIR/${tag}_prepare.log" 2>&1
}

if [[ "${1:-}" == "--prepare" ]]; then
  prepare_source "$2" "$3" "$4" "$5"
  exit 0
fi

for required in "$dataset" "$labels" "$geometry" \
  "$dataset/language_features" "$dataset/language_features_multiscale"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_OLD" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_visibility_truncation_probe.sh" \
    --prepare old "$dataset/language_features" 1 "$old_cache" \
  > "$LOG_DIR/worker_old.log" 2>&1 &
old_worker=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_L2" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_visibility_truncation_probe.sh" \
    --prepare l2 "$dataset/language_features_multiscale" 2 "$l2_cache" \
  > "$LOG_DIR/worker_l2.log" 2>&1 &
l2_worker=$!

status=0
wait "$old_worker" || status=$?
wait "$l2_worker" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "One or more visibility workers failed with status=$status" >&2
  exit "$status"
fi

if [[ ! -f "$fused" ]]; then
  "$PYTHON_BIN" -u build_split_consistency_fusion.py \
    --base_consensus "$old_cache/consensus.pt" \
    --aux_consensus "$l2_cache/consensus.pt" \
    --output "$fused" --max_aux_weight 1.5 --temperature 0.05 \
    > "$LOG_DIR/fusion.log" 2>&1
fi

if [[ ! -f "$evaluation/metrics.json" ]]; then
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$GPU_OLD" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$fused" \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --output "$evaluation" \
    > "$LOG_DIR/eval.log" 2>&1
fi

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "Waldo visibility truncation probe complete: $RUN_ROOT"
