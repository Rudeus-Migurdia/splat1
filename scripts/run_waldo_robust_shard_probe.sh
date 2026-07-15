#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/robust_shard_consensus/waldo_four_shards}
LOG_DIR=${LOG_DIR:-$ROOT/logs/robust_shard_consensus_waldo}
GPU_IDS=(${GPU_IDS:-0 1 2 3})

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

if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four physical GPU indices" >&2
  exit 2
fi

dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth
mkdir -p "$RUN_ROOT" "$LOG_DIR"

prepare_shard() {
  local source=$1
  local feature_dir=$2
  local feature_level=$3
  local offset=$4
  local output=$RUN_ROOT/${source}_shard${offset}
  if [[ -f "$output/manifest.json" ]]; then
    echo "[$(date +%FT%T)] source=$source shard=$offset reuse"
    return
  fi
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$output" \
    --geometry_checkpoint "$geometry" \
    --feature_dir "$feature_dir" --feature_level "$feature_level" \
    --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --consensus_only --compact_consensus --consensus_chunk_pixels 1024 \
    --view_stride 4 --view_offset "$offset" \
    > "$LOG_DIR/${source}_shard${offset}.log" 2>&1
}

if [[ "${1:-}" == "--prepare" ]]; then
  prepare_shard "$2" "$3" "$4" "$5"
  exit 0
fi

run_source_wave() {
  local source=$1
  local feature_dir=$2
  local feature_level=$3
  local pids=()
  for offset in 0 1 2 3; do
    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "${GPU_IDS[$offset]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- \
      bash "$ROOT/scripts/run_waldo_robust_shard_probe.sh" \
        --prepare "$source" "$feature_dir" "$feature_level" "$offset" \
      > "$LOG_DIR/${source}_worker${offset}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  if [[ "$status" -ne 0 ]]; then
    echo "A $source shard worker failed with status=$status" >&2
    exit "$status"
  fi
}

merge_source() {
  local source=$1
  local shards=()
  for offset in 0 1 2 3; do
    shards+=("$RUN_ROOT/${source}_shard${offset}/consensus.pt")
  done
  for method in weighted_mean geometric_median; do
    local output=$RUN_ROOT/${source}_${method}.pt
    if [[ ! -f "$output" ]]; then
      "$PYTHON_BIN" -u build_robust_shard_consensus.py \
        --shards "${shards[@]}" --method "$method" --output "$output" \
        > "$LOG_DIR/${source}_${method}.log" 2>&1
    fi
  done
}

for required in "$dataset" "$labels" "$geometry" \
  "$dataset/language_features" "$dataset/language_features_multiscale"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

run_source_wave old "$dataset/language_features" 1
merge_source old
run_source_wave l2 "$dataset/language_features_multiscale" 2
merge_source l2

for method in weighted_mean geometric_median; do
  fused=$RUN_ROOT/fused_${method}_w1p5_t005.pt
  if [[ ! -f "$fused" ]]; then
    "$PYTHON_BIN" -u build_split_consistency_fusion.py \
      --base_consensus "$RUN_ROOT/old_${method}.pt" \
      --aux_consensus "$RUN_ROOT/l2_${method}.pt" \
      --output "$fused" --max_aux_weight 1.5 --temperature 0.05 \
      > "$LOG_DIR/fuse_${method}.log" 2>&1
  fi
done

eval_pids=()
eval_index=0
for method in weighted_mean geometric_median; do
  output=$RUN_ROOT/eval_${method}_paper
  if [[ -f "$output/metrics.json" ]]; then
    continue
  fi
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "${GPU_IDS[$eval_index]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$RUN_ROOT/fused_${method}_w1p5_t005.pt" \
      --label_dir "$labels" --evaluation_protocol drsplat_3d_selection \
      --output "$output" \
    > "$LOG_DIR/eval_${method}.log" 2>&1 &
  eval_pids+=("$!")
  eval_index=$((eval_index + 1))
done

status=0
for pid in "${eval_pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "A robust-shard evaluator failed with status=$status" >&2
  exit "$status"
fi

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "Waldo robust shard probe complete: $RUN_ROOT"
