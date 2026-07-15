#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/drsplat_env.sh"

GPU_ID="${GPU_ID:-0}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the COLMAP/Blender/ScanNet scene path}"
OUTPUT_PATH="${OUTPUT_PATH:-output/drsplat_baseline}"
TRAINED_3DGS_PATH="${TRAINED_3DGS_PATH:-output/3dgs_baseline}"
SAM_CKPT_PATH="${SAM_CKPT_PATH:-ckpts/sam_vit_h_4b8939.pth}"
PQ_INDEX="${PQ_INDEX:-ckpts/pq_index.faiss}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"
TOPK="${TOPK:-45}"
PORT="${PORT:-55560}"
RESOLUTION="${RESOLUTION:--1}"
PREPROCESS_ONLY_MISSING="${PREPROCESS_ONLY_MISSING:-1}"
PREPROCESS_NUM_SHARDS="${PREPROCESS_NUM_SHARDS:-1}"
TRAIN_3DGS_IF_MISSING="${TRAIN_3DGS_IF_MISSING:-1}"
GS_ITERATIONS="${GS_ITERATIONS:-30000}"
DEFAULT_START_CHECKPOINT="${TRAINED_3DGS_PATH}/chkpnt${GS_ITERATIONS}.pth"
START_CHECKPOINT="${START_CHECKPOINT:-${DEFAULT_START_CHECKPOINT}}"
RUN_PREPROCESSING="${RUN_PREPROCESSING:-auto}"
RUN_TRAIN="${RUN_TRAIN:-1}"
EXTRA_3DGS_ARGS="${EXTRA_3DGS_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  if [[ "${START_CHECKPOINT}" != "${DEFAULT_START_CHECKPOINT}" ]]; then
    echo "START_CHECKPOINT was set but does not exist: ${START_CHECKPOINT}" >&2
    echo "Unset START_CHECKPOINT to train a new 3DGS checkpoint at ${DEFAULT_START_CHECKPOINT}." >&2
    exit 1
  fi

  if [[ "${TRAIN_3DGS_IF_MISSING}" == "1" || "${TRAIN_3DGS_IF_MISSING}" == "true" ]]; then
    echo "3DGS checkpoint not found at ${START_CHECKPOINT}; training vanilla 3DGS first."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_3dgs.py \
      -s "${DATASET_PATH}" \
      -m "${TRAINED_3DGS_PATH}" \
      --iterations "${GS_ITERATIONS}" \
      --save_iterations "${GS_ITERATIONS}" \
      --checkpoint_iterations "${GS_ITERATIONS}" \
      ${EXTRA_3DGS_ARGS}
  else
    echo "Set START_CHECKPOINT, or set TRAINED_3DGS_PATH to a directory containing chkpnt30000.pth." >&2
    exit 1
  fi
fi

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "3DGS checkpoint is still missing after training: ${START_CHECKPOINT}" >&2
  exit 1
fi

python scripts/check_drsplat_ready.py \
  --dataset "${DATASET_PATH}" \
  --stage preprocess \
  --sam-checkpoint "${SAM_CKPT_PATH}" \
  --pq-index "${PQ_INDEX}" \
  --start-checkpoint "${START_CHECKPOINT}"

if [[ "${RUN_PREPROCESSING}" == "1" || "${RUN_PREPROCESSING}" == "true" || "${RUN_PREPROCESSING}" == "auto" ]]; then
  image_count=$(find "${DATASET_PATH}/images" -maxdepth 1 -type f | wc -l)
  feature_count=$(find "${DATASET_PATH}/language_features" -name '*_f.npy' -type f 2>/dev/null | wc -l || true)
  if [[ "${RUN_PREPROCESSING}" == "auto" && "${feature_count}" -ge "${image_count}" ]]; then
    echo "language_features already complete (${feature_count}/${image_count}); skipping preprocessing. Set RUN_PREPROCESSING=1 to force it."
  else
    echo "language_features incomplete (${feature_count}/${image_count}); running preprocessing with ${PREPROCESS_NUM_SHARDS} shard(s)."
    if [[ "${PREPROCESS_NUM_SHARDS}" -gt 1 ]]; then
      pids=()
      for shard_index in $(seq 0 $((PREPROCESS_NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python preprocessing.py \
          --dataset_path "${DATASET_PATH}" \
          --resolution "${RESOLUTION}" \
          --sam_ckpt_path "${SAM_CKPT_PATH}" \
          --num_shards "${PREPROCESS_NUM_SHARDS}" \
          --shard_index "${shard_index}" \
          $(if [[ "${PREPROCESS_ONLY_MISSING}" == "1" || "${PREPROCESS_ONLY_MISSING}" == "true" ]]; then printf '%s' '--only_missing'; fi) &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        wait "$pid"
      done
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python preprocessing.py \
        --dataset_path "${DATASET_PATH}" \
        --resolution "${RESOLUTION}" \
        --sam_ckpt_path "${SAM_CKPT_PATH}" \
        $(if [[ "${PREPROCESS_ONLY_MISSING}" == "1" || "${PREPROCESS_ONLY_MISSING}" == "true" ]]; then printf '%s' '--only_missing'; fi)
    fi
  fi
fi

python scripts/check_drsplat_ready.py \
  --dataset "${DATASET_PATH}" \
  --stage train \
  --sam-checkpoint "${SAM_CKPT_PATH}" \
  --pq-index "${PQ_INDEX}" \
  --start-checkpoint "${START_CHECKPOINT}"

if [[ "${RUN_TRAIN}" == "1" || "${RUN_TRAIN}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_PATH}" \
    --start_checkpoint "${START_CHECKPOINT}" \
    --feature_level "${FEATURE_LEVEL}" \
    --name_extra pq_openclip \
    --use_pq \
    --pq_index "${PQ_INDEX}" \
    --port "${PORT}" \
    --topk "${TOPK}" \
    ${EXTRA_TRAIN_ARGS}
fi
