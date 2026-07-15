#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/anlanfan/Dr-Splat}"
DATASET="${DATASET:-${ROOT}/drsplat_data/lerf_ovs/figurines}"
SAM_CKPT="${SAM_CKPT:-ckpts/sam_vit_h_4b8939.pth}"
PY="${PY:-${ROOT}/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
NUM_SHARDS="${NUM_SHARDS:-4}"
THREADS_PER_PROC="${THREADS_PER_PROC:-2}"

mkdir -p "${LOG_DIR}" "${DATASET}/language_features"
cd "${ROOT}"

SITE="${ROOT}/.venv/lib/python3.9/site-packages"

for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU="${SHARD}"
  LOG="${LOG_DIR}/preprocess_171_gpu${GPU}_shard${SHARD}.log"
  PIDFILE="${LOG_DIR}/preprocess_171_gpu${GPU}_shard${SHARD}.pid"
  printf 'START gpu=%s shard=%s threads=%s time=%s\n' \
    "${GPU}" "${SHARD}" "${THREADS_PER_PROC}" "$(date)" >> "${LOG}"
  nohup bash -lc "
    cd '${ROOT}' &&
    source scripts/drsplat_env.sh &&
    export PYTHONPATH='${SITE}:${SITE}/setuptools/_vendor' &&
    export PIP_CACHE_DIR='${ROOT}/.cache/pip' &&
    export TORCH_HOME='${ROOT}/.cache/torch' &&
    export HF_HOME='${ROOT}/.cache/huggingface' &&
    export HF_ENDPOINT='https://hf-mirror.com' &&
    export HF_HUB_OFFLINE=1 &&
    export HF_HUB_DISABLE_TELEMETRY=1 &&
    export PYTHONUNBUFFERED=1 &&
    export OMP_NUM_THREADS='${THREADS_PER_PROC}' &&
    export MKL_NUM_THREADS='${THREADS_PER_PROC}' &&
    export OPENBLAS_NUM_THREADS='${THREADS_PER_PROC}' &&
    export NUMEXPR_NUM_THREADS='${THREADS_PER_PROC}' &&
    '${PY}' -u scripts/gpu_guard.py \
      --gpu '${GPU}' \
      --hold-mb 512 \
      --max-used-mb 256 \
      --max-utilization 5 \
      -- '${PY}' -u preprocessing.py \
        --dataset_path '${DATASET}' \
        --sam_ckpt_path '${SAM_CKPT}' \
        --num_shards '${NUM_SHARDS}' \
        --shard_index '${SHARD}' \
        --only_missing
  " >> "${LOG}" 2>&1 < /dev/null &
  echo $! > "${PIDFILE}"
done

printf 'Started %s preprocessing shards.\n' "${NUM_SHARDS}"
