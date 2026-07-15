#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/anlanfan/Dr-Splat}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
PY="${PY:-${ROOT}/.venv/bin/python}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

LOG="${LOG_DIR}/gpu0_chain_guard_171.log"
PIDFILE="${LOG_DIR}/gpu0_chain_guard_171.pid"
: > "${LOG_DIR}/gpu0_chain_171.log"

nohup bash -lc '
  cd /home/anlanfan/Dr-Splat
  source scripts/drsplat_env.sh
  SITE=/home/anlanfan/Dr-Splat/.venv/lib/python3.9/site-packages
  export PYTHONPATH="${SITE}:${SITE}/setuptools/_vendor"
  export PIP_CACHE_DIR=/home/anlanfan/Dr-Splat/.cache/pip
  export TORCH_HOME=/home/anlanfan/Dr-Splat/.cache/torch
  export HF_HOME=/home/anlanfan/Dr-Splat/.cache/huggingface
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2
  /home/anlanfan/Dr-Splat/.venv/bin/python -u scripts/gpu_guard.py \
    --gpu 0 \
    --hold-mb 512 \
    --max-used-mb 256 \
    --max-utilization 5 \
    -- bash scripts/gpu0_preprocess_then_drsplat_171.sh
' >> "${LOG}" 2>&1 < /dev/null &

echo $! > "${PIDFILE}"
printf 'Started GPU0 chain launcher pid=%s\n' "$(cat "${PIDFILE}")"
