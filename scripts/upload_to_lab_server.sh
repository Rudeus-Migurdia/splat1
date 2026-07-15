#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${SSH_TARGET:-10.105.100.170}"
REMOTE_ROOT="${REMOTE_ROOT:-/volume1/cvnext/datasets}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_ROOT}/Dr-Splat}"
DATASET_PATH="${DATASET_PATH:-}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-${REMOTE_ROOT}/drsplat_data}"

RSYNC_SSH_OPTS="${RSYNC_SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=10}"

rsync_common=(
  -avh --info=progress2
  -e "ssh ${RSYNC_SSH_OPTS}"
)

repo_excludes=(
  --exclude ".git/"
  --exclude ".cache/"
  --exclude ".conda/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "output/"
  --exclude "wandb/"
  --exclude "data/"
  --exclude "datasets/"
)

ssh ${RSYNC_SSH_OPTS} "${SSH_TARGET}" "mkdir -p '${REMOTE_PROJECT_DIR}' '${REMOTE_DATASET_DIR}'"

rsync "${rsync_common[@]}" "${repo_excludes[@]}" ./ "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/"

if [[ -n "${DATASET_PATH}" ]]; then
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "DATASET_PATH does not exist or is not a directory: ${DATASET_PATH}" >&2
    exit 1
  fi
  rsync "${rsync_common[@]}" "${DATASET_PATH%/}/" "${SSH_TARGET}:${REMOTE_DATASET_DIR}/"
else
  echo "DATASET_PATH is empty; uploaded code only."
fi

ssh ${RSYNC_SSH_OPTS} "${SSH_TARGET}" "du -sh '${REMOTE_PROJECT_DIR}' '${REMOTE_DATASET_DIR}' 2>/dev/null || true"
