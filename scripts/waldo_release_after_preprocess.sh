#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-waldo_kitchen}
POLL_INTERVAL=${POLL_INTERVAL:-300}
GRACE_SECONDS=${GRACE_SECONDS:-600}
LOG_DIR=${LOG_DIR:-$ROOT/logs/multigpu_waldo}
LOG_FILE=${LOG_FILE:-$LOG_DIR/09_release_after_preprocess.log}
STATUS_FILE=${STATUS_FILE:-$ROOT/waldo_kitchen_release_status.txt}

DATASET=$ROOT/drsplat_data/lerf_ovs/$SCENE
FEATURE_DIR=$DATASET/language_features
IMAGE_DIR=$DATASET/images

mkdir -p "$LOG_DIR"
cd "$ROOT"

log() {
  echo "[$(date +%FT%T)] $*" | tee -a "$LOG_FILE"
}

count_features() {
  find "$FEATURE_DIR" -name '*_f.npy' -type f 2>/dev/null | wc -l
}

count_images() {
  find "$IMAGE_DIR" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPG' -o -name '*.PNG' \) 2>/dev/null | wc -l
}

waldo_pids() {
  pgrep -f 'preprocessing.py --dataset_path .*/waldo_kitchen|gpu_guard.py --gpu [0-3].*waldo_kitchen' || true
}

write_status() {
  {
    echo "time=$(date +%F_%T)"
    echo "features=$(count_features)"
    echo "images=$(count_images)"
    echo "waldo_pids=$(waldo_pids | tr '\n' ' ')"
    echo "gpu_status="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '1,4p'
  } > "$STATUS_FILE"
}

image_count=$(count_images)
log "watching $SCENE preprocessing: feature_dir=$FEATURE_DIR image_count=$image_count"

while true; do
  feature_count=$(count_features)
  write_status
  log "progress features=$feature_count/$image_count"
  if [[ "$image_count" -gt 0 && "$feature_count" -ge "$image_count" ]]; then
    break
  fi
  sleep "$POLL_INTERVAL"
done

log "all language features are present; waiting up to ${GRACE_SECONDS}s for preprocessing/guards to exit naturally"
deadline=$((SECONDS + GRACE_SECONDS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  pids=$(waldo_pids | tr '\n' ' ')
  if [[ -z "${pids// }" ]]; then
    log "preprocessing/guards exited naturally"
    write_status
    exit 0
  fi
  log "still running after completion: $pids"
  sleep 30
done

pids=$(waldo_pids | tr '\n' ' ')
if [[ -n "${pids// }" ]]; then
  log "terminating stale waldo preprocessing/guard pids: $pids"
  kill $pids || true
  sleep 30
fi

pids=$(waldo_pids | tr '\n' ' ')
if [[ -n "${pids// }" ]]; then
  log "force-killing stale waldo preprocessing/guard pids: $pids"
  kill -9 $pids || true
fi

write_status
log "release check complete"
