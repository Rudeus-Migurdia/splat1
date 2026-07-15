#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/anlanfan/Dr-Splat}"
DATASET="${DATASET:-${ROOT}/drsplat_data/lerf_ovs/figurines}"
START_CHECKPOINT="${START_CHECKPOINT:-${ROOT}/runs/3dgs/figurines/chkpnt30000.pth}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/runs/drsplat/figurines}"
PQ_INDEX="${PQ_INDEX:-${ROOT}/ckpts/pq_index.faiss}"
PY="${PY:-${ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-55560}"
TOPK="${TOPK:-45}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG="${LOG:-${LOG_DIR}/drsplat_171_supervisor.log}"

mkdir -p "${LOG_DIR}" "${ROOT}/runs/drsplat"
cd "${ROOT}"

SITE="${ROOT}/.venv/lib/python3.9/site-packages"
export PYTHONPATH="${SITE}:${SITE}/setuptools/_vendor:${PYTHONPATH:-}"
export PIP_CACHE_DIR="${ROOT}/.cache/pip"
export TORCH_HOME="${ROOT}/.cache/torch"
export HF_HOME="${ROOT}/.cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

log() {
  local message
  message="[$(date '+%F %T')] $*"
  printf '%s\n' "${message}" >> "${LOG}"
  printf '%s\n' "${message}"
}

count_features() {
  find "${DATASET}/language_features" -maxdepth 1 -name '*_f.npy' 2>/dev/null | wc -l
}

count_segments() {
  find "${DATASET}/language_features" -maxdepth 1 -name '*_s.npy' 2>/dev/null | wc -l
}

wait_for_pidfile() {
  local pidfile="$1"
  if [[ ! -f "${pidfile}" ]]; then
    log "pidfile missing, skipping wait: ${pidfile}"
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}")"
  if [[ -z "${pid}" ]]; then
    log "pidfile empty, skipping wait: ${pidfile}"
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    log "waiting for ${pidfile} pid=${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 60
      log "still waiting pid=${pid}; features=$(count_features)/$(count_segments)"
    done
  else
    log "pid already exited: ${pidfile} pid=${pid}"
  fi
}

validate_language_features() {
  "${PY}" - <<'PY'
import glob
import json
import os
import sys
import numpy as np

dataset = os.environ["DATASET"]
images = sorted(glob.glob(os.path.join(dataset, "images", "*")))
images = [p for p in images if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png"}]
lf = os.path.join(dataset, "language_features")
f_paths = sorted(glob.glob(os.path.join(lf, "*_f.npy")))
s_paths = sorted(glob.glob(os.path.join(lf, "*_s.npy")))
image_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
f_stems = {os.path.basename(p).replace("_f.npy", "") for p in f_paths}
s_stems = {os.path.basename(p).replace("_s.npy", "") for p in s_paths}
paired = f_stems & s_stems
missing = sorted(image_stems - paired)
bad = []
mask_counts = []
for stem in sorted(paired):
    f = np.load(os.path.join(lf, stem + "_f.npy"), mmap_mode="r")
    s = np.load(os.path.join(lf, stem + "_s.npy"), mmap_mode="r")
    if f.ndim != 2 or f.shape[1] != 512 or f.shape[0] == 0:
        bad.append({"stem": stem, "feature_shape": tuple(f.shape)})
    if s.ndim != 3 or s.shape[0] < 2:
        bad.append({"stem": stem, "seg_shape": tuple(s.shape)})
    mask_counts.append(int(f.shape[0]))
report = {
    "images": len(images),
    "feature_files": len(f_paths),
    "segment_files": len(s_paths),
    "paired": len(paired),
    "missing_count": len(missing),
    "first_missing": missing[:20],
    "bad_count": len(bad),
    "first_bad": bad[:20],
    "mask_count_min": min(mask_counts) if mask_counts else None,
    "mask_count_max": max(mask_counts) if mask_counts else None,
    "mask_count_mean": float(np.mean(mask_counts)) if mask_counts else None,
}
print(json.dumps(report, indent=2))
if report["missing_count"] or report["bad_count"] or report["paired"] != report["images"]:
    sys.exit(2)
PY
}

log "supervisor started on $(hostname)"
log "dataset=${DATASET}"
log "waiting for preprocessing pid files"
for pidfile in "${LOG_DIR}"/preprocess_171_gpu*_shard*.pid; do
  wait_for_pidfile "${pidfile}"
done

log "preprocessing processes exited; validating feature files"
export DATASET
validate_language_features 2>&1 | tee -a "${LOG}"
log "language features validated"

TRAIN_LOG="${LOG_DIR}/drsplat_train_171_gpu${GPU_ID}.log"
log "starting Dr.Splat registration on gpu=${GPU_ID}; train log=${TRAIN_LOG}"
"${PY}" -u scripts/gpu_guard.py \
  --gpu "${GPU_ID}" \
  --hold-mb 512 \
  --max-used-mb 256 \
  --max-utilization 5 \
  -- "${PY}" -u train.py \
    -s "${DATASET}" \
    -m "${OUTPUT_BASE}" \
    --start_checkpoint "${START_CHECKPOINT}" \
    --feature_level "${FEATURE_LEVEL}" \
    --name_extra pq_openclip \
    --use_pq \
    --pq_index "${PQ_INDEX}" \
    --port "${PORT}" \
    --topk "${TOPK}" \
  2>&1 | tee -a "${TRAIN_LOG}"

log "Dr.Splat registration finished; locating output"
MODEL_DIR="$(find "${ROOT}/runs/drsplat" -maxdepth 1 -type d -name 'figurines_*pq_openclip*topk45*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${MODEL_DIR}" ]]; then
  log "could not locate Dr.Splat output directory"
  exit 3
fi
log "model_dir=${MODEL_DIR}"

CHECK_LOG="${LOG_DIR}/drsplat_check_171.log"
{
  echo "MODEL_DIR ${MODEL_DIR}"
  find "${MODEL_DIR}" -maxdepth 4 -type f \( -name 'chkpnt*.pth' -o -name 'point_cloud.ply' -o -name 'cfg_args' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
  "${PY}" - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
files = {
    "checkpoint": model_dir / "chkpnt0.pth",
    "point_cloud": model_dir / "point_cloud" / "iteration_0" / "point_cloud.ply",
    "cfg_args": model_dir / "cfg_args",
}
report = {name: {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0} for name, path in files.items()}
ply = files["point_cloud"]
if ply.exists():
    header = []
    with ply.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
    report["ply_header"] = header[:80]
    for line in header:
        if line.startswith("element vertex"):
            report["vertex_count"] = int(line.split()[-1])
            break
print(json.dumps(report, indent=2))
if not all(item["exists"] and item["size"] > 0 for item in report.values() if isinstance(item, dict) and "exists" in item):
    raise SystemExit(4)
PY
} 2>&1 | tee -a "${CHECK_LOG}"

EVAL_LOG="${LOG_DIR}/drsplat_eval_3dgs_171_gpu${GPU_ID}.log"
if [[ -f "${ROOT}/eval_3dgs.py" ]]; then
  log "running photometric sanity eval on registered checkpoint"
  "${PY}" -u scripts/gpu_guard.py \
    --gpu "${GPU_ID}" \
    --hold-mb 512 \
    --max-used-mb 256 \
    --max-utilization 5 \
    -- "${PY}" -u eval_3dgs.py \
      -s "${DATASET}" \
      -m "${MODEL_DIR}" \
      --checkpoint "${MODEL_DIR}/chkpnt0.pth" \
    2>&1 | tee -a "${EVAL_LOG}" || log "photometric eval failed; see ${EVAL_LOG}"
else
  log "eval_3dgs.py not found; skipped photometric sanity eval"
fi

log "supervisor complete"
