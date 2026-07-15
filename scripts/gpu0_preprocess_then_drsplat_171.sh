#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/anlanfan/Dr-Splat}"
DATASET="${DATASET:-${ROOT}/drsplat_data/lerf_ovs/figurines}"
START_CHECKPOINT="${START_CHECKPOINT:-${ROOT}/runs/3dgs/figurines/chkpnt30000.pth}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/runs/drsplat/figurines}"
PQ_INDEX="${PQ_INDEX:-${ROOT}/ckpts/pq_index.faiss}"
SAM_CKPT="${SAM_CKPT:-ckpts/sam_vit_h_4b8939.pth}"
PY="${PY:-${ROOT}/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG="${LOG:-${LOG_DIR}/gpu0_chain_171.log}"
THREADS_PER_PROC="${THREADS_PER_PROC:-2}"
TOPK="${TOPK:-45}"
PORT="${PORT:-55560}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"

mkdir -p "${LOG_DIR}" "${DATASET}/language_features" "${ROOT}/runs/drsplat"
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
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"
}

count_features() {
  find "${DATASET}/language_features" -maxdepth 1 -name '*_f.npy' 2>/dev/null | wc -l
}

count_segments() {
  find "${DATASET}/language_features" -maxdepth 1 -name '*_s.npy' 2>/dev/null | wc -l
}

validate_language_features() {
  DATASET="${DATASET}" "${PY}" - <<'PY'
import glob
import json
import os
import sys

dataset = os.environ["DATASET"]
images = [
    p for p in sorted(glob.glob(os.path.join(dataset, "images", "*")))
    if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png"}
]
lf = os.path.join(dataset, "language_features")
f_stems = {os.path.basename(p).replace("_f.npy", "") for p in glob.glob(os.path.join(lf, "*_f.npy"))}
s_stems = {os.path.basename(p).replace("_s.npy", "") for p in glob.glob(os.path.join(lf, "*_s.npy"))}
image_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
paired = f_stems & s_stems
missing = sorted(image_stems - paired)
report = {
    "images": len(images),
    "feature_files": len(f_stems),
    "segment_files": len(s_stems),
    "paired": len(paired),
    "missing_count": len(missing),
    "first_missing": missing[:20],
}
print(json.dumps(report, indent=2))
if missing or len(paired) != len(images):
    sys.exit(2)
PY
}

log "GPU0 chain started; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "step 1: run shard0 preprocessing while guard keeps GPU0 reserved"
"${PY}" -u preprocessing.py \
  --dataset_path "${DATASET}" \
  --sam_ckpt_path "${SAM_CKPT}" \
  --num_shards 4 \
  --shard_index 0 \
  --only_missing 2>&1 | tee -a "${LOG}"

log "shard0 finished; keeping GPU0 reserved and waiting for all language features"
while true; do
  f_count="$(count_features)"
  s_count="$(count_segments)"
  log "feature progress ${f_count}/${s_count}"
  if [[ "${f_count}" -ge 299 && "${s_count}" -ge 299 ]]; then
    break
  fi
  sleep 60
done

log "validating language features"
validate_language_features 2>&1 | tee -a "${LOG}"

TRAIN_LOG="${LOG_DIR}/drsplat_train_171_gpu0.log"
log "step 2: run Dr.Splat registration without releasing GPU0"
"${PY}" -u train.py \
  -s "${DATASET}" \
  -m "${OUTPUT_BASE}" \
  --start_checkpoint "${START_CHECKPOINT}" \
  --feature_level "${FEATURE_LEVEL}" \
  --name_extra pq_openclip \
  --use_pq \
  --pq_index "${PQ_INDEX}" \
  --port "${PORT}" \
  --topk "${TOPK}" 2>&1 | tee -a "${TRAIN_LOG}"

MODEL_DIR="$(find "${ROOT}/runs/drsplat" -maxdepth 1 -type d -name 'figurines_*pq_openclip*topk45*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${MODEL_DIR}" ]]; then
  log "could not locate Dr.Splat output directory"
  exit 3
fi
export MODEL_DIR
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
print(json.dumps(report, indent=2))
if not all(item["exists"] and item["size"] > 0 for item in report.values()):
    raise SystemExit(4)
PY
} 2>&1 | tee -a "${CHECK_LOG}"

if [[ -f "${ROOT}/eval_3dgs.py" ]]; then
  EVAL_LOG="${LOG_DIR}/drsplat_eval_3dgs_171_gpu0.log"
  log "step 3: run photometric sanity eval on registered checkpoint"
  "${PY}" -u eval_3dgs.py \
    -s "${DATASET}" \
    -m "${MODEL_DIR}" \
    --checkpoint "${MODEL_DIR}/chkpnt0.pth" \
    2>&1 | tee -a "${EVAL_LOG}" || log "photometric eval failed; see ${EVAL_LOG}"
fi

log "GPU0 chain complete"
