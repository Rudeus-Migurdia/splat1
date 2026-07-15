#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export PREPROCESS_NUM_SHARDS=${PREPROCESS_NUM_SHARDS:-2}

LOG_DIR=${LOG_DIR:-$ROOT/logs/multiscene_teacher_codebook}
mkdir -p "$LOG_DIR"

run_scene_inner() {
  local scene=$1
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local label_dir=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local gs_dir=$ROOT/runs/3dgs/$scene
  local drs_base=$ROOT/runs/drsplat/$scene
  local drs_ckpt=$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth
  local method_out=$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256

  echo "[$(date +%FT%T)] scene=$scene baseline start"
  DATASET_PATH="$dataset" \
  TRAINED_3DGS_PATH="$gs_dir" \
  OUTPUT_PATH="$drs_base" \
  GPU_ID=0 \
  GS_ITERATIONS="${GS_ITERATIONS:-30000}" \
  FEATURE_LEVEL=1 \
  TOPK=45 \
  RUN_PREPROCESSING=auto \
  RUN_TRAIN=1 \
  EXTRA_3DGS_ARGS="${EXTRA_3DGS_ARGS:-}" \
  EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}" \
    bash scripts/run_drsplat_baseline.sh \
    > "$LOG_DIR/${scene}_01_baseline.log" 2>&1

  echo "[$(date +%FT%T)] scene=$scene baseline eval"
  .venv/bin/python -u eval_lerf_ovs_miou.py \
    -s "$dataset" \
    -m "$(dirname "$drs_ckpt")" \
    --checkpoint "$drs_ckpt" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --label_dir "$label_dir" \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$(dirname "$drs_ckpt")/eval/lerf_ovs_miou" \
    > "$LOG_DIR/${scene}_02_baseline_eval.log" 2>&1

  echo "[$(date +%FT%T)] scene=$scene teacher-codebook method"
  SCENE="$scene" \
  DATASET="$dataset" \
  LABEL_DIR="$label_dir" \
  START_CKPT="$gs_dir/chkpnt${GS_ITERATIONS:-30000}.pth" \
  DRSPLAT_CKPT="$drs_ckpt" \
  OUT="$method_out" \
  LOG_DIR="$LOG_DIR/${scene}_method" \
    bash scripts/run_teacher_codebook_method_gpu0.sh --inner \
    > "$LOG_DIR/${scene}_03_teacher_codebook.log" 2>&1

  echo "[$(date +%FT%T)] scene=$scene done"
}

if [[ "${1:-}" == "--inner" ]]; then
  shift
  run_scene_inner "$1"
  exit 0
fi

SCENES=("$@")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=(ramen teatime waldo_kitchen)
fi

for scene in "${SCENES[@]}"; do
  echo "[$(date +%FT%T)] waiting for GPU0 before scene=$scene"
  .venv/bin/python -u scripts/gpu_guard.py \
    --gpu 0 --hold-mb 512 --max-used-mb 256 --max-utilization 5 --wait-timeout 600 --poll-interval 15 -- \
    bash -lc "set -euo pipefail; cd '$ROOT'; bash scripts/run_lerf_ovs_multiscene_gpu0.sh --inner '$scene'"
done
