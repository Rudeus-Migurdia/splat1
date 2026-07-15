#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=waldo_kitchen
GPU_ID=${GPU_ID:-1}
DATASET=$ROOT/drsplat_data/lerf_ovs/$SCENE
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/$SCENE
GS_DIR=$ROOT/runs/3dgs/$SCENE
GS_CKPT=$GS_DIR/chkpnt30000.pth
DRS_BASE=$ROOT/runs/drsplat/$SCENE
DRS_DIR=$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128
DRS_CKPT=$DRS_DIR/chkpnt0.pth
METHOD_OUT=$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256
LOG_DIR=$ROOT/logs/multigpu_waldo

cd "$ROOT"
mkdir -p "$LOG_DIR"

source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

image_count() {
  find "$DATASET/images" -maxdepth 1 -type f | wc -l
}

feature_count() {
  find "$DATASET/language_features" -name '*_f.npy' -type f 2>/dev/null | wc -l || true
}

run_inner() {
  if [[ ! -f "$DRS_CKPT" ]]; then
    echo "[$(date +%FT%T)] waldo baseline train"
    DATASET_PATH="$DATASET" \
    TRAINED_3DGS_PATH="$GS_DIR" \
    START_CHECKPOINT="$GS_CKPT" \
    OUTPUT_PATH="$DRS_BASE" \
    GPU_ID="$GPU_ID" \
    TRAIN_3DGS_IF_MISSING=0 \
    FEATURE_LEVEL=1 \
    TOPK=45 \
    PORT=55564 \
    RUN_PREPROCESSING=auto \
    PREPROCESS_ONLY_MISSING=1 \
    RUN_TRAIN=1 \
      bash scripts/run_drsplat_baseline.sh \
      > "$LOG_DIR/03_waldo_baseline.log" 2>&1
  else
    echo "[$(date +%FT%T)] waldo baseline checkpoint exists; skip train"
  fi

  if [[ ! -f "$DRS_DIR/eval/lerf_ovs_miou/metrics.json" ]]; then
    echo "[$(date +%FT%T)] waldo baseline eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_miou.py \
      -s "$DATASET" \
      -m "$DRS_DIR" \
      --checkpoint "$DRS_CKPT" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --label_dir "$LABEL_DIR" \
      --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
      --output "$DRS_DIR/eval/lerf_ovs_miou" \
      > "$LOG_DIR/04_waldo_baseline_eval.log" 2>&1
  else
    echo "[$(date +%FT%T)] waldo baseline metrics exist; skip eval"
  fi

  if [[ ! -f "$METHOD_OUT/eval/lerf_ovs_teacher_codebook_k256_weighted/metrics.json" ]]; then
    echo "[$(date +%FT%T)] waldo teacher-codebook method"
    SCENE="$SCENE" \
    DATASET="$DATASET" \
    LABEL_DIR="$LABEL_DIR" \
    START_CKPT="$GS_CKPT" \
    DRSPLAT_CKPT="$DRS_CKPT" \
    OUT="$METHOD_OUT" \
    LOG_DIR="$LOG_DIR/waldo_method" \
    ROOT="$ROOT" \
    VENV_PATH="$VENV_PATH" \
    PYTHON_BIN="$PYTHON_BIN" \
    GPU_ID="$GPU_ID" \
      bash scripts/run_teacher_codebook_method_gpu0.sh --inner \
      > "$LOG_DIR/05_waldo_teacher_codebook.log" 2>&1
  else
    echo "[$(date +%FT%T)] waldo method metrics exist; skip method"
  fi

  echo "[$(date +%FT%T)] waldo postprocess done"
}

if [[ "${1:-}" == "--inner" ]]; then
  run_inner
  exit 0
fi

target_images=$(image_count)
while true; do
  features=$(feature_count)
  if [[ -f "$GS_CKPT" && "$features" -ge "$target_images" ]]; then
    echo "[$(date +%FT%T)] waldo ready: features=$features/$target_images ckpt=$GS_CKPT"
    break
  fi
  echo "[$(date +%FT%T)] waiting waldo: features=$features/$target_images ckpt=$([[ -f "$GS_CKPT" ]] && echo yes || echo no)"
  sleep 120
done

"$PYTHON_BIN" -u scripts/gpu_guard.py \
  --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 --wait-timeout 1800 --poll-interval 30 -- \
  bash -lc "set -euo pipefail; cd '$ROOT'; ROOT='$ROOT' VENV_PATH='$VENV_PATH' PYTHON_BIN='$PYTHON_BIN' GPU_ID='$GPU_ID' bash scripts/run_waldo_after_ready_gpu1.sh --inner"
