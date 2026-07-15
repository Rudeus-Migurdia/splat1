#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-waldo_kitchen}
GPU_ID=${GPU_ID:-0}
GPU_CANDIDATES=${GPU_CANDIDATES:-$GPU_ID}
RUN_BASELINE_AND_METHOD=${RUN_BASELINE_AND_METHOD:-1}
RUN_COARSE_FINE=${RUN_COARSE_FINE:-1}
RUN_QUERY_ATTENTION=${RUN_QUERY_ATTENTION:-1}
RUN_REVERSE_CODEBOOK=${RUN_REVERSE_CODEBOOK:-1}
WRITE_REPORT=${WRITE_REPORT:-1}

cd "$ROOT"
source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline

DATASET=$ROOT/drsplat_data/lerf_ovs/$SCENE
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/$SCENE
BASELINE_METRICS=$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128/eval/lerf_ovs_miou/metrics.json
METHOD_OUT=$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256
METHOD_METRICS=$METHOD_OUT/eval/lerf_ovs_teacher_codebook_k256_weighted/metrics.json
REPORT=$METHOD_OUT/eval/waldo_sweep_report.md
LOG_DIR=$ROOT/logs/multigpu_waldo

mkdir -p "$LOG_DIR"

echo "[$(date +%FT%T)] requested GPU_ID=$GPU_ID candidates=$GPU_CANDIDATES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

image_count=$(find "$DATASET/images" -maxdepth 1 -type f | wc -l)
feature_count=$(find "$DATASET/language_features" -name '*_f.npy' -type f 2>/dev/null | wc -l || true)
if [[ "$image_count" -eq 0 || "$feature_count" -lt "$image_count" ]]; then
  echo "waldo language features are incomplete: features=$feature_count images=$image_count" >&2
  exit 1
fi
if [[ ! -d "$LABEL_DIR" ]]; then
  echo "Missing label dir: $LABEL_DIR" >&2
  exit 1
fi

if [[ "$RUN_BASELINE_AND_METHOD" == "1" ]]; then
  echo "[$(date +%FT%T)] run waldo baseline and teacher-codebook if missing"
  ROOT="$ROOT" VENV_PATH="$VENV_PATH" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$GPU_ID" bash scripts/run_waldo_after_ready_gpu1.sh
fi

if [[ "$RUN_COARSE_FINE" == "1" ]]; then
  echo "[$(date +%FT%T)] run waldo coarse/fine sweep if missing"
  ROOT="$ROOT" VENV_PATH="$VENV_PATH" PYTHON_BIN="$PYTHON_BIN" GPU_CANDIDATES="$GPU_CANDIDATES" bash scripts/run_waldo_coarse_fine_after_metrics.sh
fi

if [[ "$RUN_QUERY_ATTENTION" == "1" ]]; then
  echo "[$(date +%FT%T)] run waldo query-attention sweep if missing"
  ROOT="$ROOT" VENV_PATH="$VENV_PATH" PYTHON_BIN="$PYTHON_BIN" GPU_CANDIDATES="$GPU_CANDIDATES" bash scripts/run_waldo_query_attention_after_metrics.sh
fi

if [[ "$RUN_REVERSE_CODEBOOK" == "1" ]]; then
  while [[ ! -f "$METHOD_METRICS" ]]; do
    echo "[$(date +%FT%T)] waiting method metrics before reverse codebook: method=no"
    sleep 180
  done
  echo "[$(date +%FT%T)] run waldo reverse-mounted codebook sweep"
  "$PYTHON_BIN" -u scripts/gpu_guard.py \
    --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 --wait-timeout 1800 --poll-interval 30 -- \
    bash -lc "set -euo pipefail; cd '$ROOT'; ROOT='$ROOT' VENV_PATH='$VENV_PATH' PYTHON_BIN='$PYTHON_BIN' SCENE='$SCENE' GPU_ID='$GPU_ID' bash scripts/run_reverse_codebook_sweep.sh" \
    > "$LOG_DIR/10_waldo_reverse_codebook_gpu${GPU_ID}.log" 2>&1
fi

if [[ "$WRITE_REPORT" == "1" && -f "$BASELINE_METRICS" ]]; then
  echo "[$(date +%FT%T)] write waldo sweep report"
  "$PYTHON_BIN" scripts/write_lerf_ovs_experiment_report.py \
    --baseline "$BASELINE_METRICS" \
    --output "$REPORT" \
    "$BASELINE_METRICS" "$METHOD_OUT/eval" \
    > "$LOG_DIR/11_waldo_report.log" 2>&1
fi

echo "[$(date +%FT%T)] waldo GPU-ready suite finished"
