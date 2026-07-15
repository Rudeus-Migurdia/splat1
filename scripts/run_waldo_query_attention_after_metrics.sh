#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-waldo_kitchen}
LOG_DIR=${LOG_DIR:-$ROOT/logs/multigpu_waldo}
POLL_INTERVAL=${POLL_INTERVAL:-180}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1 2 3"}
BASELINE_METRICS=$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128/eval/lerf_ovs_miou/metrics.json
METHOD_OUT=$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256
METHOD_METRICS=$METHOD_OUT/eval/lerf_ovs_teacher_codebook_k256_weighted/metrics.json
COARSE_SWEEP_DONE=$METHOD_OUT/eval/lerf_ovs_teacher_codebook_k256_coarse_fine_sweep.done
SWEEP_DONE=$METHOD_OUT/eval/lerf_ovs_teacher_codebook_k256_query_attention_sweep.done

mkdir -p "$LOG_DIR"
cd "$ROOT"
source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"

if [[ -f "$SWEEP_DONE" ]]; then
  echo "[$(date +%FT%T)] query-attention sweep already done: $SWEEP_DONE"
  exit 0
fi

while [[ ! -f "$BASELINE_METRICS" || ! -f "$METHOD_METRICS" ]]; do
  echo "[$(date +%FT%T)] waiting waldo metrics for query attention: baseline=$([[ -f "$BASELINE_METRICS" ]] && echo yes || echo no) method=$([[ -f "$METHOD_METRICS" ]] && echo yes || echo no)"
  sleep "$POLL_INTERVAL"
done

while [[ ! -f "$COARSE_SWEEP_DONE" ]]; do
  echo "[$(date +%FT%T)] waiting coarse/fine weighted sweep before query attention: coarse_done=no"
  sleep "$POLL_INTERVAL"
done

echo "[$(date +%FT%T)] waldo metrics ready; looking for idle GPU for query attention in: $GPU_CANDIDATES"
while true; do
  for gpu in $GPU_CANDIDATES; do
    if "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "$gpu" --hold-mb 512 --max-used-mb 256 --max-utilization 5 --wait-timeout 1 --poll-interval 1 -- \
      bash -lc "set -euo pipefail; cd '$ROOT'; ROOT='$ROOT' VENV_PATH='$VENV_PATH' PYTHON_BIN='$PYTHON_BIN' SCENE='$SCENE' GPU_ID='$gpu' AGGREGATIONS='query_softmax' BLEND_MODES='fixed query_adaptive' bash scripts/run_coarse_fine_sweep.sh" \
      > "$LOG_DIR/08_waldo_query_attention_gpu${gpu}.log" 2>&1; then
      "$PYTHON_BIN" scripts/write_lerf_ovs_experiment_report.py \
        --baseline "$BASELINE_METRICS" \
        --output "$METHOD_OUT/eval/waldo_sweep_report.md" \
        "$BASELINE_METRICS" "$METHOD_OUT/eval" \
        > "$LOG_DIR/08_waldo_report.log" 2>&1
      touch "$SWEEP_DONE"
      echo "[$(date +%FT%T)] query-attention sweep finished on gpu $gpu"
      exit 0
    fi
    echo "[$(date +%FT%T)] gpu $gpu not available or query-attention sweep failed; see $LOG_DIR/08_waldo_query_attention_gpu${gpu}.log"
  done
  echo "[$(date +%FT%T)] no idle GPU for query-attention sweep; retrying in $POLL_INTERVAL seconds"
  sleep "$POLL_INTERVAL"
done
