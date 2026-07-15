#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-/home/anlanfan/Dr-Splat-envs/drsplat236_py39}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
HYBRID_ALPHAS=${HYBRID_ALPHAS:-"0.25 0.50 0.75"}
CALIBRATIONS=${CALIBRATIONS:-"none:0:100 frame_minmax:0:100 category_percentile:1:99"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/report_hybrid_probe}

cd "$ROOT"
source scripts/drsplat_env.sh
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

scene_out_dir() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
  else
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256"
  fi
}

tag_float() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_one() {
  local scene=$1
  local alpha=$2
  local cal_mode=$3
  local cal_low=$4
  local cal_high=$5
  local out
  out=$(scene_out_dir "$scene")
  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local drs_ckpt="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
  local assignments="$out/point_group_assignments.npz"
  local group_features="$out/group_features.npy"
  local alpha_tag
  alpha_tag=$(tag_float "$alpha")
  local cal_tag
  if [[ "$cal_mode" == "none" ]]; then
    cal_tag="none"
  else
    cal_tag="${cal_mode}_l$(tag_float "$cal_low")_h$(tag_float "$cal_high")"
  fi
  local eval_dir="$out/eval/lerf_ovs_hybrid_rawgroup_a${alpha_tag}_weighted_cal_${cal_tag}"

  for path in "$dataset" "$label_dir" "$drs_ckpt" "$ROOT/ckpts/pq_index.faiss" "$assignments" "$group_features"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing hybrid input for $scene: $path" >&2
      return 1
    fi
  done
  if [[ -f "$eval_dir/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse hybrid scene=$scene alpha=$alpha cal=$cal_tag"
    return 0
  fi

  echo "[$(date +%FT%T)] eval hybrid scene=$scene alpha=$alpha cal=$cal_tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_hybrid_miou.py \
    -s "$dataset" \
    -m "$out" \
    --drsplat_checkpoint "$drs_ckpt" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --label_dir "$label_dir" \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --group_aggregation weighted \
    --score_power 1.0 \
    --hybrid_alpha "$alpha" \
    --score_calibration "$cal_mode" \
    --calibration_low "$cal_low" \
    --calibration_high "$cal_high" \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$eval_dir" \
    > "$LOG_DIR/${scene}_hybrid_a${alpha_tag}_${cal_tag}.log" 2>&1
}

if [[ "${1:-}" == "--inner" ]]; then
  for scene in $SCENES; do
    for alpha in $HYBRID_ALPHAS; do
      for item in $CALIBRATIONS; do
        IFS=: read -r cal_mode cal_low cal_high <<< "$item"
        run_one "$scene" "$alpha" "$cal_mode" "$cal_low" "$cal_high"
      done
    done
  done
  echo "[$(date +%FT%T)] report hybrid probe done: scenes=$SCENES"
  exit 0
fi

"$PYTHON_BIN" -u scripts/gpu_guard.py \
  --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 -- \
  env ROOT="$ROOT" VENV_PATH="$VENV_PATH" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$GPU_ID" \
    SCENES="$SCENES" HYBRID_ALPHAS="$HYBRID_ALPHAS" CALIBRATIONS="$CALIBRATIONS" \
    LOG_DIR="$LOG_DIR" bash scripts/run_lerf_ovs_report_hybrid_probe.sh --inner
