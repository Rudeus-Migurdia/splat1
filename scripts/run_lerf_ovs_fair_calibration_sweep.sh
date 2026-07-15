#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
TARGETS=${TARGETS:-"baseline teacher_codebook"}
CALIBRATIONS=${CALIBRATIONS:-"none:1:99 frame_minmax:0:100 frame_percentile:1:99 category_percentile:1:99"}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/fair_calibration_sweep}

cd "$ROOT"
source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
export WANDB_MODE=offline

mkdir -p "$LOG_DIR"

scene_method_out() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
  else
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256"
  fi
}

scene_method_eval_prefix() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "lerf_ovs_teacher_w0p75_codebook_k256_weighted"
  else
    printf "%s\n" "lerf_ovs_teacher_codebook_k256_weighted"
  fi
}

run_baseline() {
  local scene=$1
  local mode=$2
  local low=$3
  local high=$4
  local tag=$5
  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local drs_dir="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128"
  local eval_name="lerf_ovs_miou_cal_${tag}"
  if [[ "$mode" == "none" ]]; then
    eval_name="lerf_ovs_miou"
  fi
  local output="$drs_dir/eval/$eval_name"
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse baseline scene=$scene cal=$tag"
    return
  fi
  for path in "$drs_dir/chkpnt0.pth" "$ROOT/ckpts/pq_index.faiss" "$label_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing baseline input for $scene: $path" >&2
      return 1
    fi
  done
  echo "[$(date +%FT%T)] eval baseline scene=$scene cal=$tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_miou.py \
    -s "$dataset" -m "$drs_dir" \
    --checkpoint "$drs_dir/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --label_dir "$label_dir" \
    --thresholds $THRESHOLDS \
    --score_calibration "$mode" \
    --calibration_low "$low" \
    --calibration_high "$high" \
    --output "$output" \
    > "$LOG_DIR/${scene}_baseline_${tag}.log" 2>&1
}

run_teacher_codebook() {
  local scene=$1
  local mode=$2
  local low=$3
  local high=$4
  local tag=$5
  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local out
  out=$(scene_method_out "$scene")
  local eval_prefix
  eval_prefix=$(scene_method_eval_prefix "$scene")
  local output="$out/eval/${eval_prefix}_cal_${tag}"
  if [[ "$mode" == "none" ]]; then
    output="$out/eval/$eval_prefix"
  fi
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse teacher_codebook scene=$scene cal=$tag"
    return
  fi
  local group_features="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy"
  for path in "$out/chkpnt0.pth" "$out/point_group_assignments.npz" "$group_features" "$label_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing teacher-codebook input for $scene: $path" >&2
      return 1
    fi
  done
  echo "[$(date +%FT%T)] eval teacher_codebook scene=$scene cal=$tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$dataset" -m "$out" \
    --checkpoint "$out/chkpnt0.pth" \
    --label_dir "$label_dir" \
    --group_features "$group_features" \
    --assignments "$out/point_group_assignments.npz" \
    --aggregation weighted \
    --score_power 1.0 \
    --thresholds $THRESHOLDS \
    --score_calibration "$mode" \
    --calibration_low "$low" \
    --calibration_high "$high" \
    --output "$output" \
    > "$LOG_DIR/${scene}_teacher_codebook_${tag}.log" 2>&1
}

for scene in $SCENES; do
  for calibration in $CALIBRATIONS; do
    IFS=: read -r mode low high <<< "$calibration"
    tag="${mode}_l${low}_h${high}"
    for target in $TARGETS; do
      case "$target" in
        baseline)
          run_baseline "$scene" "$mode" "$low" "$high" "$tag"
          ;;
        teacher_codebook)
          run_teacher_codebook "$scene" "$mode" "$low" "$high" "$tag"
          ;;
        *)
          echo "Unknown target: $target" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "[$(date +%FT%T)] fair calibration sweep done: scenes=$SCENES targets=$TARGETS"
