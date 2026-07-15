#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime"}
CALIBRATIONS=${CALIBRATIONS:-"none:1:99 frame_minmax:0:100 frame_percentile:1:99 category_percentile:1:99"}
REVERSE_CONFIGS=${REVERSE_CONFIGS:-"query_softmax:0:0.5:0.0 query_softmax:0:1.0:0.25 query_softmax:64:1.0:0.25"}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/reverse_calibration_probe}

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

eval_prefix() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "lerf_ovs_teacher_w0p75_codebook_k256_reverse_probe"
  else
    printf "%s\n" "lerf_ovs_teacher_codebook_k256_reverse_probe"
  fi
}

tag_float() {
  printf "%s\n" "${1/./p}"
}

ensure_reverse_index() {
  local scene=$1
  local out=$2
  local assignments="$out/point_group_assignments.npz"
  local group_features="$out/teacher_distilled/group_features_teacher_w0p75.npy"
  local codebook_dir="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage"
  local reverse_dir="$out/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage"

  for path in "$assignments" "$group_features" "$codebook_dir/codebook.npy" "$codebook_dir/group_to_code.npy"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing reverse-index input for $scene: $path" >&2
      return 1
    fi
  done

  mkdir -p "$reverse_dir"
  if [[ -f "$reverse_dir/reverse_codebook_summary.json" ]]; then
    echo "[$(date +%FT%T)] reuse reverse index scene=$scene"
    return
  fi

  echo "[$(date +%FT%T)] build reverse index scene=$scene"
  "$PYTHON_BIN" build_reverse_codebook_index.py \
    --codebook_dir "$codebook_dir" \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --output_dir "$reverse_dir" \
    > "$LOG_DIR/${scene}_00_build_reverse_index.log" 2>&1
}

run_reverse_eval() {
  local scene=$1
  local aggregation=$2
  local top_codes=$3
  local residual_weight=$4
  local code_blend=$5
  local mode=$6
  local low=$7
  local high=$8
  local cal_tag=$9

  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local out
  out=$(scene_method_out "$scene")
  local checkpoint="$out/chkpnt0.pth"
  local assignments="$out/point_group_assignments.npz"
  local group_features="$out/teacher_distilled/group_features_teacher_w0p75.npy"
  local reverse_dir="$out/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage"
  local prefix
  prefix=$(eval_prefix "$scene")

  for path in "$checkpoint" "$assignments" "$group_features" "$label_dir" "$reverse_dir/reverse_codebook_summary.json"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing reverse eval input for $scene: $path" >&2
      return 1
    fi
  done

  local top_tag residual_tag blend_tag
  top_tag=$(tag_float "$top_codes")
  residual_tag=$(tag_float "$residual_weight")
  blend_tag=$(tag_float "$code_blend")
  local base_name="${prefix}_${aggregation}_top${top_tag}_res${residual_tag}_codeblend${blend_tag}"
  local output="$out/eval/${base_name}"
  if [[ "$mode" != "none" ]]; then
    output="$out/eval/${base_name}_cal_${cal_tag}"
  fi
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse reverse scene=$scene cfg=$base_name cal=$cal_tag"
    return
  fi

  echo "[$(date +%FT%T)] eval reverse scene=$scene cfg=$base_name cal=$cal_tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$dataset" -m "$out" \
    --checkpoint "$checkpoint" \
    --label_dir "$label_dir" \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --reverse_codebook_dir "$reverse_dir" \
    --reverse_top_codes "$top_codes" \
    --reverse_residual_weight "$residual_weight" \
    --reverse_code_blend "$code_blend" \
    --aggregation "$aggregation" \
    --score_power 1.0 \
    --query_temperature "$QUERY_TEMPERATURE" \
    --query_prior_power "$QUERY_PRIOR_POWER" \
    --thresholds $THRESHOLDS \
    --score_calibration "$mode" \
    --calibration_low "$low" \
    --calibration_high "$high" \
    --output "$output" \
    > "$LOG_DIR/${scene}_${base_name}_${cal_tag}.log" 2>&1
}

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  ensure_reverse_index "$scene" "$out"
  for config in $REVERSE_CONFIGS; do
    IFS=: read -r aggregation top_codes residual_weight code_blend <<< "$config"
    for calibration in $CALIBRATIONS; do
      IFS=: read -r mode low high <<< "$calibration"
      cal_tag="${mode}_l${low}_h${high}"
      run_reverse_eval "$scene" "$aggregation" "$top_codes" "$residual_weight" "$code_blend" "$mode" "$low" "$high" "$cal_tag"
    done
  done
done

echo "[$(date +%FT%T)] reverse calibration probe done: scenes=$SCENES"
