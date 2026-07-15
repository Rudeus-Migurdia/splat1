#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 category_percentile:1:99"}
METHODS=${METHODS:-"tokpct:token_percentile:none:1.0:0.5:0.07:1.0:0:1.0:0.25 toksig:token_zscore_sigmoid:none:1.0:0.5:0.07:1.0:0:1.0:0.25 usage_res:none:usage_residual:0.5:0.5:0.07:1.0:0:1.0:0.25 cosine_temp:none:cosine:1.0:0.5:0.10:0.75:0:0.5:0.0"}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/routing_method_probe}

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
    printf "%s\n" "lerf_ovs_teacher_w0p75_codebook_k256_routing_probe"
  else
    printf "%s\n" "lerf_ovs_teacher_codebook_k256_routing_probe"
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

run_method_eval() {
  local scene=$1
  local method_name=$2
  local activation_norm=$3
  local reverse_prior=$4
  local prior_power=$5
  local residual_temp=$6
  local query_temp=$7
  local query_prior=$8
  local top_codes=$9
  local residual_weight=${10}
  local code_blend=${11}
  local mode=${12}
  local low=${13}
  local high=${14}
  local cal_tag=${15}

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
      echo "Missing routing eval input for $scene: $path" >&2
      return 1
    fi
  done

  local top_tag residual_tag blend_tag qtemp_tag qprior_tag
  top_tag=$(tag_float "$top_codes")
  residual_tag=$(tag_float "$residual_weight")
  blend_tag=$(tag_float "$code_blend")
  qtemp_tag=$(tag_float "$query_temp")
  qprior_tag=$(tag_float "$query_prior")
  local base_name="${prefix}_${method_name}_top${top_tag}_res${residual_tag}_blend${blend_tag}_qt${qtemp_tag}_qp${qprior_tag}"
  local output="$out/eval/${base_name}_cal_${cal_tag}"
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse routing scene=$scene method=$method_name cal=$cal_tag"
    return
  fi

  echo "[$(date +%FT%T)] eval routing scene=$scene method=$method_name cal=$cal_tag"
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
    --reverse_group_prior "$reverse_prior" \
    --reverse_prior_power "$prior_power" \
    --reverse_residual_temperature "$residual_temp" \
    --activation_normalization "$activation_norm" \
    --activation_norm_low 1 \
    --activation_norm_high 99 \
    --activation_norm_temperature 1.0 \
    --aggregation query_softmax \
    --score_power 1.0 \
    --query_temperature "$query_temp" \
    --query_prior_power "$query_prior" \
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
  for method in $METHODS; do
    IFS=: read -r method_name activation_norm reverse_prior prior_power residual_temp query_temp query_prior top_codes residual_weight code_blend <<< "$method"
    for calibration in $CALIBRATIONS; do
      IFS=: read -r mode low high <<< "$calibration"
      cal_tag="${mode}_l${low}_h${high}"
      run_method_eval \
        "$scene" "$method_name" "$activation_norm" "$reverse_prior" "$prior_power" "$residual_temp" \
        "$query_temp" "$query_prior" "$top_codes" "$residual_weight" "$code_blend" \
        "$mode" "$low" "$high" "$cal_tag"
    done
  done
done

echo "[$(date +%FT%T)] routing method probe done: scenes=$SCENES"
