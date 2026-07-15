#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
COARSE_CODES=${COARSE_CODES:-32}
FINE_CODES=${FINE_CODES:-8}
AGGREGATIONS=${AGGREGATIONS:-"weighted query_softmax"}
BLEND_MODES=${BLEND_MODES:-"fixed query_adaptive"}
BLENDS=${BLENDS:-"0.15 0.30"}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 category_percentile:1:99"}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
EVAL_HIERARCHICAL_RECON=${EVAL_HIERARCHICAL_RECON:-1}
COARSE_MIN_BLEND=${COARSE_MIN_BLEND:-0.05}
COARSE_SPECIFICITY_TOPK=${COARSE_SPECIFICITY_TOPK:-16}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/latest_pending_probe}

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
    printf "%s\n" "lerf_ovs_teacher_w0p75_codebook_k256"
  else
    printf "%s\n" "lerf_ovs_teacher_codebook_k256"
  fi
}

tag_float() {
  printf "%s\n" "${1/./p}"
}

ensure_hierarchical_tokens() {
  local scene=$1
  local out=$2
  local group_features="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy"
  local assignments="$out/point_group_assignments.npz"
  local hier_dir="$out/teacher_distilled/hier_k${COARSE_CODES}_f${FINE_CODES}_from_codebook"
  local scene_log_dir="$LOG_DIR/$scene"

  mkdir -p "$hier_dir" "$scene_log_dir"
  for path in "$group_features" "$assignments"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing hierarchical input for $scene: $path" >&2
      return 1
    fi
  done

  if [[ -f "$hier_dir/coarse_codebook.npy" && -f "$hier_dir/coarse_ids.npy" && -f "$hier_dir/group_features_hierarchical.npy" ]]; then
    echo "[$(date +%FT%T)] reuse hierarchical tokens scene=$scene"
    return
  fi

  echo "[$(date +%FT%T)] build hierarchical tokens scene=$scene"
  "$PYTHON_BIN" hierarchical_group_codebook.py \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --coarse_codes "$COARSE_CODES" \
    --fine_codes "$FINE_CODES" \
    --iterations 120 \
    --seed 71 \
    --usage_weighted \
    --output_dir "$hier_dir" \
    > "$scene_log_dir/00_hierarchical_codebook.log" 2>&1
}

run_eval() {
  local scene=$1
  local tag=$2
  local output=$3
  shift 3

  local out
  out=$(scene_method_out "$scene")
  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local checkpoint="$out/chkpnt0.pth"
  local assignments="$out/point_group_assignments.npz"
  local scene_log_dir="$LOG_DIR/$scene"
  mkdir -p "$scene_log_dir"

  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse latest scene=$scene tag=$tag"
    return
  fi
  for path in "$checkpoint" "$assignments" "$label_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing latest eval input for $scene: $path" >&2
      return 1
    fi
  done

  echo "[$(date +%FT%T)] eval latest scene=$scene tag=$tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$dataset" -m "$out" \
    --checkpoint "$checkpoint" \
    --label_dir "$label_dir" \
    --assignments "$assignments" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    "$@" \
    > "$scene_log_dir/${tag}.log" 2>&1
}

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  prefix=$(eval_prefix "$scene")
  ensure_hierarchical_tokens "$scene" "$out"
  codebook_features="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy"
  hier_dir="$out/teacher_distilled/hier_k${COARSE_CODES}_f${FINE_CODES}_from_codebook"

  for calibration in $CALIBRATIONS; do
    IFS=: read -r mode low high <<< "$calibration"
    cal_tag="${mode}_l${low}_h${high}"

    for aggregation in $AGGREGATIONS; do
      if [[ "$EVAL_HIERARCHICAL_RECON" == "1" ]]; then
        run_eval "$scene" "hier_${aggregation}_${cal_tag}" \
          "$out/eval/${prefix}_${aggregation}_hier_k${COARSE_CODES}_f${FINE_CODES}_cal_${cal_tag}" \
          --group_features "$hier_dir/group_features_hierarchical.npy" \
          --aggregation "$aggregation" \
          --score_power 1.0 \
          --query_temperature "$QUERY_TEMPERATURE" \
          --query_prior_power "$QUERY_PRIOR_POWER" \
          --score_calibration "$mode" \
          --calibration_low "$low" \
          --calibration_high "$high"
      fi

      for blend_mode in $BLEND_MODES; do
        for blend in $BLENDS; do
          blend_tag=$(tag_float "$blend")
          run_eval "$scene" "coarse_${aggregation}_${blend_mode}_${blend_tag}_${cal_tag}" \
            "$out/eval/${prefix}_${aggregation}_coarse${COARSE_CODES}_${blend_mode}_blend_${blend_tag}_cal_${cal_tag}" \
            --group_features "$codebook_features" \
            --coarse_features "$hier_dir/coarse_codebook.npy" \
            --group_to_coarse "$hier_dir/coarse_ids.npy" \
            --coarse_blend "$blend" \
            --coarse_blend_mode "$blend_mode" \
            --coarse_min_blend "$COARSE_MIN_BLEND" \
            --coarse_specificity_topk "$COARSE_SPECIFICITY_TOPK" \
            --aggregation "$aggregation" \
            --score_power 1.0 \
            --query_temperature "$QUERY_TEMPERATURE" \
            --query_prior_power "$QUERY_PRIOR_POWER" \
            --score_calibration "$mode" \
            --calibration_low "$low" \
            --calibration_high "$high"
        done
      done
    done
  done
done

echo "[$(date +%FT%T)] latest pending probe done: scenes=$SCENES"
