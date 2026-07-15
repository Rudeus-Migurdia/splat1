#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-waldo_kitchen}
GPU_ID=${GPU_ID:-0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/calibration_sweep/$SCENE}

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

DATASET=${DATASET:-$ROOT/drsplat_data/lerf_ovs/$SCENE}
LABEL_DIR=${LABEL_DIR:-$ROOT/drsplat_data/lerf_ovs/label/$SCENE}
OUT=${OUT:-$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256}
CHECKPOINT=${CHECKPOINT:-$OUT/chkpnt0.pth}
ASSIGNMENTS=${ASSIGNMENTS:-$OUT/point_group_assignments.npz}
GROUP_FEATURES=${GROUP_FEATURES:-$OUT/teacher_distilled/group_features_teacher_w0p75.npy}
CODEBOOK_GROUP_FEATURES=${CODEBOOK_GROUP_FEATURES:-$OUT/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy}
REVERSE_DIR=${REVERSE_DIR:-$OUT/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage}
HIER_DIR=${HIER_DIR:-$OUT/teacher_distilled/hier_k32_f8_from_codebook}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 frame_percentile:1:99 frame_percentile:2:98 frame_percentile:5:95 category_percentile:1:99 category_percentile:2:98 category_percentile:5:95"}
TARGETS=${TARGETS:-"reverse_best coarse_best"}

mkdir -p "$LOG_DIR"

run_eval() {
  local tag=$1
  shift
  local output=$1
  shift
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse $tag"
    return
  fi
  echo "[$(date +%FT%T)] eval $tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$DATASET" -m "$OUT" \
    --checkpoint "$CHECKPOINT" \
    --label_dir "$LABEL_DIR" \
    --assignments "$ASSIGNMENTS" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    "$@" \
    > "$LOG_DIR/${tag}.log" 2>&1
}

for calibration in $CALIBRATIONS; do
  IFS=: read -r mode low high <<< "$calibration"
  cal_tag="${mode}_l${low}_h${high}"

  for target in $TARGETS; do
    case "$target" in
      reverse_best)
        run_eval "reverse_best_${cal_tag}" \
          "$OUT/eval/lerf_ovs_teacher_codebook_k256_reverse_query_softmax_top0_res0p5_codeblend0p0_cal_${cal_tag}" \
          --group_features "$GROUP_FEATURES" \
          --reverse_codebook_dir "$REVERSE_DIR" \
          --reverse_top_codes 0 \
          --reverse_residual_weight 0.5 \
          --reverse_code_blend 0.0 \
          --aggregation query_softmax \
          --query_temperature 0.07 \
          --query_prior_power 1.0 \
          --score_calibration "$mode" \
          --calibration_low "$low" \
          --calibration_high "$high"
        ;;
      reverse_second)
        run_eval "reverse_second_${cal_tag}" \
          "$OUT/eval/lerf_ovs_teacher_codebook_k256_reverse_query_softmax_top0_res1p0_codeblend0p25_cal_${cal_tag}" \
          --group_features "$GROUP_FEATURES" \
          --reverse_codebook_dir "$REVERSE_DIR" \
          --reverse_top_codes 0 \
          --reverse_residual_weight 1.0 \
          --reverse_code_blend 0.25 \
          --aggregation query_softmax \
          --query_temperature 0.07 \
          --query_prior_power 1.0 \
          --score_calibration "$mode" \
          --calibration_low "$low" \
          --calibration_high "$high"
        ;;
      coarse_best)
        run_eval "coarse_best_${cal_tag}" \
          "$OUT/eval/lerf_ovs_teacher_codebook_k256_weighted_coarse32_query_adaptive_blend_0p15_cal_${cal_tag}" \
          --group_features "$CODEBOOK_GROUP_FEATURES" \
          --coarse_features "$HIER_DIR/coarse_codebook.npy" \
          --group_to_coarse "$HIER_DIR/coarse_ids.npy" \
          --coarse_blend 0.15 \
          --coarse_blend_mode query_adaptive \
          --coarse_min_blend 0.05 \
          --coarse_specificity_topk 16 \
          --aggregation weighted \
          --score_calibration "$mode" \
          --calibration_low "$low" \
          --calibration_high "$high"
        ;;
      *)
        echo "Unknown target: $target" >&2
        exit 1
        ;;
    esac
  done
done

echo "[$(date +%FT%T)] calibration sweep done for $SCENE"
