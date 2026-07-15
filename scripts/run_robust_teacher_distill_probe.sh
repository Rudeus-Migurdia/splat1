#!/usr/bin/env bash
set -euo pipefail

# The PQ checkpoint is consumed only by the optional offline build stage.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
TEACHER_WEIGHT=${TEACHER_WEIGHT:-0.75}
ROBUST_ITERATIONS=${ROBUST_ITERATIONS:-2}
ROBUST_AGREEMENT_POWER=${ROBUST_AGREEMENT_POWER:-2.0}
ROBUST_MARGIN_FLOOR=${ROBUST_MARGIN_FLOOR:-0.25}
MIN_TEACHER_BLEND=${MIN_TEACHER_BLEND:-0.25}
ADAPTIVE_TEACHER_BLEND=${ADAPTIVE_TEACHER_BLEND:-1}
AGGREGATION=${AGGREGATION:-query_softmax}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 category_percentile:1:99"}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
BUILD_ONLY=${BUILD_ONLY:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/robust_teacher_distill_probe}

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

tag_float() {
  printf "%s\n" "${1/./p}"
}

weight_tag=$(tag_float "$TEACHER_WEIGHT")
agreement_tag=$(tag_float "$ROBUST_AGREEMENT_POWER")
margin_tag=$(tag_float "$ROBUST_MARGIN_FLOOR")
min_blend_tag=$(tag_float "$MIN_TEACHER_BLEND")
if [[ "$ADAPTIVE_TEACHER_BLEND" == "1" ]]; then
  blend_mode=adaptive
  adaptive_arg=(--adaptive_teacher_blend --min_teacher_blend "$MIN_TEACHER_BLEND")
else
  blend_mode=fixed
  adaptive_arg=()
fi

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  checkpoint="$out/chkpnt0.pth"
  pq_checkpoint="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
  assignments="$out/point_group_assignments.npz"
  distill_dir="$out/teacher_distilled/robust_irls_i${ROBUST_ITERATIONS}_a${agreement_tag}_m${margin_tag}_${blend_mode}_min${min_blend_tag}"
  robust_features="$distill_dir/group_features_robust_w${weight_tag}.npy"
  original_features="$out/teacher_distilled/group_features_teacher_w${weight_tag}.npy"

  for path in "$dataset" "$label_dir" "$checkpoint" "$pq_checkpoint" "$assignments" "$original_features"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  if [[ "$EVAL_ONLY" != "1" && ! -f "$robust_features" ]]; then
    "$PYTHON_BIN" distill_group_tokens_from_drsplat.py \
      --artifact_dir "$out" \
      --drsplat_checkpoint "$pq_checkpoint" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --score_power 1.0 \
      --teacher_weights "$TEACHER_WEIGHT" \
      --robust_iterations "$ROBUST_ITERATIONS" \
      --robust_agreement_power "$ROBUST_AGREEMENT_POWER" \
      --robust_margin_floor "$ROBUST_MARGIN_FLOOR" \
      "${adaptive_arg[@]}" \
      --output_dir "$distill_dir" \
      > "$LOG_DIR/${scene}_00_robust_distill.log" 2>&1
  fi
  [[ -f "$robust_features" ]] || { echo "Missing robust features for $scene: $robust_features" >&2; exit 1; }
  if [[ "$BUILD_ONLY" == "1" ]]; then
    continue
  fi

  for feature_mode in teacher robust; do
    if [[ "$feature_mode" == "teacher" ]]; then
      features="$original_features"
    else
      features="$robust_features"
    fi
    for calibration in $CALIBRATIONS; do
      IFS=: read -r mode low high <<< "$calibration"
      cal_tag="${mode}_l${low}_h${high}"
      output="$out/eval/lerf_ovs_${feature_mode}_w${weight_tag}_${AGGREGATION}_robustirls_i${ROBUST_ITERATIONS}_a${agreement_tag}_m${margin_tag}_${blend_mode}_min${min_blend_tag}_cal_${cal_tag}"
      if [[ -f "$output/metrics.json" ]]; then
        echo "reuse scene=$scene feature=$feature_mode output=$output"
        continue
      fi
      "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
        -s "$dataset" -m "$out" \
        --checkpoint "$checkpoint" \
        --label_dir "$label_dir" \
        --group_features "$features" \
        --assignments "$assignments" \
        --aggregation "$AGGREGATION" \
        --query_temperature "$QUERY_TEMPERATURE" \
        --query_prior_power "$QUERY_PRIOR_POWER" \
        --thresholds $THRESHOLDS \
        --score_calibration "$mode" \
        --calibration_low "$low" \
        --calibration_high "$high" \
        --output "$output" \
        > "$LOG_DIR/${scene}_${feature_mode}_${cal_tag}.log" 2>&1
    done
  done
done

echo "robust teacher distillation probe complete: scenes=$SCENES"
