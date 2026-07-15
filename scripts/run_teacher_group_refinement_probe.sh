#!/usr/bin/env bash
set -euo pipefail

# Offline PQ decoding is limited to the build stage. Evaluation uses only the
# refined group features and integer Gaussian-to-group assignments.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
PARENT_DIR_TAG=${PARENT_DIR_TAG:-robust_irls_i2_a2p0_m0p25_fixed_min0p25}
TEACHER_WEIGHT=${TEACHER_WEIGHT:-0.75}
MIN_POINTS=${MIN_POINTS:-1024}
MIN_DISPERSION=${MIN_DISPERSION:-0.04}
MAX_SPLITS=${MAX_SPLITS:-32}
CHILD_TEACHER_WEIGHT=${CHILD_TEACHER_WEIGHT:-0.75}
AGGREGATION=${AGGREGATION:-query_softmax}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 category_percentile:1:99"}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
BUILD_ONLY=${BUILD_ONLY:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/teacher_group_refinement_probe}

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
dispersion_tag=$(tag_float "$MIN_DISPERSION")
child_weight_tag=$(tag_float "$CHILD_TEACHER_WEIGHT")

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  checkpoint="$out/chkpnt0.pth"
  pq_checkpoint="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
  parent_dir="$out/teacher_distilled/$PARENT_DIR_TAG"
  parent_features="$parent_dir/group_features_robust_w${weight_tag}.npy"
  refined_dir="$out/teacher_distilled/refined_split_p${MIN_POINTS}_d${dispersion_tag}_w${child_weight_tag}"
  refined_features="$refined_dir/group_features_refined.npy"
  refined_assignments="$refined_dir/point_group_assignments_refined.npz"

  for path in "$dataset" "$label_dir" "$checkpoint" "$pq_checkpoint" "$parent_features"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  if [[ "$EVAL_ONLY" != "1" && ! -f "$refined_features" ]]; then
    "$PYTHON_BIN" refine_teacher_groups.py \
      --artifact_dir "$out" \
      --parent_features "$parent_features" \
      --drsplat_checkpoint "$pq_checkpoint" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --min_points "$MIN_POINTS" \
      --min_dispersion "$MIN_DISPERSION" \
      --max_splits "$MAX_SPLITS" \
      --child_teacher_weight "$CHILD_TEACHER_WEIGHT" \
      --iterations 50 \
      --seed 101 \
      --output_dir "$refined_dir" \
      > "$LOG_DIR/${scene}_00_refine.log" 2>&1
  fi
  [[ -f "$refined_features" && -f "$refined_assignments" ]] || { echo "Missing refinement output for $scene" >&2; exit 1; }
  if [[ "$BUILD_ONLY" == "1" ]]; then
    continue
  fi
  for feature_mode in parent refined; do
    if [[ "$feature_mode" == "parent" ]]; then
      features="$parent_features"
      assignments="$out/point_group_assignments.npz"
    else
      features="$refined_features"
      assignments="$refined_assignments"
    fi
    for calibration in $CALIBRATIONS; do
      IFS=: read -r mode low high <<< "$calibration"
      cal_tag="${mode}_l${low}_h${high}"
      output="$out/eval/lerf_ovs_${feature_mode}_teacher_refined_p${MIN_POINTS}_d${dispersion_tag}_w${child_weight_tag}_${AGGREGATION}_cal_${cal_tag}"
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

echo "teacher group refinement probe complete: scenes=$SCENES"
