#!/usr/bin/env bash
set -euo pipefail

# PQ is used only while generating teacher_distilled/group_features*. The
# evaluator below reads the discrete artifact, integer attachments, and labels.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
COARSE_CODES=${COARSE_CODES:-32}
FINE_CODES=${FINE_CODES:-8}
TOP_M=${TOP_M:-2}
DYNAMIC_SOURCE=${DYNAMIC_SOURCE:-codebook_quantized}
DYNAMIC_FINE_TEMPERATURE=${DYNAMIC_FINE_TEMPERATURE:-0.10}
DYNAMIC_CANDIDATE_PRIOR_POWER=${DYNAMIC_CANDIDATE_PRIOR_POWER:-0.25}
DYNAMIC_FINE_MIN_BLEND=${DYNAMIC_FINE_MIN_BLEND:-0.10}
DYNAMIC_FINE_MAX_BLEND=${DYNAMIC_FINE_MAX_BLEND:-0.90}
DYNAMIC_REVERSE_TOP_CODES=${DYNAMIC_REVERSE_TOP_CODES:-"0 32"}
AGGREGATION=${AGGREGATION:-query_softmax}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100"}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/dynamic_hierarchical_codebook_probe}

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

source_tag=${DYNAMIC_SOURCE//[^A-Za-z0-9]/_}

temperature_tag=$(tag_float "$DYNAMIC_FINE_TEMPERATURE")
prior_tag=$(tag_float "$DYNAMIC_CANDIDATE_PRIOR_POWER")
min_blend_tag=$(tag_float "$DYNAMIC_FINE_MIN_BLEND")
max_blend_tag=$(tag_float "$DYNAMIC_FINE_MAX_BLEND")
query_temperature_tag=$(tag_float "$QUERY_TEMPERATURE")
query_prior_tag=$(tag_float "$QUERY_PRIOR_POWER")

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  assignments="$out/point_group_assignments.npz"
  teacher_features="$out/teacher_distilled/group_features_teacher_w0p75.npy"
  quantized_features="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy"
  case "$DYNAMIC_SOURCE" in
    teacher)
      source_features="$teacher_features"
      ;;
    codebook_quantized)
      source_features="$quantized_features"
      ;;
    *)
      echo "Unknown DYNAMIC_SOURCE=$DYNAMIC_SOURCE (expected teacher or codebook_quantized)" >&2
      exit 2
      ;;
  esac
  checkpoint="$out/chkpnt0.pth"
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  artifact="$out/teacher_distilled/dynamic_hier_${source_tag}_c${COARSE_CODES}_f${FINE_CODES}_m${TOP_M}"

  for path in "$assignments" "$source_features" "$checkpoint" "$dataset" "$label_dir"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  if [[ ! -f "$artifact/dynamic_hierarchical_codebook_summary.json" ]]; then
    "$PYTHON_BIN" dynamic_hierarchical_codebook.py \
      --group_features "$source_features" \
      --assignments "$assignments" \
      --coarse_codes "$COARSE_CODES" \
      --fine_codes "$FINE_CODES" \
      --top_m "$TOP_M" \
      --iterations 120 \
      --seed 71 \
      --usage_weighted \
      --output_dir "$artifact" \
      > "$LOG_DIR/${scene}_00_build_dynamic_hierarchical.log" 2>&1
  fi

  for reverse_top_codes in $DYNAMIC_REVERSE_TOP_CODES; do
    for calibration in $CALIBRATIONS; do
      IFS=: read -r mode low high <<< "$calibration"
      cal_tag="${mode}_l${low}_h${high}"
      output="$out/eval/lerf_ovs_dynamic_hier_${source_tag}_c${COARSE_CODES}_f${FINE_CODES}_m${TOP_M}_t${temperature_tag}_p${prior_tag}_b${min_blend_tag}-${max_blend_tag}_${AGGREGATION}_qt${query_temperature_tag}_qp${query_prior_tag}_top${reverse_top_codes}_cal_${cal_tag}"
      if [[ -f "$output/metrics.json" ]]; then
        echo "reuse scene=$scene output=$output"
        continue
      fi
      "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
        -s "$dataset" -m "$out" \
        --checkpoint "$checkpoint" \
        --label_dir "$label_dir" \
        --assignments "$assignments" \
        --dynamic_hierarchical_codebook_dir "$artifact" \
        --dynamic_fine_temperature "$DYNAMIC_FINE_TEMPERATURE" \
        --dynamic_candidate_prior_power "$DYNAMIC_CANDIDATE_PRIOR_POWER" \
        --dynamic_fine_min_blend "$DYNAMIC_FINE_MIN_BLEND" \
        --dynamic_fine_max_blend "$DYNAMIC_FINE_MAX_BLEND" \
        --dynamic_reverse_top_codes "$reverse_top_codes" \
        --aggregation "$AGGREGATION" \
        --query_temperature "$QUERY_TEMPERATURE" \
        --query_prior_power "$QUERY_PRIOR_POWER" \
        --thresholds $THRESHOLDS \
        --score_calibration "$mode" \
        --calibration_low "$low" \
        --calibration_high "$high" \
        --output "$output" \
        > "$LOG_DIR/${scene}_dynamic_top${reverse_top_codes}_${cal_tag}.log" 2>&1
    done
  done
done

echo "dynamic hierarchical codebook probe complete: scenes=$SCENES"
