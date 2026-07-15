#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
COARSE_CODES=${COARSE_CODES:-32}
FINE_CODES=${FINE_CODES:-8}
COARSE_BLENDS=${COARSE_BLENDS:-"0.15 0.30"}
QUERY_TEMPERATURES=${QUERY_TEMPERATURES:-"0.10 0.12"}
QUERY_PRIOR_POWERS=${QUERY_PRIOR_POWERS:-"0.75 1.0"}
CALIBRATIONS=${CALIBRATIONS:-"frame_minmax:0:100 category_percentile:1:99"}
REVERSE_TOP_CODES=${REVERSE_TOP_CODES:-0}
REVERSE_RESIDUAL_WEIGHT=${REVERSE_RESIDUAL_WEIGHT:-0.5}
REVERSE_CODE_BLEND=${REVERSE_CODE_BLEND:-0.0}
REVERSE_GROUP_PRIOR=${REVERSE_GROUP_PRIOR:-cosine}
REVERSE_PRIOR_POWER=${REVERSE_PRIOR_POWER:-1.0}
REVERSE_RESIDUAL_TEMPERATURE=${REVERSE_RESIDUAL_TEMPERATURE:-0.5}
COARSE_BLEND_MODE=${COARSE_BLEND_MODE:-fixed}
COARSE_MIN_BLEND=${COARSE_MIN_BLEND:-0.05}
COARSE_SPECIFICITY_TOPK=${COARSE_SPECIFICITY_TOPK:-16}
THRESHOLDS=${THRESHOLDS:-"0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/cosine_coarse_combo_probe}

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

ensure_reverse_index() {
  local scene=$1
  local out=$2
  local assignments="$out/point_group_assignments.npz"
  local group_features="$out/teacher_distilled/group_features_teacher_w0p75.npy"
  local codebook_dir="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage"
  local reverse_dir="$out/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage"
  local scene_log_dir="$LOG_DIR/$scene"

  mkdir -p "$reverse_dir" "$scene_log_dir"
  for path in "$assignments" "$group_features" "$codebook_dir/codebook.npy" "$codebook_dir/group_to_code.npy"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing reverse-index input for $scene: $path" >&2
      return 1
    fi
  done
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
    > "$scene_log_dir/00_build_reverse_index.log" 2>&1
}

ensure_hierarchical_tokens() {
  local scene=$1
  local out=$2
  local assignments="$out/point_group_assignments.npz"
  local codebook_features="$out/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy"
  local hier_dir="$out/teacher_distilled/hier_k${COARSE_CODES}_f${FINE_CODES}_from_codebook"
  local scene_log_dir="$LOG_DIR/$scene"

  mkdir -p "$hier_dir" "$scene_log_dir"
  for path in "$assignments" "$codebook_features"; do
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
    --group_features "$codebook_features" \
    --assignments "$assignments" \
    --coarse_codes "$COARSE_CODES" \
    --fine_codes "$FINE_CODES" \
    --iterations 120 \
    --seed 71 \
    --usage_weighted \
    --output_dir "$hier_dir" \
    > "$scene_log_dir/01_hierarchical_codebook.log" 2>&1
}

run_combo_eval() {
  local scene=$1
  local coarse_blend=$2
  local query_temperature=$3
  local query_prior_power=$4
  local mode=$5
  local low=$6
  local high=$7
  local cal_tag=$8

  local out
  out=$(scene_method_out "$scene")
  local dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  local label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  local checkpoint="$out/chkpnt0.pth"
  local assignments="$out/point_group_assignments.npz"
  local group_features="$out/teacher_distilled/group_features_teacher_w0p75.npy"
  local reverse_dir="$out/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage"
  local hier_dir="$out/teacher_distilled/hier_k${COARSE_CODES}_f${FINE_CODES}_from_codebook"
  local prefix
  prefix=$(eval_prefix "$scene")
  local scene_log_dir="$LOG_DIR/$scene"
  mkdir -p "$scene_log_dir"

  for path in "$checkpoint" "$assignments" "$group_features" "$label_dir" "$reverse_dir/reverse_codebook_summary.json" "$hier_dir/coarse_codebook.npy" "$hier_dir/coarse_ids.npy"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing combo eval input for $scene: $path" >&2
      return 1
    fi
  done

  local blend_tag qtemp_tag qprior_tag
  blend_tag=$(tag_float "$coarse_blend")
  qtemp_tag=$(tag_float "$query_temperature")
  qprior_tag=$(tag_float "$query_prior_power")
  local name="${prefix}_combo_cosine_coarse${COARSE_CODES}_${COARSE_BLEND_MODE}_blend_${blend_tag}_qt${qtemp_tag}_qp${qprior_tag}_cal_${cal_tag}"
  local output="$out/eval/$name"
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(date +%FT%T)] reuse combo scene=$scene name=$name"
    return
  fi

  echo "[$(date +%FT%T)] eval combo scene=$scene blend=$coarse_blend qt=$query_temperature qp=$query_prior_power cal=$cal_tag"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$dataset" -m "$out" \
    --checkpoint "$checkpoint" \
    --label_dir "$label_dir" \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --reverse_codebook_dir "$reverse_dir" \
    --reverse_top_codes "$REVERSE_TOP_CODES" \
    --reverse_residual_weight "$REVERSE_RESIDUAL_WEIGHT" \
    --reverse_code_blend "$REVERSE_CODE_BLEND" \
    --reverse_group_prior "$REVERSE_GROUP_PRIOR" \
    --reverse_prior_power "$REVERSE_PRIOR_POWER" \
    --reverse_residual_temperature "$REVERSE_RESIDUAL_TEMPERATURE" \
    --coarse_features "$hier_dir/coarse_codebook.npy" \
    --group_to_coarse "$hier_dir/coarse_ids.npy" \
    --coarse_blend "$coarse_blend" \
    --coarse_blend_mode "$COARSE_BLEND_MODE" \
    --coarse_min_blend "$COARSE_MIN_BLEND" \
    --coarse_specificity_topk "$COARSE_SPECIFICITY_TOPK" \
    --aggregation query_softmax \
    --score_power 1.0 \
    --query_temperature "$query_temperature" \
    --query_prior_power "$query_prior_power" \
    --thresholds $THRESHOLDS \
    --score_calibration "$mode" \
    --calibration_low "$low" \
    --calibration_high "$high" \
    --output "$output" \
    > "$scene_log_dir/${name}.log" 2>&1
}

for scene in $SCENES; do
  out=$(scene_method_out "$scene")
  ensure_reverse_index "$scene" "$out"
  ensure_hierarchical_tokens "$scene" "$out"
  for calibration in $CALIBRATIONS; do
    IFS=: read -r mode low high <<< "$calibration"
    cal_tag="${mode}_l${low}_h${high}"
    for coarse_blend in $COARSE_BLENDS; do
      for query_temperature in $QUERY_TEMPERATURES; do
        for query_prior_power in $QUERY_PRIOR_POWERS; do
          run_combo_eval "$scene" "$coarse_blend" "$query_temperature" "$query_prior_power" "$mode" "$low" "$high" "$cal_tag"
        done
      done
    done
  done
done

echo "[$(date +%FT%T)] cosine coarse combo probe done: scenes=$SCENES"
