#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
HYBRID_ALPHA_MAX=${HYBRID_ALPHA_MAX:-0.75}
HYBRID_ALPHA_MIN=${HYBRID_ALPHA_MIN:-0.0}
HYBRID_BLEND_MODE=${HYBRID_BLEND_MODE:-positive_residual}
GATE_TEMPERATURE=${GATE_TEMPERATURE:-0.1}
GATE_BIAS=${GATE_BIAS:-0.0}
GATE_MODE=${GATE_MODE:-fixed}
POINT_GATE_MODE=${POINT_GATE_MODE:-assignment_margin}
POINT_GATE_FLOOR=${POINT_GATE_FLOOR:-0.1}
POINT_GATE_POWER=${POINT_GATE_POWER:-1.0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/hybrid_reliability_gate_probe}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

scene_out_dir() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
  else
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256"
  fi
}

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

tag_float() {
  printf "%s\n" "${1/./p}"
}

alpha_tag=$(tag_float "$HYBRID_ALPHA_MAX")
min_tag=$(tag_float "$HYBRID_ALPHA_MIN")
temp_tag=$(tag_float "$GATE_TEMPERATURE")
bias_tag=$(tag_float "$GATE_BIAS")
floor_tag=$(tag_float "$POINT_GATE_FLOOR")
power_tag=$(tag_float "$POINT_GATE_POWER")

for scene in $SCENES; do
  out=$(scene_out_dir "$scene")
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  label_dir="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  drs_ckpt="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
  assignments="$out/point_group_assignments.npz"
  group_features="$out/group_features.npy"
  IFS=: read -r cal_mode cal_low cal_high <<< "$(scene_calibration "$scene")"
  cal_tag="${cal_mode}_l${cal_low}_h${cal_high}"
  output="$out/eval/lerf_ovs_hybrid_rawgroup_${GATE_MODE}_blend${HYBRID_BLEND_MODE}_point${POINT_GATE_MODE}_f${floor_tag}_p${power_tag}_max${alpha_tag}_min${min_tag}_t${temp_tag}_b${bias_tag}_cal_${cal_tag}"

  for path in "$dataset" "$label_dir" "$drs_ckpt" "$assignments" "$group_features" "$ROOT/ckpts/pq_index.faiss"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  if [[ -f "$output/metrics.json" ]]; then
    echo "reuse scene=$scene output=$output"
    continue
  fi
  "$PYTHON_BIN" -u eval_lerf_ovs_hybrid_miou.py \
    -s "$dataset" -m "$out" \
    --drsplat_checkpoint "$drs_ckpt" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --label_dir "$label_dir" \
    --group_features "$group_features" \
    --assignments "$assignments" \
    --group_aggregation weighted \
    --score_power 1.0 \
    --hybrid_alpha "$HYBRID_ALPHA_MAX" \
    --hybrid_blend_mode "$HYBRID_BLEND_MODE" \
    --hybrid_alpha_min "$HYBRID_ALPHA_MIN" \
    --hybrid_gate_mode "$GATE_MODE" \
    --gate_temperature "$GATE_TEMPERATURE" \
    --gate_bias "$GATE_BIAS" \
    --point_group_gate_mode "$POINT_GATE_MODE" \
    --point_group_gate_floor "$POINT_GATE_FLOOR" \
    --point_group_gate_power "$POINT_GATE_POWER" \
    --score_calibration "$cal_mode" \
    --calibration_low "$cal_low" \
    --calibration_high "$cal_high" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${scene}_${GATE_MODE}_blend${HYBRID_BLEND_MODE}_point${POINT_GATE_MODE}_f${floor_tag}_p${power_tag}_max${alpha_tag}_min${min_tag}_t${temp_tag}_b${bias_tag}_${cal_tag}.log" 2>&1
done

echo "hybrid reliability gate probe complete: scenes=$SCENES"
