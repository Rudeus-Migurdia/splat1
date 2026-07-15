#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
INPUT_VARIANT=${INPUT_VARIANT:-true_l3}
VIEW_WEIGHTING=${VIEW_WEIGHTING:-information_kl}
TOP_M=${TOP_M:-1}
IMPORTANCE_TEMPERATURE=${IMPORTANCE_TEMPERATURE:-1.0}
MAX_VIEW_KL=${MAX_VIEW_KL:-0.1}
IMPORTANCE_RATIO_CLIP=${IMPORTANCE_RATIO_CLIP:-5.0}
RGR_ALPHA=${RGR_ALPHA:-1.0}
AGREEMENT_FLOOR=${AGREEMENT_FLOOR:-0.5}
RUN_BUILD=${RUN_BUILD:-1}
RUN_EVAL=${RUN_EVAL:-1}
LOG_DIR=${LOG_DIR:-$ROOT/logs/group_view_importance}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
mkdir -p "$LOG_DIR"

case "$INPUT_VARIANT" in
  true_l3)
    [[ "$SCENE" == "waldo_kitchen" ]] || {
      echo "true_l3 cache is currently available only for waldo_kitchen" >&2
      exit 1
    }
    base="$ROOT/runs/hierarchical_multiscale_codebook/waldo_kitchen_fine4096x4096_obj1024_p32768_topk45"
    cache="$base/object_cache_l3"
    fixed_codebook="$base/recovery/l3_base/artifact"
    ;;
  stage_b)
    base="$ROOT/runs/self_trained_gaussian_codebook/${SCENE}_d512_k4096x4096_p32768"
    cache="$base/cache"
    fixed_codebook="$base/lovo_kl/artifact"
    ;;
  *)
    echo "Unknown INPUT_VARIANT=$INPUT_VARIANT" >&2
    exit 2
    ;;
esac
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
kl_tag=${MAX_VIEW_KL/./p}
run_root="$ROOT/runs/group_view_importance/${SCENE}_${INPUT_VARIANT}_${VIEW_WEIGHTING}_kl${kl_tag}_top${TOP_M}"
hierarchy="$run_root/group_hierarchy"
if [[ "$SCENE" == "ramen" || "$SCENE" == "waldo_kitchen" ]]; then
  calibration=category_percentile
  calibration_low=1
  calibration_high=99
else
  calibration=frame_minmax
  calibration_low=0
  calibration_high=100
fi
output="$run_root/eval_positive_a${RGR_ALPHA}_agree${AGREEMENT_FLOOR}_${calibration}_l${calibration_low}_h${calibration_high}"

for path in "$cache/manifest.json" "$fixed_codebook/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$run_root"

if [[ "$RUN_BUILD" == "1" && ! -f "$hierarchy/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_multiview_mask_track_hierarchy.py \
    --cache_dir "$cache" \
    --output_dir "$hierarchy" \
    --similarity_threshold 0.82 \
    --min_track_support 32 \
    --view_weighting "$VIEW_WEIGHTING" \
    --top_m "$TOP_M" \
    --importance_temperature "$IMPORTANCE_TEMPERATURE" \
    --max_view_kl "$MAX_VIEW_KL" \
    --importance_ratio_clip "$IMPORTANCE_RATIO_CLIP" \
    > "$LOG_DIR/${SCENE}_${INPUT_VARIANT}_${VIEW_WEIGHTING}_kl${kl_tag}_top${TOP_M}_build.log" 2>&1
fi

if [[ "$RUN_EVAL" == "1" && ! -f "$output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$fixed_codebook" \
    --label_dir "$labels" \
    --group_hierarchy_dir "$hierarchy" \
    --group_topk "$TOP_M" \
    --group_aggregation weighted \
    --rgr_alpha "$RGR_ALPHA" \
    --rgr_mode positive \
    --group_feature_agreement_floor "$AGREEMENT_FLOOR" \
    --score_calibration "$calibration" \
    --calibration_low "$calibration_low" --calibration_high "$calibration_high" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${SCENE}_${INPUT_VARIANT}_${VIEW_WEIGHTING}_kl${kl_tag}_top${TOP_M}_eval.log" 2>&1
fi

echo "group view importance complete: scene=$SCENE mode=$VIEW_WEIGHTING"
