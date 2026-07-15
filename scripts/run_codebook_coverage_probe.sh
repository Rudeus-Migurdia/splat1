#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.7264}
INPUT_ARTIFACT=${INPUT_ARTIFACT:-$ROOT/runs/codebook_objective_ablation/${SCENE}_direct_only_lr1e4_i1000/artifact}
LOG_DIR=${LOG_DIR:-$ROOT/logs/codebook_coverage}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

if [[ "$SCENE" == "ramen" || "$SCENE" == "waldo_kitchen" ]]; then
  calibration=category_percentile
  calibration_low=1
  calibration_high=99
else
  calibration=frame_minmax
  calibration_low=0
  calibration_high=100
fi

coverage_tag=${TARGET_COVERAGE/./p}
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
run_root="$ROOT/runs/codebook_coverage/${SCENE}_target${coverage_tag}"
artifact="$run_root/artifact"
eval_dir="$run_root/eval_category_percentile_l1_h99"

for path in "$INPUT_ARTIFACT/manifest.json" "$geometry" "$dataset" "$labels"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$run_root"

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u propagate_gaussian_codebook_coverage.py \
    --input_dir "$INPUT_ARTIFACT" \
    --geometry_checkpoint "$geometry" \
    --output_dir "$artifact" \
    --target_coverage "$TARGET_COVERAGE" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_build.log" 2>&1
fi

if [[ ! -f "$eval_dir/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$artifact" \
    --label_dir "$labels" \
    --rgr_alpha 0 \
    --score_calibration "$calibration" \
    --calibration_low "$calibration_low" \
    --calibration_high "$calibration_high" \
    --thresholds $THRESHOLDS \
    --output "$eval_dir" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_eval.log" 2>&1
fi

echo "codebook coverage probe complete: scene=$SCENE target_coverage=$TARGET_COVERAGE"
