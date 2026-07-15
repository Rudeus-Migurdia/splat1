#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.7264}
INPUT_ARTIFACT=${INPUT_ARTIFACT:-$ROOT/runs/codebook_objective_ablation/${SCENE}_direct_only_lr1e4_i1000/artifact}
GROUP_HIERARCHY=${GROUP_HIERARCHY:-$ROOT/runs/group_view_importance/waldo_kitchen_information_kl_kl0p02_top1/group_hierarchy}
LOG_DIR=${LOG_DIR:-$ROOT/logs/multiview_sam_coverage}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
coverage_tag=${TARGET_COVERAGE/./p}
run_root="$ROOT/runs/multiview_sam_coverage/${SCENE}_target${coverage_tag}"
source_mask="$run_root/multiview_sam_sources.npy"
artifact="$run_root/artifact"
output="$run_root/eval_category_percentile_l1_h99"
mkdir -p "$LOG_DIR" "$run_root"

for path in "$INPUT_ARTIFACT/manifest.json" "$GROUP_HIERARCHY/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$source_mask" ]]; then
  "$PYTHON_BIN" -u build_multiview_sam_source_mask.py \
    --codebook_dir "$INPUT_ARTIFACT" \
    --group_hierarchy_dir "$GROUP_HIERARCHY" \
    --output_path "$source_mask" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_sources.log" 2>&1
fi

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u propagate_gaussian_codebook_coverage.py \
    --input_dir "$INPUT_ARTIFACT" \
    --geometry_checkpoint "$geometry" \
    --output_dir "$artifact" \
    --target_coverage "$TARGET_COVERAGE" \
    --source_mask "$source_mask" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_build.log" 2>&1
fi

if [[ ! -f "$output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$artifact" \
    --label_dir "$labels" \
    --rgr_alpha 0 \
    --score_calibration category_percentile \
    --calibration_low 1 --calibration_high 99 \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_eval.log" 2>&1
fi

echo "multiview SAM coverage probe complete: scene=$SCENE target_coverage=$TARGET_COVERAGE"
