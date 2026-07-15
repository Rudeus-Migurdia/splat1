#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.7264}
MIN_AGREEMENT=${MIN_AGREEMENT:-0.8}
MAX_NEIGHBOR_DISTANCE_RATIO=${MAX_NEIGHBOR_DISTANCE_RATIO:-1.5}
INPUT_ARTIFACT=${INPUT_ARTIFACT:-$ROOT/runs/codebook_objective_ablation/${SCENE}_direct_only_lr1e4_i1000/artifact}
LOG_DIR=${LOG_DIR:-$ROOT/logs/consensus_codebook_coverage}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
coverage_tag=${TARGET_COVERAGE/./p}
agreement_tag=${MIN_AGREEMENT/./p}
ratio_tag=${MAX_NEIGHBOR_DISTANCE_RATIO/./p}
run_root="$ROOT/runs/consensus_codebook_coverage/${SCENE}_target${coverage_tag}_a${agreement_tag}_r${ratio_tag}"
artifact="$run_root/artifact"
output="$run_root/eval_category_percentile_l1_h99"
mkdir -p "$LOG_DIR" "$run_root"

for path in "$INPUT_ARTIFACT/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_consensus_propagated_codebook.py \
    --input_dir "$INPUT_ARTIFACT" \
    --geometry_checkpoint "$geometry" \
    --output_dir "$artifact" \
    --target_coverage "$TARGET_COVERAGE" \
    --min_agreement "$MIN_AGREEMENT" \
    --max_neighbor_distance_ratio "$MAX_NEIGHBOR_DISTANCE_RATIO" \
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_a${agreement_tag}_r${ratio_tag}_build.log" 2>&1
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
    > "$LOG_DIR/${SCENE}_target${coverage_tag}_a${agreement_tag}_r${ratio_tag}_eval.log" 2>&1
fi

echo "consensus codebook coverage probe complete: scene=$SCENE target_coverage=$TARGET_COVERAGE"
