#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
TOPK=${TOPK:-45}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-language_features}
FEATURE_LEVEL=${FEATURE_LEVEL:-1}
RUN_SUFFIX=${RUN_SUFFIX:-}
CONSENSUS_CHUNK_PIXELS=${CONSENSUS_CHUNK_PIXELS:-1024}
CONSENSUS_SPLITS=${CONSENSUS_SPLITS:-1}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.95}
LOG_DIR=${LOG_DIR:-$ROOT/logs/baseline_voting_consensus}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
feature_dir="$dataset/$FEATURE_DIR_NAME"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
coverage_tag=${TARGET_COVERAGE/./p}
run_root="$ROOT/runs/baseline_voting_consensus/${SCENE}_topk${TOPK}_fullraw${RUN_SUFFIX}"
run_tag="${SCENE}${RUN_SUFFIX}"
cache="$run_root/consensus"
artifact="$run_root/initial_codebook"
coverage_artifact="$run_root/coverage${coverage_tag}/artifact"
initial_output="$run_root/eval_initial_category_percentile_l1_h99"
coverage_output="$run_root/eval_coverage${coverage_tag}_category_percentile_l1_h99"
mkdir -p "$LOG_DIR" "$run_root"

for path in "$dataset" "$labels" "$geometry" "$feature_dir"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$cache/manifest.json" ]]; then
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$cache" \
    --geometry_checkpoint "$geometry" \
    --feature_dir "$feature_dir" \
    --feature_level "$FEATURE_LEVEL" \
    --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk "$TOPK" --raw_contribution_weights \
    --consensus_only --consensus_chunk_pixels "$CONSENSUS_CHUNK_PIXELS" \
    --consensus_splits "$CONSENSUS_SPLITS" \
    > "$LOG_DIR/${run_tag}_fullraw_prepare.log" 2>&1
fi

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
    --consensus "$cache/consensus.pt" \
    --codes_per_level 4096 4096 \
    --train_samples 262144 --iterations 25 --assignment_chunk 16384 --faiss_gpu \
    --output_dir "$artifact" \
    > "$LOG_DIR/${run_tag}_fullraw_codebook.log" 2>&1
fi

if [[ ! -f "$initial_output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" --geometry_checkpoint "$geometry" \
    --codebook_dir "$artifact" --label_dir "$labels" --rgr_alpha 0 \
    --score_calibration category_percentile --calibration_low 1 --calibration_high 99 \
    --thresholds $THRESHOLDS --output "$initial_output" \
    > "$LOG_DIR/${run_tag}_fullraw_initial_eval.log" 2>&1
fi

if [[ ! -f "$coverage_artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u propagate_gaussian_codebook_coverage.py \
    --input_dir "$artifact" --geometry_checkpoint "$geometry" \
    --output_dir "$coverage_artifact" --target_coverage "$TARGET_COVERAGE" \
    > "$LOG_DIR/${run_tag}_fullraw_coverage${coverage_tag}.log" 2>&1
fi

if [[ ! -f "$coverage_output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" --geometry_checkpoint "$geometry" \
    --codebook_dir "$coverage_artifact" --label_dir "$labels" --rgr_alpha 0 \
    --score_calibration category_percentile --calibration_low 1 --calibration_high 99 \
    --thresholds $THRESHOLDS --output "$coverage_output" \
    > "$LOG_DIR/${run_tag}_fullraw_coverage${coverage_tag}_eval.log" 2>&1
fi

echo "baseline voting consensus probe complete: scene=$SCENE"
