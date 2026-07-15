#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
ITERATIONS=${ITERATIONS:-500}
CODEBOOK_LR=${CODEBOOK_LR:-1e-5}
SEGMENT_CONTRASTIVE_WEIGHT=${SEGMENT_CONTRASTIVE_WEIGHT:-0.01}
SEGMENT_CONTRASTIVE_TEMPERATURE=${SEGMENT_CONTRASTIVE_TEMPERATURE:-0.07}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.95}
INPUT_ARTIFACT=${INPUT_ARTIFACT:-$ROOT/runs/codebook_objective_ablation/${SCENE}_direct_only_lr1e4_i1000/artifact}
CACHE_DIR=${CACHE_DIR:-$ROOT/runs/self_trained_gaussian_codebook/waldo_kitchen_d512_k4096x4096_p32768/cache}
LOG_DIR=${LOG_DIR:-$ROOT/logs/segment_contrastive_codebook}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
weight_tag=${SEGMENT_CONTRASTIVE_WEIGHT/./p}
temperature_tag=${SEGMENT_CONTRASTIVE_TEMPERATURE/./p}
coverage_tag=${TARGET_COVERAGE/./p}
run_root="$ROOT/runs/segment_contrastive_codebook/${SCENE}_i${ITERATIONS}_lr${CODEBOOK_LR/./p}_w${weight_tag}_t${temperature_tag}"
trained="$run_root/trained"
coverage_artifact="$run_root/coverage${coverage_tag}/artifact"
output="$run_root/coverage${coverage_tag}/eval_category_percentile_l1_h99"
mkdir -p "$LOG_DIR" "$run_root"

for path in "$INPUT_ARTIFACT/manifest.json" "$CACHE_DIR/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$trained/artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u train_gaussian_multilevel_codebook.py \
    --cache_dir "$CACHE_DIR" \
    --initial_codebook_dir "$INPUT_ARTIFACT" \
    --output "$trained" \
    --iterations "$ITERATIONS" \
    --batch_pixels 4096 \
    --codebook_lr "$CODEBOOK_LR" \
    --direct_weight 1.0 \
    --lovo_weight 0.0 \
    --nuisance_rank 0 \
    --segment_contrastive_weight "$SEGMENT_CONTRASTIVE_WEIGHT" \
    --segment_contrastive_temperature "$SEGMENT_CONTRASTIVE_TEMPERATURE" \
    > "$LOG_DIR/${SCENE}_i${ITERATIONS}_w${weight_tag}_t${temperature_tag}_train.log" 2>&1
fi

if [[ ! -f "$coverage_artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u propagate_gaussian_codebook_coverage.py \
    --input_dir "$trained/artifact" \
    --geometry_checkpoint "$geometry" \
    --output_dir "$coverage_artifact" \
    --target_coverage "$TARGET_COVERAGE" \
    > "$LOG_DIR/${SCENE}_coverage${coverage_tag}_build.log" 2>&1
fi

if [[ ! -f "$output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$coverage_artifact" \
    --label_dir "$labels" \
    --rgr_alpha 0 \
    --score_calibration category_percentile \
    --calibration_low 1 --calibration_high 99 \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${SCENE}_coverage${coverage_tag}_eval.log" 2>&1
fi

echo "segment contrastive codebook probe complete: scene=$SCENE"
