#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
ITERATIONS=${ITERATIONS:-500}
CODEBOOK_LR=${CODEBOOK_LR:-1e-5}
QUERY_KL_WEIGHT=${QUERY_KL_WEIGHT:-0.02}
QUERY_CONFIDENCE_POWER=${QUERY_CONFIDENCE_POWER:-2.0}
TARGET_COVERAGE=${TARGET_COVERAGE:-0.95}
INPUT_ARTIFACT=${INPUT_ARTIFACT:-$ROOT/runs/codebook_objective_ablation/${SCENE}_direct_only_lr1e4_i1000/artifact}
CACHE_DIR=${CACHE_DIR:-$ROOT/runs/self_trained_gaussian_codebook/waldo_kitchen_d512_k4096x4096_p32768/cache}
LOG_DIR=${LOG_DIR:-$ROOT/logs/text_rank_codebook}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
weight_tag=${QUERY_KL_WEIGHT/./p}
power_tag=${QUERY_CONFIDENCE_POWER/./p}
coverage_tag=${TARGET_COVERAGE/./p}
run_root="$ROOT/runs/text_rank_codebook/${SCENE}_i${ITERATIONS}_lr${CODEBOOK_LR/./p}_q${weight_tag}_p${power_tag}"
text_bank="$ROOT/runs/text_rank_codebook/generic_text_anchors.npy"
trained="$run_root/trained"
coverage_artifact="$run_root/coverage${coverage_tag}/artifact"
output="$run_root/coverage${coverage_tag}/eval_category_percentile_l1_h99"
mkdir -p "$LOG_DIR" "$run_root"

for path in "$INPUT_ARTIFACT/manifest.json" "$CACHE_DIR/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$text_bank" ]]; then
  "$PYTHON_BIN" -u build_text_query_bank.py --output "$text_bank" \
    > "$LOG_DIR/generic_text_bank.log" 2>&1
fi

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
    --query_bank "$text_bank" \
    --query_kl_weight "$QUERY_KL_WEIGHT" \
    --query_confidence_power "$QUERY_CONFIDENCE_POWER" \
    > "$LOG_DIR/${SCENE}_i${ITERATIONS}_q${weight_tag}_p${power_tag}_train.log" 2>&1
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

echo "text rank codebook probe complete: scene=$SCENE"
