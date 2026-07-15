#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
RUN_CONTINUOUS=${RUN_CONTINUOUS:-1}
LOG_DIR=${LOG_DIR:-$ROOT/logs/unified_codebook_equivalence}
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

base_root="$ROOT/runs/self_trained_gaussian_codebook/${SCENE}_d512_k4096x4096_p32768"
source_artifact="$base_root/lovo_kl/artifact"
consensus="$base_root/cache/consensus.pt"
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
run_root="$ROOT/runs/unified_codebook_equivalence/${SCENE}_from_k4096x4096"
artifact="$run_root/artifact"
merged_eval="$run_root/eval_unit_sum_${calibration}_l${calibration_low}_h${calibration_high}"
continuous_eval="$run_root/eval_consensus_upper_bound_${calibration}_l${calibration_low}_h${calibration_high}"

for path in "$source_artifact/manifest.json" "$consensus" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$run_root"

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u merge_multilevel_codebook_to_shared.py \
    --input_dir "$source_artifact" \
    --output_dir "$artifact" \
    > "$LOG_DIR/${SCENE}_merge.log" 2>&1
fi

if [[ ! -f "$merged_eval/metrics.json" ]]; then
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
    --output "$merged_eval" \
    > "$LOG_DIR/${SCENE}_unit_sum_eval.log" 2>&1
fi

if [[ "$RUN_CONTINUOUS" == "1" && ! -f "$continuous_eval/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --consensus_path "$consensus" \
    --label_dir "$labels" \
    --rgr_alpha 0 \
    --score_calibration "$calibration" \
    --calibration_low "$calibration_low" \
    --calibration_high "$calibration_high" \
    --thresholds $THRESHOLDS \
    --output "$continuous_eval" \
    > "$LOG_DIR/${SCENE}_consensus_eval.log" 2>&1
fi

echo "unified codebook equivalence probe complete: scene=$SCENE"
