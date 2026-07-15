#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
SCENE=${SCENE:-waldo_kitchen}
FUSION_WEIGHTS=${FUSION_WEIGHTS:-"0.10 0.25 0.50 0.75 1.00"}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}
LOG_DIR=${LOG_DIR:-$ROOT/logs/multiscale_codebook_fusion}

cd "$ROOT"
source scripts/drsplat_env.sh
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}

dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
base_root="$ROOT/runs/baseline_voting_consensus/${SCENE}_topk45_fullraw"
fine_root="$ROOT/runs/baseline_voting_consensus/${SCENE}_topk45_fullraw_multiscale_l2"
base_codebook="$base_root/initial_codebook"
fine_codebook="$fine_root/initial_codebook"
run_root="$ROOT/runs/multiscale_codebook_fusion/$SCENE"

for path in "$dataset" "$labels" "$geometry" "$base_codebook/manifest.json" "$fine_codebook/manifest.json"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$run_root" "$LOG_DIR"

for weight in $FUSION_WEIGHTS; do
  tag=${weight/./p}
  output="$run_root/base_plus_l2_w${tag}_category_percentile_l1_h99"
  if [[ -f "$output/metrics.json" ]]; then
    continue
  fi
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$base_codebook" \
    --object_codebook_dir "$fine_codebook" \
    --object_feature_weight "$weight" \
    --label_dir "$labels" --rgr_alpha 0 \
    --score_calibration category_percentile --calibration_low 1 --calibration_high 99 \
    --thresholds $THRESHOLDS --output "$output" \
    > "$LOG_DIR/${SCENE}_base_plus_l2_w${tag}.log" 2>&1
done

echo "multiscale codebook fusion complete: weights=$FUSION_WEIGHTS"
