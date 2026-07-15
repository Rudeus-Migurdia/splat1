#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SCENE=${SCENE:?Set SCENE}
RUN_TAG=${RUN_TAG:?Set the uncertainty-capacity run tag}
GPU_ID=${GPU_ID:-0}
SAMPLES_PER_VIEW=${SAMPLES_PER_VIEW:-256}

cd "$ROOT"
source scripts/drsplat_env.sh
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
cache_root="$ROOT/runs/split_consistency_heldout/$SCENE"
codebook="$ROOT/runs/uncertainty_capacity/$SCENE/${RUN_TAG}_shared_k16384"
output_root="$ROOT/runs/uncertainty_capacity/$SCENE/heldout"
log_root="$ROOT/logs/uncertainty_capacity"
mkdir -p "$output_root" "$log_root"

for source in old l2; do
  cache="$cache_root/cache_$source"
  output="$output_root/${RUN_TAG}_${source}.json"
  [[ -f "$cache/manifest.json" ]] || { echo "Missing cache: $cache" >&2; exit 1; }
  [[ -f "$codebook/manifest.json" ]] || { echo "Missing codebook: $codebook" >&2; exit 1; }
  if [[ ! -f "$output" ]]; then
    "$PYTHON_BIN" -u eval_semantic_field_consistency.py \
      --cache_dir "$cache" --codebook_dir "$codebook" --label_dir "$labels" \
      --samples_per_view "$SAMPLES_PER_VIEW" --lovo_topk 4 --seed 0 \
      --output "$output" > "$log_root/${SCENE}_${RUN_TAG}_${source}_heldout.log" 2>&1
  fi
done

echo "uncertainty-capacity held-out probe complete: scene=$SCENE tag=$RUN_TAG"
