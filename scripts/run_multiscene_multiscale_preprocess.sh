#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
SCENES=${SCENES:-"ramen teatime figurines"}
NUM_SHARDS=${NUM_SHARDS:-2}
SHARD_INDEX=${SHARD_INDEX:?Set SHARD_INDEX to a value in [0, NUM_SHARDS)}
OUTPUT_FEATURE_DIR=${OUTPUT_FEATURE_DIR:-language_features_multiscale}
SAM_CKPT=${SAM_CKPT:-$ROOT/ckpts/sam_vit_h_4b8939.pth}

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "SHARD_INDEX must be in [0, NUM_SHARDS)" >&2
  exit 2
fi

cd "$ROOT"
source scripts/drsplat_env.sh
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

for scene in $SCENES; do
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  [[ -d "$dataset/images" ]] || { echo "Missing scene images: $dataset/images" >&2; exit 1; }
  mkdir -p "$dataset/$OUTPUT_FEATURE_DIR"
  "$PYTHON_BIN" -u preprocessing.py \
    --dataset_path "$dataset" \
    --output_feature_dir "$OUTPUT_FEATURE_DIR" \
    --sam_ckpt_path "$SAM_CKPT" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --only_missing --fail_fast --fast_mask_nms
done

echo "multiscene multiscale preprocessing complete: shard=$SHARD_INDEX/$NUM_SHARDS scenes=$SCENES"
