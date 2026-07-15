#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
SCENE=${SCENE:?Set SCENE}
MAX_PIXELS_PER_VIEW=${MAX_PIXELS_PER_VIEW:-4096}
SAMPLES_PER_VIEW=${SAMPLES_PER_VIEW:-256}
LOG_DIR=${LOG_DIR:-$ROOT/logs/split_consistency_heldout}

cd "$ROOT"
source scripts/drsplat_env.sh
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
run_root="$ROOT/runs/split_consistency_heldout/$SCENE"
old_cache="$run_root/cache_old"
l2_cache="$run_root/cache_l2"
old_codebook="$run_root/codebook_old_k4096x2"

if [[ "$SCENE" == waldo_kitchen ]]; then
  base_consensus="$ROOT/runs/multiscale_split_consistency/waldo_base_split2/consensus.pt"
  current_codebook="$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2"
else
  base_consensus="$ROOT/runs/multiscale_split_consistency/$SCENE/base_split2/consensus.pt"
  current_codebook="$ROOT/runs/multiscale_split_consistency/$SCENE/fused_w1p5_t005_codebook_k4096x2"
fi

mkdir -p "$LOG_DIR" "$run_root"
for path in "$dataset" "$labels" "$geometry" "$base_consensus" \
  "$current_codebook/manifest.json" "$dataset/language_features" \
  "$dataset/language_features_multiscale"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

prepare_cache() {
  local output=$1
  local feature_dir=$2
  local feature_level=$3
  local tag=$4
  if [[ ! -f "$output/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$output" --geometry_checkpoint "$geometry" \
      --feature_dir "$feature_dir" --feature_level "$feature_level" \
      --semantic_dim 512 --identity_codec --topk 45 --raw_contribution_weights \
      --max_pixels_per_view "$MAX_PIXELS_PER_VIEW" \
      > "$LOG_DIR/${SCENE}_${tag}_cache.log" 2>&1
  fi
}

prepare_cache "$old_cache" "$dataset/language_features" 1 old
prepare_cache "$l2_cache" "$dataset/language_features_multiscale" 2 l2

if [[ ! -f "$old_codebook/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
    --consensus "$base_consensus" --codes_per_level 4096 4096 \
    --train_samples 262144 --iterations 25 --assignment_chunk 16384 --faiss_gpu \
    --output_dir "$old_codebook" \
    > "$LOG_DIR/${SCENE}_old_codebook.log" 2>&1
fi

for source in old l2; do
  cache_var="${source}_cache"
  cache_dir=${!cache_var}
  for representation in old current; do
    if [[ "$representation" == old ]]; then
      codebook_dir=$old_codebook
    else
      codebook_dir=$current_codebook
    fi
    output="$run_root/${source}_source_${representation}_codebook.json"
    if [[ ! -f "$output" ]]; then
      "$PYTHON_BIN" -u eval_semantic_field_consistency.py \
        --cache_dir "$cache_dir" --codebook_dir "$codebook_dir" \
        --label_dir "$labels" --samples_per_view "$SAMPLES_PER_VIEW" \
        --lovo_topk 4 --seed 0 --output "$output" \
        > "$LOG_DIR/${SCENE}_${source}_source_${representation}.log" 2>&1
    fi
  done
done

echo "held-out consistency probe complete: scene=$SCENE output=$run_root"
