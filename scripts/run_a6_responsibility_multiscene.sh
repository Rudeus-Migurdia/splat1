#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_responsibility_multiscene_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_responsibility_multiscene_20260715}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local source_root=$ROOT/runs/multiscale_split_consistency/$scene
  local base=$source_root/fused_w1p5_t005.pt
  local split=$source_root/l2_split2/consensus.pt
  local cache=$output/cache_l2_raw
  local candidate=$output/candidate
  local blend=$output/consensus_alpha050.pt
  local codebook=$output/codebook_k16384x2
  mkdir -p "$output"

  for required in "$dataset" "$labels" "$geometry" "$base" "$split" \
    "$dataset/language_features_multiscale"; do
    [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; return 2; }
  done

  if [[ ! -f "$cache/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$cache" \
      --geometry_checkpoint "$geometry" \
      --feature_dir "$dataset/language_features_multiscale" \
      --feature_level 2 \
      --semantic_dim 512 \
      --identity_codec \
      --topk 45 \
      --max_pixels_per_view 32768 \
      --raw_contribution_weights \
      > "$LOG_DIR/${scene}_cache.log" 2>&1
  fi

  if [[ ! -f "$candidate/consensus.pt" ]]; then
    "$PYTHON_BIN" -u train_a6_semantic_residual.py \
      --cache_dir "$cache" \
      --base_consensus "$base" \
      --split_consensus "$split" \
      --output_dir "$candidate" \
      --iterations 3000 \
      --batch_pixels 2048 \
      --topk 8 \
      --rank 8 \
      --code_lr 0.02 \
      --basis_lr 0.002 \
      --direct_weight 1.0 \
      --lovo_weight 0.5 \
      --contrastive_weight 0.02 \
      --contrastive_temperature 0.07 \
      --agreement_floor 0.65 \
      --direct_confidence_floor 0.25 \
      --anchor_weight 0.2 \
      --code_regularization 0.0001 \
      --train_semantic_opacity \
      --opacity_lr 0.01 \
      --opacity_regularization 0.01 \
      --seed 20260715 \
      > "$LOG_DIR/${scene}_train.log" 2>&1
  fi

  if [[ ! -f "$blend" ]]; then
    "$PYTHON_BIN" -u blend_semantic_consensus.py \
      --base_consensus "$base" \
      --candidate_consensus "$candidate/consensus.pt" \
      --candidate_weight 0.5 \
      --output "$blend" \
      > "$LOG_DIR/${scene}_blend.log" 2>&1
  fi

  if [[ ! -f "$codebook/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$blend" \
      --num_codes 16384 \
      --min_ids 2 \
      --max_ids 2 \
      --min_cosine_gain 0 \
      --target_cosine 1 \
      --train_samples 262144 \
      --iterations 25 \
      --assignment_chunk 4096 \
      --faiss_gpu \
      --seed 20260715 \
      --output_dir "$codebook" \
      > "$LOG_DIR/${scene}_codebook.log" 2>&1
  fi

  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --codebook_dir "$codebook" \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --occupancy_threshold 0.7 \
      --output "$output/eval" \
      > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi
}

run_waldo_full_grid() {
  local scene=waldo_kitchen
  local output=$RUN_ROOT/$scene
  local codebook=$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/codebook
  mkdir -p "$output/eval"
  if [[ ! -f "$output/eval/metrics.json" ]]; then
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$codebook" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --occupancy_threshold 0.7 \
      --output "$output/eval" \
      > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi
if [[ "${1:-}" == "--waldo-worker" ]]; then
  run_waldo_full_grid
  exit 0
fi

specs=("figurines 0" "ramen 1" "teatime 2")
pids=()
for spec in "${specs[@]}"; do
  read -r scene gpu <<< "$spec"
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_a6_responsibility_multiscene.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu 3 --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_a6_responsibility_multiscene.sh" --waldo-worker \
  > "$LOG_DIR/worker_waldo_kitchen.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
  "$RUN_ROOT/figurines/eval/metrics.json" \
  "$RUN_ROOT/ramen/eval/metrics.json" \
  "$RUN_ROOT/teatime/eval/metrics.json" \
  "$RUN_ROOT/waldo_kitchen/eval/metrics.json" \
  --output "$RUN_ROOT/four_scene_metrics.json" \
  > "$RUN_ROOT/four_scene_table.md"

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 responsibility multiscene evaluation complete: $RUN_ROOT"
