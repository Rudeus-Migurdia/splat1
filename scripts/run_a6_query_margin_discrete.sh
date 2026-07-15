#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_query_margin_discrete_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_query_margin_discrete_20260715}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

base_path() {
  if [[ "$1" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt"
  else
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/$1/fused_w1p5_t005.pt"
  fi
}

candidate_codebook_path() {
  if [[ "$1" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/codebook"
  else
    printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/codebook_k16384x2"
  fi
}

run_scene() {
  local scene=$1
  local base
  local candidate
  base=$(base_path "$scene")
  candidate=$(candidate_codebook_path "$scene")
  local output=$RUN_ROOT/$scene
  local base_ids=$output/base_ids_in_shared_vocab
  mkdir -p "$output"

  if [[ ! -f "$base_ids/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$base" \
      --codebook "$candidate/codebook_shared.npy" \
      --num_codes 16384 \
      --min_ids 2 \
      --max_ids 2 \
      --min_cosine_gain 0 \
      --target_cosine 1 \
      --assignment_chunk 4096 \
      --faiss_gpu \
      --seed 20260715 \
      --output_dir "$base_ids" \
      > "$LOG_DIR/${scene}_base_ids.log" 2>&1
    rm -f "$base_ids/codebook_shared.npy"
    ln -s "$candidate/codebook_shared.npy" "$base_ids/codebook_shared.npy"
  fi

  if [[ ! -L "$base_ids/codebook_shared.npy" ]]; then
    rm -f "$base_ids/codebook_shared.npy"
    ln -s "$candidate/codebook_shared.npy" "$base_ids/codebook_shared.npy"
  fi

  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$candidate" \
      --query_route_base_codebook_dir "$base_ids" \
      --codebook_query_route margin_positive \
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

scenes=(figurines ramen teatime waldo_kitchen)
pids=()
for gpu in 0 1 2 3; do
  scene=${scenes[$gpu]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_a6_query_margin_discrete.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

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

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import os
import sys

root = sys.argv[1]
rows = {}
for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
    base = json.load(open(os.path.join(root, scene, "base_ids_in_shared_vocab", "manifest.json")))
    result = json.load(open(os.path.join(root, scene, "eval", "metrics.json")))
    candidate = result["codebook_manifest"]
    unique_bytes = (
        candidate["storage"]["total_semantic_bytes"]
        + base["storage"]["total_semantic_bytes"]
        - candidate["storage"]["codebook_bytes_fp16"]
    )
    rows[scene] = {
        "average_base_ids": base["average_ids_per_valid_gaussian"],
        "average_candidate_ids": candidate["average_ids_per_valid_gaussian"],
        "base_reconstruction_cosine": base["mean_reconstruction_cosine"],
        "candidate_reconstruction_cosine": candidate["mean_reconstruction_cosine"],
        "unique_shared_vocabulary_storage_megabytes": unique_bytes / 2**20,
    }
with open(os.path.join(root, "storage_summary.json"), "w") as output:
    json.dump(rows, output, indent=2)
print(json.dumps(rows, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 discrete query-margin evaluation complete: $RUN_ROOT"
