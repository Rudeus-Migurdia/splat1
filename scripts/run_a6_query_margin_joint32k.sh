#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_query_margin_joint32k_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_query_margin_joint32k_20260715}

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

candidate_path() {
  if [[ "$1" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt"
  else
    printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/consensus_alpha050.pt"
  fi
}

link_shared_vocabulary() {
  local artifact=$1
  local vocabulary=$2
  rm -f "$artifact/codebook_shared.npy"
  ln -s "$vocabulary" "$artifact/codebook_shared.npy"
}

assign_mode() {
  local consensus=$1
  local vocabulary=$2
  local output=$3
  local log=$4
  if [[ ! -f "$output/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$consensus" \
      --codebook "$vocabulary" \
      --num_codes 32768 \
      --min_ids 2 \
      --max_ids 2 \
      --min_cosine_gain 0 \
      --target_cosine 1 \
      --assignment_chunk 4096 \
      --faiss_gpu \
      --seed 20260715 \
      --output_dir "$output" \
      > "$log" 2>&1
  fi
  if [[ ! -L "$output/codebook_shared.npy" ]]; then
    link_shared_vocabulary "$output" "$vocabulary"
  fi
}

run_scene() {
  local scene=$1
  local base
  local candidate
  base=$(base_path "$scene")
  candidate=$(candidate_path "$scene")
  local output=$RUN_ROOT/$scene
  local joint=$output/joint_vocabulary
  local vocabulary=$joint/codebook_shared.npy
  local base_ids=$output/base_ids
  local candidate_ids=$output/candidate_ids
  mkdir -p "$output"

  if [[ ! -f "$joint/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_joint_semantic_vocabulary.py \
      --base_consensus "$base" \
      --candidate_consensus "$candidate" \
      --num_codes 32768 \
      --samples_per_source 262144 \
      --iterations 25 \
      --faiss_gpu \
      --seed 20260715 \
      --output_dir "$joint" \
      > "$LOG_DIR/${scene}_joint_vocab.log" 2>&1
  fi

  assign_mode "$base" "$vocabulary" "$base_ids" "$LOG_DIR/${scene}_base_ids.log"
  assign_mode "$candidate" "$vocabulary" "$candidate_ids" "$LOG_DIR/${scene}_candidate_ids.log"

  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$candidate_ids" \
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
    bash "$ROOT/scripts/run_a6_query_margin_joint32k.sh" --worker "$scene" \
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
    joint = json.load(open(os.path.join(root, scene, "joint_vocabulary", "manifest.json")))
    base = json.load(open(os.path.join(root, scene, "base_ids", "manifest.json")))
    candidate = json.load(open(os.path.join(root, scene, "candidate_ids", "manifest.json")))
    unique_bytes = (
        joint["storage_bytes_fp16"]
        + base["storage"]["total_semantic_bytes"]
        - base["storage"]["codebook_bytes_fp16"]
        + candidate["storage"]["total_semantic_bytes"]
        - candidate["storage"]["codebook_bytes_fp16"]
    )
    rows[scene] = {
        "num_codes": joint["num_codes"],
        "average_base_ids": base["average_ids_per_valid_gaussian"],
        "average_candidate_ids": candidate["average_ids_per_valid_gaussian"],
        "base_reconstruction_cosine": base["mean_reconstruction_cosine"],
        "candidate_reconstruction_cosine": candidate["mean_reconstruction_cosine"],
        "unique_storage_megabytes": unique_bytes / 2**20,
    }
with open(os.path.join(root, "storage_summary.json"), "w") as output:
    json.dump(rows, output, indent=2)
print(json.dumps(rows, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 joint-32k query-margin evaluation complete: $RUN_ROOT"
