#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2}
GPU_ID=${GPU_ID:-0}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

base=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt
candidate=$ROOT/runs/a6_semantic_residual_waldo_20260715/m2b_split_ccl_opacity/consensus.pt
blend=$RUN_ROOT/consensus_alpha050.pt
codebook=$RUN_ROOT/codebook
dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth

run_pipeline() {
  if [[ ! -f "$blend" ]]; then
    "$PYTHON_BIN" -u blend_semantic_consensus.py \
      --base_consensus "$base" \
      --candidate_consensus "$candidate" \
      --candidate_weight 0.5 \
      --output "$blend" \
      > "$LOG_DIR/blend.log" 2>&1
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
      > "$LOG_DIR/codebook.log" 2>&1
  fi
  if [[ ! -f "$RUN_ROOT/eval/metrics.json" ]]; then
    mkdir -p "$RUN_ROOT/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
      --geometry_checkpoint "$geometry" \
      --codebook_dir "$codebook" \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds 0.55 \
      --occupancy_threshold 0.7 \
      --output "$RUN_ROOT/eval" \
      > "$LOG_DIR/eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_pipeline
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "$GPU_ID" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_a6_responsibility_codebook.sh" --worker \
  > "$LOG_DIR/worker.log" 2>&1

"$PYTHON_BIN" - "$ROOT" "$RUN_ROOT" <<'PY'
import json
import os
import sys

root, run_root = sys.argv[1:]
paths = {
    "a6": os.path.join(root, "runs/paper_selection_20260714/waldo_kitchen/a6/metrics.json"),
    "continuous_alpha050": os.path.join(
        root, "runs/a6_semantic_residual_waldo_20260715/blends/alpha050/metrics.json"
    ),
    "discrete_alpha050_k16384x2": os.path.join(run_root, "eval/metrics.json"),
}
rows = {}
for name, path in paths.items():
    result = json.load(open(path))
    row = next(
        item
        for item in result["threshold_summary"]
        if abs(item["selection_threshold"] - 0.55) < 1e-9
    )
    rows[name] = {key: row[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}
base = rows["a6"]["mIoU"]
for row in rows.values():
    row["delta_mIoU_vs_a6"] = row["mIoU"] - base
manifest = json.load(open(os.path.join(run_root, "codebook/manifest.json")))
summary = {
    "results": rows,
    "codebook": {
        "num_codes": manifest["num_codes"],
        "average_ids_per_valid_gaussian": manifest["average_ids_per_valid_gaussian"],
        "mean_reconstruction_cosine": manifest["mean_reconstruction_cosine"],
        "semantic_storage_megabytes": manifest["storage"]["total_semantic_bytes"] / 2**20,
    },
}
with open(os.path.join(run_root, "comparison.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
