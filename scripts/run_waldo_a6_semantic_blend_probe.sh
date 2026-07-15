#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_semantic_residual_waldo_20260715/blends}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_semantic_residual_waldo_20260715/blends}

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
dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth

run_weight() {
  local tag=$1
  local weight=$2
  local output=$RUN_ROOT/$tag
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" \
    --consensus_path "$candidate" \
    --ignore_consensus_semantic_opacity \
    --consensus_blend_base "$base" \
    --consensus_candidate_weight "$weight" \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 \
    --occupancy_threshold 0.7 \
    --output "$output" \
    > "$LOG_DIR/eval_${tag}.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_weight "$2" "$3"
  exit 0
fi

pids=()
specs=("alpha025 0.25 0" "alpha050 0.50 1" "alpha075 0.75 2")
for spec in "${specs[@]}"; do
  read -r tag weight gpu <<< "$spec"
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_waldo_a6_semantic_blend_probe.sh" \
      --worker "$tag" "$weight" \
    > "$LOG_DIR/worker_${tag}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$ROOT" "$RUN_ROOT" <<'PY'
import json
import os
import sys

root, run_root = sys.argv[1:]
paths = {
    "a6": os.path.join(root, "runs/paper_selection_20260714/waldo_kitchen/a6/metrics.json"),
    "alpha025": os.path.join(run_root, "alpha025/metrics.json"),
    "alpha050": os.path.join(run_root, "alpha050/metrics.json"),
    "alpha075": os.path.join(run_root, "alpha075/metrics.json"),
    "alpha100": os.path.join(
        root, "runs/a6_semantic_residual_waldo_20260715/m2b_eval_no_opacity/metrics.json"
    ),
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
with open(os.path.join(run_root, "comparison.json"), "w") as output:
    json.dump(rows, output, indent=2)
print(json.dumps(rows, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
