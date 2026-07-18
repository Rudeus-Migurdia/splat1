#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A27_ROOT=${A27_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must point to the isolated A28 run}
LOG_DIR=${LOG_DIR:?LOG_DIR must point to the isolated A28 logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
THRESHOLDS=${THRESHOLDS:-"0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70"}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

consensus_path() {
  local scene=$1 method=$2
  case "$method" in
    old) printf '%s\n' "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" ;;
    l2) printf '%s\n' "$A27_ROOT/$scene/sam_l2_split2/consensus.pt" ;;
    l3) printf '%s\n' "$A27_ROOT/$scene/sam_l3_split2/consensus.pt" ;;
    a28) printf '%s\n' "$RUN_ROOT/$scene/moe_continuous/consensus.pt" ;;
    *) echo "Unknown diagnostic method: $method" >&2; return 2 ;;
  esac
}

run_scene() {
  local scene=$1 method source output
  for method in old l2 l3 a28; do
    source=$(consensus_path "$scene" "$method")
    output=$RUN_ROOT/$scene/diagnostics/eval_${method}_grid
    [[ -f "$output/metrics.json" ]] && continue
    mkdir -p "$output"
    read -r -a thresholds <<< "$THRESHOLDS"
    "$PYTHON_BIN" -u "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --consensus_path "$source" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "${thresholds[@]}" --occupancy_threshold 0.7 \
      --output "$output" \
      > "$LOG_DIR/${scene}_${method}_threshold_grid.log" 2>&1
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || exit 2
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "${BASH_SOURCE[0]}" --worker "$scene" \
    > "$LOG_DIR/diagnostic_worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, *scenes = sys.argv[1:]
methods = ("old", "l2", "l3", "a28")
summary = {
    "purpose": "posthoc score-calibration diagnosis only",
    "must_not_be_used_for_training_or_model_selection": True,
    "scenes": {},
}
for scene in scenes:
    summary["scenes"][scene] = {}
    for method in methods:
        path = os.path.join(root, scene, "diagnostics", f"eval_{method}_grid", "metrics.json")
        rows = json.load(open(path))["threshold_summary"]
        best = max(rows, key=lambda row: float(row["mIoU"]))
        fixed = next(row for row in rows if abs(float(row["selection_threshold"]) - 0.55) < 1e-8)
        summary["scenes"][scene][method] = {
            "fixed_0.55": {key: fixed[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
            "oracle_threshold": float(best["selection_threshold"]),
            "oracle": {key: best[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
        }

for method in methods:
    first_path = os.path.join(root, scenes[0], "diagnostics", f"eval_{method}_grid", "metrics.json")
    thresholds = [float(row["selection_threshold"]) for row in json.load(open(first_path))["threshold_summary"]]
    curve = []
    for threshold in thresholds:
        values = []
        for scene in scenes:
            path = os.path.join(root, scene, "diagnostics", f"eval_{method}_grid", "metrics.json")
            row = next(
                item for item in json.load(open(path))["threshold_summary"]
                if abs(float(item["selection_threshold"]) - threshold) < 1e-8
            )
            values.append(row)
        curve.append({
            "selection_threshold": threshold,
            "mIoU": sum(float(row["mIoU"]) for row in values) / len(values),
            "mAcc@0.25": sum(float(row["mAcc@0.25"]) for row in values) / len(values),
            "mAcc@0.5": sum(float(row["mAcc@0.5"]) for row in values) / len(values),
        })
    summary[method + "_shared_threshold_curve"] = curve
    summary[method + "_best_shared_threshold"] = max(curve, key=lambda row: row["mIoU"])

with open(os.path.join(root, "expert_threshold_diagnostics.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/DIAGNOSTIC_COMPLETE"
echo "A28 expert threshold diagnostics complete: $RUN_ROOT"
