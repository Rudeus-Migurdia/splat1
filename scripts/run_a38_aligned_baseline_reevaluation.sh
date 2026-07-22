#!/usr/bin/env bash
set -euo pipefail

# A38: rerun the paper baseline under the exact A37 evaluation protocol while
# retaining the older threshold settings for an auditable sensitivity check.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
GPU_GUARD=${GPU_GUARD:?GPU_GUARD is required}
DATA_ROOT=${DATA_ROOT:-$ROOT}
BASELINE_MODEL_ROOT=${BASELINE_MODEL_ROOT:?BASELINE_MODEL_ROOT is required}
HISTORICAL_BASELINE_ROOT=${HISTORICAL_BASELINE_ROOT:?HISTORICAL_BASELINE_ROOT is required}
A37_SUMMARY=${A37_SUMMARY:?A37_SUMMARY is required}
PQ_INDEX=${PQ_INDEX:-$ROOT/ckpts/pq_index.faiss}
SCENES=${SCENES:-"figurines ramen waldo_kitchen teatime"}
COMPARISON_SCENES=${COMPARISON_SCENES:-"figurines ramen waldo_kitchen"}
SELECTION_THRESHOLDS=${SELECTION_THRESHOLDS:-"0.40 0.45 0.50 0.55"}
OCCUPANCY_THRESHOLD=${OCCUPANCY_THRESHOLD:-0.7}
ALIGNED_THRESHOLD=${ALIGNED_THRESHOLD:-0.55}
SEED=${SEED:-20260719}
GPU=${GPU:-1}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

SITE=${SITE:-$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:?OPENCLIP_PRETRAINED is required}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

evaluate_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene/baseline_aligned
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_miou.py" \
    -s "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" \
    -m "$BASELINE_MODEL_ROOT/${scene}_1_pq_openclip_topk45_weight_128" \
    --checkpoint "$BASELINE_MODEL_ROOT/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    --pq_index "$PQ_INDEX" \
    --label_dir "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds $SELECTION_THRESHOLDS \
    --occupancy_threshold "$OCCUPANCY_THRESHOLD" \
    --output "$output" \
    > "$LOG_DIR/${scene}.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  for scene in $SCENES; do
    echo "[$(date +%FT%T)] scene=$scene stage=evaluation_start"
    evaluate_scene "$scene"
    echo "[$(date +%FT%T)] scene=$scene stage=evaluation_complete"
  done
  exit 0
fi

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_miou.py" \
  "$SOURCE_DIR/evaluation/openclip_encoder.py" \
  "$SOURCE_DIR/lerf_ovs_paper_protocol.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" "$PQ_INDEX" "$A37_SUMMARY"; do
  [[ -f "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done
for scene in $SCENES; do
  for required in \
    "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" \
    "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$BASELINE_MODEL_ROOT/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    "$HISTORICAL_BASELINE_ROOT/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
  done
done

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$HISTORICAL_BASELINE_ROOT" "$A37_SUMMARY" \
  "$SELECTION_THRESHOLDS" "$ALIGNED_THRESHOLD" "$OCCUPANCY_THRESHOLD" \
  "$SEED" "$SCENES" "$COMPARISON_SCENES" <<'PY'
import json
import os
import statistics
import sys

(
    root,
    historical_root,
    a37_summary_path,
    raw_thresholds,
    raw_aligned_threshold,
    raw_occupancy_threshold,
    raw_seed,
    raw_scenes,
    raw_comparison_scenes,
) = sys.argv[1:]
thresholds = [float(value) for value in raw_thresholds.split()]
aligned_threshold = float(raw_aligned_threshold)
scenes = raw_scenes.split()
comparison_scenes = raw_comparison_scenes.split()
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def threshold_row(payload, threshold):
    return next(
        row
        for row in payload["threshold_summary"]
        if abs(float(row["selection_threshold"]) - threshold) < 1e-8
    )


def metrics(row):
    return {name: float(row[name]) for name in metric_names}


def mean(scene_rows, selected_scenes):
    return {
        name: statistics.mean(scene_rows[scene][name] for scene in selected_scenes)
        for name in metric_names
    }


rerun = {}
historical = {}
max_abs_rerun_delta = 0.0
for threshold in thresholds:
    key = f"{threshold:.2f}"
    rerun_rows = {}
    historical_rows = {}
    for scene in scenes:
        rerun_payload = json.load(
            open(os.path.join(root, scene, "baseline_aligned", "metrics.json"))
        )
        historical_payload = json.load(
            open(os.path.join(historical_root, scene, "baseline", "metrics.json"))
        )
        assert rerun_payload["evaluation_protocol"] == "drsplat_3d_selection"
        assert abs(float(rerun_payload["occupancy_threshold"]) - float(raw_occupancy_threshold)) < 1e-8
        rerun_rows[scene] = metrics(threshold_row(rerun_payload, threshold))
        historical_rows[scene] = metrics(threshold_row(historical_payload, threshold))
        max_abs_rerun_delta = max(
            max_abs_rerun_delta,
            *(abs(rerun_rows[scene][name] - historical_rows[scene][name]) for name in metric_names),
        )
    rerun[key] = {
        "scenes": rerun_rows,
        "three_scene_mean": mean(rerun_rows, comparison_scenes),
        "four_scene_mean": mean(rerun_rows, scenes),
    }
    historical[key] = {
        "scenes": historical_rows,
        "three_scene_mean": mean(historical_rows, comparison_scenes),
        "four_scene_mean": mean(historical_rows, scenes),
    }

a37 = json.load(open(a37_summary_path))
best_scale, best_variant = max(
    a37["variants"].items(), key=lambda item: float(item[1]["mean"]["mIoU"])
)
aligned_key = f"{aligned_threshold:.2f}"
baseline_aligned = rerun[aligned_key]["three_scene_mean"]
experiment_aligned = {name: float(best_variant["mean"][name]) for name in metric_names}
absolute_delta = {
    name: experiment_aligned[name] - baseline_aligned[name] for name in metric_names
}
relative_delta_percent = {
    name: 100.0 * absolute_delta[name] / baseline_aligned[name] for name in metric_names
}

summary = {
    "method": "A38 aligned baseline reevaluation",
    "seed": int(raw_seed),
    "evaluation_protocol": "drsplat_3d_selection",
    "occupancy_threshold": float(raw_occupancy_threshold),
    "selection_thresholds": thresholds,
    "scenes": scenes,
    "comparison_scenes": comparison_scenes,
    "baseline_rerun": rerun,
    "historical_baseline": historical,
    "reproduction_check": {
        "max_absolute_metric_delta": max_abs_rerun_delta,
        "exact_within_1e-8": max_abs_rerun_delta < 1e-8,
    },
    "current_experiment": {
        "name": a37["method"],
        "selection_threshold": aligned_threshold,
        "selected_variant": best_scale,
        "three_scene_mean": experiment_aligned,
    },
    "aligned_three_scene_comparison": {
        "selection_threshold": aligned_threshold,
        "baseline": baseline_aligned,
        "experiment": experiment_aligned,
        "experiment_minus_baseline": absolute_delta,
        "relative_gain_percent": relative_delta_percent,
    },
    "threshold_note": {
        "remembered_pair": [0.40, 0.45],
        "historical_summary_baseline_threshold": 0.50,
        "current_experiment_threshold": aligned_threshold,
    },
}
with open(os.path.join(root, "aligned_baseline_summary.json"), "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A38 aligned baseline reevaluation complete: $RUN_ROOT"
