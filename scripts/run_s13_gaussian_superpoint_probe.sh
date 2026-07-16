#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/s13_gaussian_superpoint_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/s13_gaussian_superpoint_20260716}
GPU_LIST=${GPU_LIST:-"1 2"}
SCENES=${SCENES:-"figurines waldo_kitchen"}

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
  local base=$SOURCE_ROOT/$scene/source_artifacts/base_ids
  local candidate=$SOURCE_ROOT/$scene/pruned_candidate_ids
  mkdir -p "$output"

  "$PYTHON_BIN" -u build_gaussian_superpoint_support.py \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --base_artifact_dir "$base" \
    --candidate_mask "$SOURCE_ROOT/$scene/candidate_mask.npy" \
    --neighbors 6 \
    --spatial_radius_factor 1.5 \
    --rgb_threshold 0.15 \
    --log_scale_threshold 0.7 \
    --semantic_threshold 0.85 \
    --semantic_dim 64 \
    --maximum_superpoint_size 512 \
    --chunk_size 65536 \
    --faiss_gpu \
    --seed 20260716 \
    --output_dir "$output/superpoints" \
    > "$LOG_DIR/${scene}_build.log" 2>&1

  for variant in s0_geometry_rgb s1_semantic; do
    mkdir -p "$output/$variant/eval"
    if [[ ! -f "$output/$variant/eval/metrics.json" ]]; then
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
        -m "$ROOT/runs/3dgs/$scene" \
        --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
        --codebook_dir "$candidate" \
        --query_route_base_codebook_dir "$base" \
        --codebook_query_route query_positive_blend \
        --query_route_candidate_mask "$output/superpoints/${variant}_support.npy" \
        --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
        --evaluation_protocol drsplat_3d_selection \
        --occupancy_threshold 0.7 \
        --output "$output/$variant/eval" \
        > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    fi
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || { echo "SCENES and GPU_LIST must have equal lengths" >&2; exit 2; }
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_s13_gaussian_superpoint_probe.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_ROOT" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, baseline_root, *scenes = sys.argv[1:]
threshold = 0.55
summary = {"fixed_selection_threshold": threshold, "scenes": {}}
for scene in scenes:
    rows = {}
    paths = {
        "e8_3": os.path.join(baseline_root, scene, "eval", "metrics.json"),
        "s0": os.path.join(run_root, scene, "s0_geometry_rgb", "eval", "metrics.json"),
        "s1": os.path.join(run_root, scene, "s1_semantic", "eval", "metrics.json"),
    }
    for name, path in paths.items():
        metrics = json.load(open(path))
        row = next(item for item in metrics["threshold_summary"] if abs(item["selection_threshold"] - threshold) < 1e-8)
        rows[name] = {key: row[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}
    for name in ("s0", "s1"):
        rows[name + "_delta"] = {key: rows[name][key] - rows["e8_3"][key] for key in rows[name]}
    summary["scenes"][scene] = rows
for name in ("e8_3", "s0", "s1", "s0_delta", "s1_delta"):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
with open(os.path.join(run_root, "fixed_threshold_probe.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "S0/S1 Gaussian superpoint probe complete: $RUN_ROOT"
