#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a17_multi_id_group_hierarchy_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a17_multi_id_group_hierarchy_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1
  local source=$A14_ROOT/$scene
  local scene_root=$RUN_ROOT/$scene
  local hierarchy=$scene_root/hierarchy
  mkdir -p "$scene_root"

  if [[ ! -f "$hierarchy/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene hierarchy start"
    "$PYTHON_BIN" -u build_multi_id_group_hierarchy.py \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --base_artifact_dir "$source/base_ids" \
      --candidate_artifact_dir "$source/candidate_ids" \
      --candidate_mask "$source/candidate_mask.npy" \
      --neighbors 8 \
      --spatial_radius_factor 1.5 \
      --rgb_threshold 0.15 \
      --log_scale_threshold 0.7 \
      --semantic_dim 64 \
      --part_base_threshold 0.88 \
      --part_set_threshold 0.90 \
      --object_base_threshold 0.80 \
      --object_set_threshold 0.85 \
      --maximum_part_size 128 \
      --maximum_object_size 2048 \
      --min_part_size 3 \
      --min_object_size 8 \
      --min_part_density 0.5 \
      --min_object_density 0.25 \
      --knn_workers 4 --faiss_gpu \
      --output_dir "$hierarchy" \
      > "$LOG_DIR/${scene}_hierarchy.log" 2>&1
    echo "[$(date +%FT%T)] scene=$scene hierarchy done"
  fi
  if [[ ! -f "$hierarchy/route_expand_hard.npy" ]]; then
    "$PYTHON_BIN" - "$hierarchy/route_expand.npy" "$hierarchy/route_expand_hard.npy" <<'PY'
import numpy as np
import sys

source, output = sys.argv[1:]
reliability = np.load(source)
np.save(output, (reliability > 0.0).astype(np.float32))
PY
  fi

  local variant mask output
  for variant in expand expand_hard consensus; do
    mask=$hierarchy/route_${variant}.npy
    output=$scene_root/eval_$variant
    [[ -f "$output/metrics.json" ]] && continue
    mkdir -p "$output"
    echo "[$(date +%FT%T)] scene=$scene variant=$variant eval start"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$source/candidate_ids" \
      --query_route_base_codebook_dir "$source/base_ids" \
      --codebook_query_route query_positive_blend \
      --query_route_candidate_mask "$mask" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 \
      --output "$output" \
      > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    echo "[$(date +%FT%T)] scene=$scene variant=$variant eval done"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  for scene in "$@"; do
    run_scene "$scene"
  done
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST cannot be empty" >&2; exit 2; }
for scene in "${scenes[@]}"; do
  for required in \
    "$A14_ROOT/$scene/base_ids/manifest.json" \
    "$A14_ROOT/$scene/candidate_ids/manifest.json" \
    "$A14_ROOT/$scene/candidate_mask.npy" \
    "$A14_ROOT/$scene/eval/metrics.json" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for scene_index in "${!scenes[@]}"; do
  scene=${scenes[$scene_index]}
  gpu=${gpus[$((scene_index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a17_multi_id_group_hierarchy_probe.sh" \
      --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, a14_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)
names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row_at(path):
    with open(path) as source:
        payload = json.load(source)
    row = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(row[name]) for name in names}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
variants = ("expand", "expand_hard", "consensus")
for scene in scenes:
    baseline = row_at(os.path.join(a14_root, scene, "eval", "metrics.json"))
    with open(os.path.join(run_root, scene, "hierarchy", "manifest.json")) as source:
        hierarchy = json.load(source)
    rows = {
        "a14_joint32k": baseline,
        "hierarchy": {
            key: hierarchy[key]
            for key in (
                "resident_id_slots_per_valid_gaussian",
                "mean_resident_ids_per_valid_gaussian",
                "full_four_id_fraction_valid",
                "original_candidate_fraction_valid",
                "expanded_fraction_valid",
                "consensus_retained_fraction_of_candidates",
                "mean_expand_reliability",
                "mean_consensus_reliability_on_candidates",
                "edges",
                "part",
                "object",
                "elapsed_seconds",
            )
        },
    }
    for variant in variants:
        row = row_at(
            os.path.join(run_root, scene, "eval_" + variant, "metrics.json")
        )
        rows[variant] = row
        rows[variant + "_minus_a14"] = {
            name: row[name] - baseline[name] for name in names
        }
    summary["scenes"][scene] = rows

for variant in variants:
    summary[variant + "_mean"] = {
        name: sum(summary["scenes"][scene][variant][name] for scene in scenes)
        / len(scenes)
        for name in names
    }
    delta = summary[variant + "_minus_a14_mean"] = {
        name: sum(
            summary["scenes"][scene][variant + "_minus_a14"][name]
            for scene in scenes
        )
        / len(scenes)
        for name in names
    }
    summary[variant + "_go_no_go"] = {
        "required_mean_mIoU_delta": 0.0015,
        "requires_no_per_scene_metric_regression": True,
        "passed": (
            delta["mIoU"] >= 0.0015
            and all(
                summary["scenes"][scene][variant + "_minus_a14"][name] >= -1e-8
                for scene in scenes
                for name in names
            )
        ),
    }

path = os.path.join(run_root, "three_scene_probe.json")
with open(path, "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A17 multi-ID group hierarchy probe complete: $RUN_ROOT"
