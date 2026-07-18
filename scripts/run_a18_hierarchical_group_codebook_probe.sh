#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A17_ROOT=${A17_ROOT:-$ROOT/runs/a17_multi_id_group_hierarchy_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a18_hierarchical_group_codebook_20260716}
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
  local discrete=$A14_DISC_ROOT/$scene
  local hierarchy=$A17_ROOT/$scene/hierarchy
  local scene_root=$RUN_ROOT/$scene
  local group_codebook=$scene_root/group_codebook
  mkdir -p "$scene_root"

  if [[ ! -f "$group_codebook/manifest.json" || ! -f "$group_codebook/continuous_diagnostic/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene group codebook start"
    "$PYTHON_BIN" -u build_hierarchical_group_semantic_codebook.py \
      --hierarchy_dir "$hierarchy" \
      --old_consensus "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" \
      --aux_consensus "$A14_CONT_ROOT/$scene/l2_split2/consensus.pt" \
      --shared_vocabulary "$discrete/joint_vocabulary/codebook_shared.npy" \
      --output_dir "$group_codebook" \
      --device cuda \
      --stability_floor 0.5 \
      --min_reliability 0.25 \
      --min_part_size 3 \
      --min_object_size 8 \
      --max_aux_weight 1.5 \
      --temperature 0.05 \
      --save_continuous_diagnostic \
      > "$LOG_DIR/${scene}_build.log" 2>&1
    echo "[$(date +%FT%T)] scene=$scene group codebook done"
  fi

  local variant topk readout group_dir output
  for variant in part part_object part_blend part_object_blend part_continuous; do
    topk=1
    [[ "$variant" == part_object* ]] && topk=2
    readout=hypothesis
    [[ "$variant" == *_blend ]] && readout=hypothesis_blend
    [[ "$variant" == "part_continuous" ]] && readout=hypothesis_blend
    group_dir=$group_codebook
    [[ "$variant" == "part_continuous" ]] && group_dir=$group_codebook/continuous_diagnostic
    output=$scene_root/eval_$variant
    [[ -f "$output/metrics.json" ]] && continue
    mkdir -p "$output"
    echo "[$(date +%FT%T)] scene=$scene variant=$variant eval start"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$discrete/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$discrete/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$group_dir" \
      --group_topk "$topk" \
      --group_readout "$readout" \
      --group_route_fraction 1 \
      --group_route_priority reliability_gain \
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
for scene in "${scenes[@]}"; do
  for required in \
    "$A17_ROOT/$scene/hierarchy/manifest.json" \
    "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" \
    "$A14_CONT_ROOT/$scene/l2_split2/consensus.pt" \
    "$A14_DISC_ROOT/$scene/joint_vocabulary/codebook_shared.npy" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/eval/metrics.json"; do
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
    bash "$ROOT/scripts/run_a18_hierarchical_group_codebook_probe.sh" \
      --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_DISC_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, reference_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row_at(path):
    with open(path) as source:
        payload = json.load(source)
    row = next(x for x in payload["threshold_summary"] if abs(x["selection_threshold"] - threshold) < 1e-8)
    return {name: float(row[name]) for name in metrics}

summary = {"evaluation_protocol": "drsplat_3d_selection", "selection_threshold": threshold, "occupancy_threshold": 0.7, "scenes": {}}
variants = ("part", "part_object", "part_blend", "part_object_blend", "part_continuous")
for scene in scenes:
    baseline = row_at(os.path.join(reference_root, scene, "eval", "metrics.json"))
    with open(os.path.join(run_root, scene, "group_codebook", "manifest.json")) as source:
        codebook = json.load(source)
    rows = {"a14_joint32k": baseline, "group_codebook": {key: codebook[key] for key in ("num_group_codes", "covered_fraction", "mean_ids_per_covered_gaussian", "quantization", "levels", "storage", "elapsed_seconds")}}
    for variant in variants:
        row = row_at(os.path.join(run_root, scene, "eval_" + variant, "metrics.json"))
        rows[variant] = row
        rows[variant + "_minus_a14"] = {name: row[name] - baseline[name] for name in metrics}
    summary["scenes"][scene] = rows
for variant in variants:
    summary[variant + "_mean"] = {name: sum(summary["scenes"][scene][variant][name] for scene in scenes) / len(scenes) for name in metrics}
    delta = summary[variant + "_minus_a14_mean"] = {name: sum(summary["scenes"][scene][variant + "_minus_a14"][name] for scene in scenes) / len(scenes) for name in metrics}
    summary[variant + "_go_no_go"] = {"required_mean_mIoU_delta": 0.0015, "requires_no_per_scene_metric_regression": True, "passed": delta["mIoU"] >= 0.0015 and all(summary["scenes"][scene][variant + "_minus_a14"][name] >= -1e-8 for scene in scenes for name in metrics)}
with open(os.path.join(run_root, "three_scene_probe.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A18 hierarchical group codebook probe complete: $RUN_ROOT"
