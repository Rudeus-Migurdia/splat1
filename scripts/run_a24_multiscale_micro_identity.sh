#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
A22_ROOT=${A22_ROOT:-$ROOT/runs/a22_dual_code_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a24_multiscale_micro_identity_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a24_multiscale_micro_identity_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local artifact=$scene_root/micro_codebook
  local output=$scene_root/eval
  local cache=$ROOT/runs/a16_sparse_view_modes_20260716/$scene/l2_signed_view_cache
  mkdir -p "$scene_root"

  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_multiscale_micro_identity_codebook.py \
      --a20_artifact_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --l2_view_cache_dir "$cache" --output_dir "$artifact" \
      --device cuda --stability_floor 0.5 --min_views_per_split 3 \
      --min_reliability 0.6 --min_disagreement 0.05 \
      > "$LOG_DIR/${scene}_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine micro \
    > "$LOG_DIR/${scene}_contract.log" 2>&1

  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive --group_hierarchy_dir "$artifact" \
      --group_topk 3 --group_readout hypothesis_blend \
      --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi

  "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
    --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --a14_base_dir "$A14_ROOT/$scene/base_ids" \
    --a14_candidate_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
    --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
    --a21_group_dir "$A21_ROOT/$scene/atom_codebook" \
    --a22_group_dir "$A22_ROOT/$scene/dual_codebook" \
    --a24_group_dir "$artifact" --samples 100000 \
    --output "$scene_root/query_consistency.json" \
    > "$LOG_DIR/${scene}_consistency.log" 2>&1
}

if [[ "${1:-}" == --worker ]]; then
  shift
  run_scene "$1"
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
for scene in "${scenes[@]}"; do
  for required in \
    "$ROOT/runs/a16_sparse_view_modes_20260716/$scene/l2_signed_view_cache/manifest.json" \
    "$A20_ROOT/$scene/fine_part_codebook/manifest.json" \
    "$A21_ROOT/$scene/atom_codebook/manifest.json" \
    "$A22_ROOT/$scene/dual_codebook/manifest.json" \
    "$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing input: $required" >&2; exit 2; }
  done
done

pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a24_multiscale_micro_identity.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

for scene in "${scenes[@]}"; do
  "$PYTHON_BIN" analyze_small_object_metrics.py \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --metrics \
      "baseline@${BASELINE_THRESHOLD}=$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json" \
      "A14@${SELECTION_THRESHOLD}=$A14_ROOT/$scene/eval/metrics.json" \
      "A20@${SELECTION_THRESHOLD}=$A20_ROOT/$scene/eval_fine_part/metrics.json" \
      "A24@${SELECTION_THRESHOLD}=$RUN_ROOT/$scene/eval/metrics.json" \
    --output "$RUN_ROOT/$scene/small_object_analysis.json" \
    > "$LOG_DIR/${scene}_small_object.log" 2>&1
done

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A20_ROOT" \
  "$ROOT/runs/paper_selection_20260714" "$SELECTION_THRESHOLD" \
  "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, a14, a20, baseline_root, raw_t, raw_bt, *scenes = sys.argv[1:]
t, bt = float(raw_t), float(raw_bt)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, threshold):
    payload = json.load(open(path))
    selected = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(selected[name]) for name in metrics}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": t,
    "baseline_threshold": bt,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    manifest = json.load(open(os.path.join(root, scene, "micro_codebook", "manifest.json")))
    consistency = json.load(open(os.path.join(root, scene, "query_consistency.json")))["representations"]
    small = json.load(open(os.path.join(root, scene, "small_object_analysis.json")))["small_category_mean_iou"]
    summary["scenes"][scene] = {
        "drsplat_pq_baseline": row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), bt),
        "a14": row(os.path.join(a14, scene, "eval", "metrics.json"), t),
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), t),
        "a24": row(os.path.join(root, scene, "eval", "metrics.json"), t),
        "consistency": consistency,
        "small_object": small,
        "micro_selection": manifest["micro_selection"],
        "vocabulary": manifest["modality_token_counts"],
    }
for method in ("drsplat_pq_baseline", "a14", "a20", "a24"):
    summary[method + "_mean"] = {
        name: sum(summary["scenes"][scene][method][name] for scene in scenes) / len(scenes)
        for name in metrics
    }
summary["a24_minus_a20_mean"] = {
    name: summary["a24_mean"][name] - summary["a20_mean"][name]
    for name in metrics
}
for method, key in {
    "drsplat_pq_baseline": "drsplat_pq_baseline",
    "a20": "a20",
    "a24": "a24_multiscale_micro",
}.items():
    summary[method + "_consistency_mean"] = {
        name: sum(summary["scenes"][scene]["consistency"][key][name] for scene in scenes) / len(scenes)
        for name in ("canonical_split_symmetric_kl", "canonical_split_top1_flip_rate")
    }
summary["small_object_mean"] = {
    method: sum(summary["scenes"][scene]["small_object"][key] for scene in scenes) / len(scenes)
    for method, key in {
        "drsplat_pq_baseline": "baseline",
        "a14": "A14",
        "a20": "A20",
        "a24": "A24",
    }.items()
}
summary["decision"] = {
    "beats_a20_mean_miou": summary["a24_minus_a20_mean"]["mIoU"] > 0.0,
    "beats_a20_small_object": summary["small_object_mean"]["a24"] > summary["small_object_mean"]["a20"],
    "all_vocabulary_contracts_updated": True,
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
