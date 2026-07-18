#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a20_fine_identity_codebook_20260716}
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
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local scene_root=$RUN_ROOT/$scene
  local fine=$scene_root/l1_signed_split2
  local artifact=$scene_root/fine_part_codebook
  local output=$scene_root/eval_fine_part
  mkdir -p "$scene_root"

  if [[ ! -f "$fine/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$fine" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --feature_dir "$dataset/language_features_multiscale" --feature_level 1 \
      --semantic_dim 512 --identity_codec --max_pixels_per_view 0 --topk 45 \
      --raw_contribution_weights --signed_segment_ownership \
      --consensus_only --consensus_chunk_pixels 1024 --consensus_splits 2 \
      > "$LOG_DIR/${scene}_l1_signed_prepare.log" 2>&1
  fi

  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_fine_part_shared_codebook.py \
      --part_artifact_dir "$A18_ROOT/$scene/interior/soft" \
      --fine_consensus "$fine/consensus.pt" \
      --output_dir "$artifact" --device cuda \
      --stability_floor 0.5 --min_group_size 3 --max_group_size 32 \
      --min_reliability 0.6 --min_disagreement 0.05 \
      > "$LOG_DIR/${scene}_fine_part_build.log" 2>&1
  fi

  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine \
    > "$LOG_DIR/${scene}_vocabulary_contract.log" 2>&1

  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$artifact" --group_topk 2 \
      --group_readout hypothesis_blend --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_fine_part_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then shift; run_scene "$1"; exit 0; fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
for scene in "${scenes[@]}"; do
  for required in \
    "$ROOT/drsplat_data/lerf_ovs/$scene/language_features_multiscale" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$A14_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$A14_ROOT/$scene/base_ids/manifest.json" \
    "$A18_ROOT/$scene/interior/soft/manifest.json" \
    "$A18_ROOT/$scene/eval_part_interior_soft/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a20_fine_identity_codebook.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A18_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, a14_root, a18_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row_at(path):
    payload = json.load(open(path))
    row = next(x for x in payload["threshold_summary"] if abs(float(x["selection_threshold"]) - threshold) < 1e-8)
    return {name: float(row[name]) for name in metrics}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    rows = {
        "a14": row_at(os.path.join(a14_root, scene, "eval", "metrics.json")),
        "a18_soft": row_at(os.path.join(a18_root, scene, "eval_part_interior_soft", "metrics.json")),
        "a20_fine_part": row_at(os.path.join(run_root, scene, "eval_fine_part", "metrics.json")),
    }
    manifest = json.load(open(os.path.join(run_root, scene, "fine_part_codebook", "manifest.json")))
    rows["fine_selection"] = manifest["fine_selection"]
    rows["vocabulary"] = manifest["modality_token_counts"]
    rows["delta_vs_a14"] = {name: rows["a20_fine_part"][name] - rows["a14"][name] for name in metrics}
    rows["delta_vs_a18"] = {name: rows["a20_fine_part"][name] - rows["a18_soft"][name] for name in metrics}
    summary["scenes"][scene] = rows
for method in ("a14", "a18_soft", "a20_fine_part"):
    summary[method + "_mean"] = {
        name: sum(summary["scenes"][scene][method][name] for scene in scenes) / len(scenes)
        for name in metrics
    }
summary["a20_minus_a14_mean"] = {
    name: summary["a20_fine_part_mean"][name] - summary["a14_mean"][name]
    for name in metrics
}
summary["a20_minus_a18_mean"] = {
    name: summary["a20_fine_part_mean"][name] - summary["a18_soft_mean"][name]
    for name in metrics
}
summary["decision"] = {
    "beats_a18_mean_miou": summary["a20_minus_a18_mean"]["mIoU"] > 0,
    "no_scene_miou_regression_vs_a18": all(summary["scenes"][scene]["delta_vs_a18"]["mIoU"] >= 0 for scene in scenes),
    "all_vocabulary_contracts_updated": True,
}
with open(os.path.join(run_root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

for scene in "${scenes[@]}"; do
  "$PYTHON_BIN" analyze_small_object_metrics.py \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --metrics \
      "A14=$A14_ROOT/$scene/eval/metrics.json" \
      "A18=$A18_ROOT/$scene/eval_part_interior_soft/metrics.json" \
      "A20=$RUN_ROOT/$scene/eval_fine_part/metrics.json" \
    --selection_threshold "$SELECTION_THRESHOLD" \
    --output "$RUN_ROOT/$scene/small_object_analysis.json" \
    > "$LOG_DIR/${scene}_small_object_analysis.log" 2>&1
done

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A20 fine identity codebook probe complete: $RUN_ROOT"
