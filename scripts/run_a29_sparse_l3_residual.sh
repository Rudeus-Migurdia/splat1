#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A29 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
E83_ROOT=${E83_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
A17_ROOT=${A17_ROOT:-$ROOT/runs/a17_multi_id_group_hierarchy_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A27_ROOT=${A27_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
A28_ROOT=${A28_ROOT:-$ROOT/runs/a28_complementary_semantic_moe_20260717_223843}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/runs/paper_selection_20260714}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A29}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A29}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SEED=${SEED:-20260717}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$RUN_ROOT" "$LOG_DIR"

old_consensus() {
  printf '%s\n' "$A14_CONT_ROOT/$1/old_split2/consensus.pt"
}

l2_consensus() {
  printf '%s\n' "$A27_ROOT/$1/sam_l2_split2/consensus.pt"
}

l3_consensus() {
  printf '%s\n' "$A27_ROOT/$1/sam_l3_split2/consensus.pt"
}

part_ids() {
  printf '%s\n' "$A17_ROOT/$1/hierarchy/part_group_ids.npy"
}

part_support() {
  printf '%s\n' "$A18_ROOT/$1/interior/part_interior_support.npy"
}

evaluate_scene() {
  local scene=$1
  local source_kind=$2
  local source_path=$3
  local output_dir=$4
  local hypothesis_dir=${5:-}
  if [[ -f "$output_dir/metrics.json" ]]; then
    return
  fi
  mkdir -p "$output_dir"
  local source_args=()
  if [[ "$source_kind" == "consensus" ]]; then
    source_args=(--consensus_path "$source_path")
  else
    source_args=(--codebook_dir "$source_path")
  fi
  local hypothesis_args=()
  if [[ -n "$hypothesis_dir" ]]; then
    hypothesis_args=(--hypothesis_dir "$hypothesis_dir" --hypothesis_readout reliability_blend)
  fi
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "${source_args[@]}" "${hypothesis_args[@]}" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output_dir"
}

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local base_dir=$scene_root/frozen_old_l2_gate
  local base_codebook=$scene_root/base_codebook_16k_x2
  local residual_dir=$scene_root/l3_residual_codebook_k2048
  mkdir -p "$scene_root"

  if [[ ! -f "$base_dir/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/train_frozen_old_l2_gate.py" \
      --old_consensus "$(old_consensus "$scene")" \
      --l2_consensus "$(l2_consensus "$scene")" \
      --output_dir "$base_dir" --device cuda --seed "$SEED" \
      --hidden_dim 16 --max_logit_delta 0.50 \
      --steps 500 --batch_size 4096 --learning_rate 0.002 \
      --stability_floor 0.50 --export_chunk 8192 \
      > "$LOG_DIR/${scene}_base_gate_train.log" 2>&1
  fi

  if [[ ! -f "$base_codebook/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$ROOT/build_gaussian_multilevel_codebook.py" \
      --consensus "$base_dir/consensus.pt" \
      --codes_per_level 16384 16384 --train_samples 100000 \
      --iterations 20 --assignment_chunk 8192 --faiss_gpu --seed "$SEED" \
      --output_dir "$base_codebook" \
      > "$LOG_DIR/${scene}_base_codebook_train.log" 2>&1
  fi

  if [[ ! -f "$residual_dir/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_sparse_l3_residual_codebook.py" \
      --base_consensus "$base_dir/consensus.pt" \
      --l3_consensus "$(l3_consensus "$scene")" \
      --part_group_ids "$(part_ids "$scene")" \
      --part_interior_support "$(part_support "$scene")" \
      --output_dir "$residual_dir" --seed "$SEED" \
      --num_codes 2048 --train_samples 100000 --iterations 20 \
      --assignment_chunk 8192 --chunk_size 8192 --faiss_gpu \
      --stability_floor 0.50 --minimum_boundary 0.25 \
      --minimum_split_cosine 0.85 --minimum_l3_reliability 0.65 \
      --relative_reliability_slack 0.05 \
      --minimum_residual 0.05 --maximum_residual 0.35 \
      --maximum_sparse_fraction 0.10 --alpha_max 0.20 \
      --global_weak_alpha 0.10 \
      > "$LOG_DIR/${scene}_l3_residual_codebook_train.log" 2>&1
  fi

  "$PYTHON_BIN" - "$base_dir" "$base_codebook" "$residual_dir" <<'PY'
import json
import os
import sys
import numpy as np

base_dir, base_codebook_dir, residual_dir = sys.argv[1:]
base = json.load(open(os.path.join(base_dir, "manifest.json")))
base_codebook = json.load(open(os.path.join(base_codebook_dir, "manifest.json")))
residual = json.load(open(os.path.join(residual_dir, "manifest.json")))
sparse = json.load(open(os.path.join(residual_dir, "sparse_residual", "manifest.json")))
global_weak = json.load(open(os.path.join(residual_dir, "global_weak_residual", "manifest.json")))
codes = np.load(os.path.join(residual_dir, "l3_codebook.npy"), mmap_mode="r")
assert base["experts_frozen"] is True
assert base["expert_names"] == ["old", "l2"]
assert base["uses_evaluation_queries"] is False and base["uses_ground_truth"] is False
assert base_codebook["code_counts"] == [16384, 16384]
assert codes.shape == (2048, base_codebook["feature_dim"])
assert residual["selection_diagnostics"]["selected_fraction"] <= 0.1000001
assert residual["uses_evaluation_queries"] is False and residual["uses_ground_truth"] is False
for artifact in (sparse, global_weak):
    assert artifact["representation"] == "sparse_quantized_semantic_hypothesis"
    assert "features" not in artifact
    assert artifact["num_codes"] == 2048
assert sparse["maximum_alpha"] <= 0.200001
print(
    "A29_CONTRACT_OK",
    residual["selection_diagnostics"]["selected_fraction"],
    residual["codebook"]["mean_reconstruction_cosine"],
)
PY

  evaluate_scene "$scene" consensus "$base_dir/consensus.pt" \
    "$scene_root/eval_base_continuous" \
    > "$LOG_DIR/${scene}_base_continuous_eval.log" 2>&1
  evaluate_scene "$scene" codebook "$base_codebook" \
    "$scene_root/eval_base_codebook" \
    > "$LOG_DIR/${scene}_base_codebook_eval.log" 2>&1
  evaluate_scene "$scene" codebook "$base_codebook" \
    "$scene_root/eval_global_weak_l3" "$residual_dir/global_weak_residual" \
    > "$LOG_DIR/${scene}_global_weak_l3_eval.log" 2>&1
  evaluate_scene "$scene" codebook "$base_codebook" \
    "$scene_root/eval_sparse_l3" "$residual_dir/sparse_residual" \
    > "$LOG_DIR/${scene}_sparse_l3_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/train_frozen_old_l2_gate.py" \
  "$SOURCE_DIR/build_sparse_l3_residual_codebook.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$ROOT/build_gaussian_multilevel_codebook.py" \
  "$ROOT/scripts/gpu_guard.py"; do
  [[ -f "$required" ]] || { echo "Missing required source: $required" >&2; exit 2; }
done

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || {
  echo "SCENES and GPU_LIST must have equal lengths" >&2
  exit 2
}
for scene in "${scenes[@]}"; do
  for required in \
    "$(old_consensus "$scene")" \
    "$(l2_consensus "$scene")" \
    "$(l3_consensus "$scene")" \
    "$(part_ids "$scene")" \
    "$(part_support "$scene")" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$E83_ROOT/$scene/eval/metrics.json" \
    "$A20_ROOT/$scene/eval_fine_part/metrics.json" \
    "$A28_ROOT/$scene/diagnostics/readout_ablation/eval_raw_top2_codebook/metrics.json" \
    "$BASELINE_ROOT/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

script_path=${BASH_SOURCE[0]}
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$script_path" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$E83_ROOT" "$A20_ROOT" "$A28_ROOT" \
  "$BASELINE_ROOT" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, e83, a20, a28, baseline_root, raw_t, raw_bt, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metric_names}

summary = {
    "method": "A29 frozen Old/L2 gate with sparse K2048 L3 score residual",
    "seed": int(os.environ["PYTHONHASHSEED"]),
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    residual = json.load(open(os.path.join(root, scene, "l3_residual_codebook_k2048", "manifest.json")))
    base_codebook = json.load(open(os.path.join(root, scene, "base_codebook_16k_x2", "manifest.json")))
    a28_metrics = os.path.join(
        a28, scene, "diagnostics", "readout_ablation", "eval_raw_top2_codebook", "metrics.json"
    )
    summary["scenes"][scene] = {
        "paper_baseline_local": row(
            os.path.join(baseline_root, scene, "baseline", "metrics.json"), baseline_threshold
        ),
        "e8_3": row(os.path.join(e83, scene, "eval", "metrics.json"), threshold),
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), threshold),
        "a28_raw_top2": row(a28_metrics, threshold),
        "a29_base_continuous": row(os.path.join(root, scene, "eval_base_continuous", "metrics.json"), threshold),
        "a29_base_codebook": row(os.path.join(root, scene, "eval_base_codebook", "metrics.json"), threshold),
        "a29_global_weak_l3": row(os.path.join(root, scene, "eval_global_weak_l3", "metrics.json"), threshold),
        "a29_sparse_l3": row(os.path.join(root, scene, "eval_sparse_l3", "metrics.json"), threshold),
        "selected_fraction": residual["selection_diagnostics"]["selected_fraction"],
        "l3_reconstruction_cosine": residual["codebook"]["mean_reconstruction_cosine"],
        "base_reconstruction_cosine": base_codebook["mean_reconstruction_cosine"],
        "base_storage": base_codebook["storage"],
        "sparse_l3_storage": residual["storage"]["sparse_residual"],
    }
methods = (
    "paper_baseline_local",
    "e8_3",
    "a20",
    "a28_raw_top2",
    "a29_base_continuous",
    "a29_base_codebook",
    "a29_global_weak_l3",
    "a29_sparse_l3",
)
for method in methods:
    summary[method + "_mean"] = {
        metric: sum(summary["scenes"][scene][method][metric] for scene in scenes) / len(scenes)
        for metric in metric_names
    }
for reference in ("a20", "a28_raw_top2", "a29_base_codebook", "a29_global_weak_l3"):
    summary["a29_sparse_l3_minus_" + reference] = {
        metric: summary["a29_sparse_l3_mean"][metric] - summary[reference + "_mean"][metric]
        for metric in metric_names
    }
summary["base_quantization_gap"] = {
    metric: summary["a29_base_codebook_mean"][metric] - summary["a29_base_continuous_mean"][metric]
    for metric in metric_names
}
summary["decision"] = {
    "sparse_l3_beats_base_miou": summary["a29_sparse_l3_minus_a29_base_codebook"]["mIoU"] > 0.0,
    "sparse_l3_beats_a20_miou": summary["a29_sparse_l3_minus_a20"]["mIoU"] > 0.0,
    "sparse_l3_preserves_a20_strict_accuracy": summary["a29_sparse_l3_minus_a20"]["mAcc@0.5"] >= 0.0,
    "sparse_l3_beats_global_weak_control": summary["a29_sparse_l3_minus_a29_global_weak_l3"]["mIoU"] > 0.0,
    "all_sparse_fractions_within_budget": all(
        summary["scenes"][scene]["selected_fraction"] <= 0.1000001 for scene in scenes
    ),
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A29 sparse L3 residual probe complete: $RUN_ROOT"
