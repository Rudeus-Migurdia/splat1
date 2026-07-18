#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A28 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A27_ROOT=${A27_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A28}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A28}
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

level_consensus() {
  printf '%s\n' "$A27_ROOT/$1/sam_l$2_split2/consensus.pt"
}

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local moe_dir=$scene_root/moe_continuous
  local codebook_dir=$scene_root/moe_codebook_16k_x2
  mkdir -p "$scene_root"

  if [[ ! -f "$moe_dir/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/train_complementary_semantic_moe.py" \
      --old_consensus "$(old_consensus "$scene")" \
      --l2_consensus "$(level_consensus "$scene" 2)" \
      --l3_consensus "$(level_consensus "$scene" 3)" \
      --output_dir "$moe_dir" --device cuda --seed "$SEED" \
      --rank 16 --hidden_dim 32 --adapter_scale 0.10 \
      --steps 800 --batch_size 4096 --learning_rate 0.002 \
      --stability_floor 0.50 --minimum_gate_entropy 0.55 \
      --complement_margin 0.50 --export_chunk 8192 \
      > "$LOG_DIR/${scene}_moe_train.log" 2>&1
  fi

  if [[ ! -f "$scene_root/eval_continuous/metrics.json" ]]; then
    mkdir -p "$scene_root/eval_continuous"
    "$PYTHON_BIN" -u "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --consensus_path "$moe_dir/consensus.pt" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$scene_root/eval_continuous" \
      > "$LOG_DIR/${scene}_continuous_eval.log" 2>&1
  fi

  if [[ ! -f "$codebook_dir/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$ROOT/build_gaussian_multilevel_codebook.py" \
      --consensus "$moe_dir/consensus.pt" \
      --codes_per_level 16384 16384 --train_samples 100000 \
      --iterations 20 --assignment_chunk 8192 --faiss_gpu --seed "$SEED" \
      --output_dir "$codebook_dir" \
      > "$LOG_DIR/${scene}_codebook_train.log" 2>&1
  fi

  "$PYTHON_BIN" - "$codebook_dir" <<'PY'
import json
import os
import sys
import numpy as np

root = sys.argv[1]
manifest = json.load(open(os.path.join(root, "manifest.json")))
assert manifest["representation"] == "gaussian_multilevel_residual_codebook"
assert manifest["code_counts"] == [16384, 16384]
ids = np.load(os.path.join(root, manifest["point_code_ids"]), mmap_mode="r")
valid = np.load(os.path.join(root, manifest["valid_mask"]), mmap_mode="r")
assert ids.shape == (manifest["num_gaussians"], 2)
assert valid.shape == (manifest["num_gaussians"],)
print("A28_CODEBOOK_CONTRACT_OK", manifest["mean_reconstruction_cosine"])
PY

  if [[ ! -f "$scene_root/eval_codebook/metrics.json" ]]; then
    mkdir -p "$scene_root/eval_codebook"
    "$PYTHON_BIN" -u "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$codebook_dir" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$scene_root/eval_codebook" \
      > "$LOG_DIR/${scene}_codebook_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/train_complementary_semantic_moe.py" \
  "$ROOT/build_gaussian_multilevel_codebook.py" \
  "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py"; do
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
    "$(level_consensus "$scene" 2)" \
    "$(level_consensus "$scene" 3)" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC_ROOT/$scene/eval/metrics.json" \
    "$A20_ROOT/$scene/eval_fine_part/metrics.json" \
    "$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json"; do
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

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_DISC_ROOT" "$A20_ROOT" \
  "$ROOT/runs/paper_selection_20260714" "$SELECTION_THRESHOLD" \
  "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, a14, a20, baseline_root, raw_t, raw_bt, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {metric: float(item[metric]) for metric in metrics}

summary = {
    "method": "A28 complementary Old/L2/L3 semantic MoE",
    "seed": int(os.environ["PYTHONHASHSEED"]),
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    manifest = json.load(open(os.path.join(root, scene, "moe_continuous", "manifest.json")))
    codebook = json.load(open(os.path.join(root, scene, "moe_codebook_16k_x2", "manifest.json")))
    summary["scenes"][scene] = {
        "pq_baseline": row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), baseline_threshold),
        "a14": row(os.path.join(a14, scene, "eval", "metrics.json"), threshold),
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), threshold),
        "a28_continuous": row(os.path.join(root, scene, "eval_continuous", "metrics.json"), threshold),
        "a28_codebook": row(os.path.join(root, scene, "eval_codebook", "metrics.json"), threshold),
        "moe_diagnostics": manifest["diagnostics"],
        "codebook_reconstruction_cosine": codebook["mean_reconstruction_cosine"],
        "semantic_storage": codebook["storage"],
    }
for method in ("pq_baseline", "a14", "a20", "a28_continuous", "a28_codebook"):
    summary[method + "_mean"] = {
        metric: sum(summary["scenes"][scene][method][metric] for scene in scenes) / len(scenes)
        for metric in metrics
    }
summary["a28_codebook_minus_a20"] = {
    metric: summary["a28_codebook_mean"][metric] - summary["a20_mean"][metric]
    for metric in metrics
}
summary["quantization_gap"] = {
    metric: summary["a28_codebook_mean"][metric] - summary["a28_continuous_mean"][metric]
    for metric in metrics
}
summary["decision"] = {
    "soft_gate_contract": all(
        item["moe_diagnostics"]["mean_normalized_gate_entropy"] >= 0.55
        and min(item["moe_diagnostics"]["fraction_weight_above_0.10"]) > 0.05
        for item in summary["scenes"].values()
    ),
    "beats_a20_mean_miou": summary["a28_codebook_minus_a20"]["mIoU"] > 0.0,
    "does_not_regress_strict_accuracy": summary["a28_codebook_minus_a20"]["mAcc@0.5"] >= 0.0,
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A28 complementary semantic MoE probe complete: $RUN_ROOT"
