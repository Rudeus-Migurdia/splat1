#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/e11_spatial_semantic_support_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/e11_spatial_semantic_support_20260716}
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

candidate_path() {
  [[ "$1" == "waldo_kitchen" ]] && printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt" || printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/consensus_alpha050.pt"
}

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene
  mkdir -p "$output"
  "$PYTHON_BIN" -u build_spatial_semantic_support_mask.py \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --candidate_consensus "$(candidate_path "$scene")" \
    --candidate_mask "$SOURCE_ROOT/$scene/candidate_mask.npy" \
    --neighbors 16 \
    --semantic_floor 0.9 \
    --minimum_support 0.25 \
    --minimum_semantic_neighbors 4 \
    --chunk_size 4096 \
    --faiss_gpu \
    --output "$output/candidate_mask.npy" \
    > "$LOG_DIR/${scene}_support.log" 2>&1

  "$PYTHON_BIN" prune_gaussian_codebook.py \
    --artifact_dir "$SOURCE_ROOT/$scene/source_artifacts/candidate_ids" \
    --keep_mask "$output/candidate_mask.npy" \
    --codebook_path "$SOURCE_ROOT/$scene/source_artifacts/candidate_ids/codebook_shared.npy" \
    --output_dir "$output/candidate_ids" \
    > "$LOG_DIR/${scene}_prune.log" 2>&1

  mkdir -p "$output/eval"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$output/candidate_ids" \
    --query_route_base_codebook_dir "$SOURCE_ROOT/$scene/source_artifacts/base_ids" \
    --codebook_query_route query_positive \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --occupancy_threshold 0.7 \
    --output "$output/eval" \
    > "$LOG_DIR/${scene}_eval.log" 2>&1
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
    bash "$ROOT/scripts/run_e11_spatial_semantic_support.sh" --worker "$scene" \
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
    for name, root in (("e8_3", baseline_root), ("e11", run_root)):
        metrics = json.load(open(os.path.join(root, scene, "eval", "metrics.json")))
        row = next(item for item in metrics["threshold_summary"] if abs(item["selection_threshold"] - threshold) < 1e-8)
        rows[name] = {key: row[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}
    rows["delta"] = {key: rows["e11"][key] - rows["e8_3"][key] for key in rows["e11"]}
    summary["scenes"][scene] = rows
for name in ("e8_3", "e11", "delta"):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
with open(os.path.join(run_root, "fixed_threshold_probe.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "E11 spatial semantic support probe complete: $RUN_ROOT"
