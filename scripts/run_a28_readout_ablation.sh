#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A28 source snapshot}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A27_ROOT=${A27_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must point to the isolated A28 run}
LOG_DIR=${LOG_DIR:?LOG_DIR must point to the isolated A28 logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

run_scene() {
  local scene=$1 root=$RUN_ROOT/$1/diagnostics/readout_ablation mode output
  if [[ ! -f "$root/summary.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_semantic_moe_readout_ablation.py" \
      --old_consensus "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" \
      --l2_consensus "$A27_ROOT/$scene/sam_l2_split2/consensus.pt" \
      --l3_consensus "$A27_ROOT/$scene/sam_l3_split2/consensus.pt" \
      --expert_weights "$RUN_ROOT/$scene/moe_continuous/expert_weights.npy" \
      --expert_valid "$RUN_ROOT/$scene/moe_continuous/expert_valid.npy" \
      --output_dir "$root" --device cuda --chunk_size 8192 --auxiliary_scale 1.0 \
      > "$LOG_DIR/${scene}_readout_ablation_build.log" 2>&1
  fi
  for mode in raw_convex raw_top2 old_anchored; do
    output=$root/eval_$mode
    [[ -f "$output/metrics.json" ]] && continue
    mkdir -p "$output"
    "$PYTHON_BIN" -u "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --consensus_path "$root/$mode/consensus.pt" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds 0.45 0.50 0.55 --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_${mode}_eval.log" 2>&1
  done
}

if [[ "${1:-}" == "--worker" ]]; then run_scene "$2"; exit 0; fi
read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || exit 2
pids=()
for index in "${!scenes[@]}"; do
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "${gpus[$index]}" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "${BASH_SOURCE[0]}" --worker "${scenes[$index]}" \
    > "$LOG_DIR/readout_worker_${scenes[$index]}_gpu_${gpus[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "${scenes[@]}" <<'PY'
import json, os, sys
root, *scenes = sys.argv[1:]
modes = ("raw_convex", "raw_top2", "old_anchored")
summary = {"purpose": "posthoc A28 readout failure attribution", "scenes": {}}
for scene in scenes:
    summary["scenes"][scene] = {}
    for mode in modes:
        base = os.path.join(root, scene, "diagnostics", "readout_ablation")
        rows = json.load(open(os.path.join(base, "eval_" + mode, "metrics.json")))["threshold_summary"]
        manifest = json.load(open(os.path.join(base, mode, "manifest.json")))
        summary["scenes"][scene][mode] = {
            "thresholds": [{key: row[key] for key in ("selection_threshold", "mIoU", "mAcc@0.25", "mAcc@0.5")} for row in rows],
            "mean_cosine_to_old_l2_l3": manifest["mean_cosine_to_old_l2_l3"],
        }
for mode in modes:
    for threshold in (0.45, 0.50, 0.55):
        values = []
        for scene in scenes:
            values.append(next(row for row in summary["scenes"][scene][mode]["thresholds"] if abs(row["selection_threshold"] - threshold) < 1e-8))
        summary[f"{mode}_mean_at_{threshold:.2f}"] = {
            metric: sum(row[metric] for row in values) / len(values)
            for metric in ("mIoU", "mAcc@0.25", "mAcc@0.5")
        }
with open(os.path.join(root, "readout_ablation_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/READOUT_ABLATION_COMPLETE"
