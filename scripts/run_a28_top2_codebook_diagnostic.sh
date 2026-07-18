#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must point to the isolated A28 run}
LOG_DIR=${LOG_DIR:?LOG_DIR must point to the isolated A28 logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SEED=${SEED:-20260717}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

run_scene() {
  local scene=$1 base=$RUN_ROOT/$1/diagnostics/readout_ablation
  local consensus=$base/raw_top2/consensus.pt
  local codebook=$base/raw_top2_codebook_16k_x2
  local output=$base/eval_raw_top2_codebook
  if [[ ! -f "$codebook/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$ROOT/build_gaussian_multilevel_codebook.py" \
      --consensus "$consensus" --codes_per_level 16384 16384 \
      --train_samples 100000 --iterations 20 --assignment_chunk 8192 \
      --faiss_gpu --seed "$SEED" --output_dir "$codebook" \
      > "$LOG_DIR/${scene}_raw_top2_codebook_train.log" 2>&1
  fi
  "$PYTHON_BIN" - "$codebook" <<'PY'
import json, os, sys
manifest = json.load(open(os.path.join(sys.argv[1], "manifest.json")))
assert manifest["code_counts"] == [16384, 16384]
assert manifest["mean_reconstruction_cosine"] > 0.99
print("A28_TOP2_CODEBOOK_CONTRACT_OK", manifest["mean_reconstruction_cosine"])
PY
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u "$ROOT/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$codebook" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds 0.55 --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_raw_top2_codebook_eval.log" 2>&1
  fi
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
    > "$LOG_DIR/top2_codebook_worker_${scenes[$index]}_gpu_${gpus[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A20_ROOT" "${scenes[@]}" <<'PY'
import json, os, sys
root, a20, *scenes = sys.argv[1:]
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")
def row(path):
    item = next(
        value for value in json.load(open(path))["threshold_summary"]
        if abs(float(value["selection_threshold"]) - 0.55) < 1e-8
    )
    return {metric: float(item[metric]) for metric in metrics}
summary = {"seed": int(os.environ["PYTHONHASHSEED"]), "selection_threshold": 0.55, "scenes": {}}
for scene in scenes:
    base = os.path.join(root, scene, "diagnostics", "readout_ablation")
    codebook = json.load(open(os.path.join(base, "raw_top2_codebook_16k_x2", "manifest.json")))
    summary["scenes"][scene] = {
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json")),
        "raw_top2_continuous": row(os.path.join(base, "eval_raw_top2", "metrics.json")),
        "raw_top2_codebook": row(os.path.join(base, "eval_raw_top2_codebook", "metrics.json")),
        "reconstruction_cosine": codebook["mean_reconstruction_cosine"],
        "storage": codebook["storage"],
    }
for method in ("a20", "raw_top2_continuous", "raw_top2_codebook"):
    summary[method + "_mean"] = {
        metric: sum(summary["scenes"][scene][method][metric] for scene in scenes) / len(scenes)
        for metric in metrics
    }
summary["codebook_minus_a20"] = {
    metric: summary["raw_top2_codebook_mean"][metric] - summary["a20_mean"][metric]
    for metric in metrics
}
summary["quantization_gap"] = {
    metric: summary["raw_top2_codebook_mean"][metric] - summary["raw_top2_continuous_mean"][metric]
    for metric in metrics
}
summary["posthoc_diagnostic"] = True
with open(os.path.join(root, "top2_codebook_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/TOP2_CODEBOOK_COMPLETE"
