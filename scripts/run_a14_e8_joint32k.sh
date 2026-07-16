#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
REFERENCE_ROOT=${REFERENCE_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a14_e8_joint32k_20260716}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

base_consensus() {
  printf '%s\n' "$A14_ROOT/$1/fused_w1p5_t005.pt"
}

candidate_consensus() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt"
  else
    printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$scene/consensus_alpha050.pt"
  fi
}

link_shared_vocabulary() {
  local artifact=$1
  local vocabulary=$2
  rm -f "$artifact/codebook_shared.npy"
  ln -s "$vocabulary" "$artifact/codebook_shared.npy"
}

assign_mode() {
  local consensus=$1
  local vocabulary=$2
  local output=$3
  local log=$4
  if [[ ! -f "$output/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$consensus" \
      --codebook "$vocabulary" \
      --num_codes 32768 \
      --min_ids 2 --max_ids 2 \
      --min_cosine_gain 0 --target_cosine 1 \
      --assignment_chunk 4096 \
      --faiss_gpu --seed 20260716 \
      --output_dir "$output" \
      > "$log" 2>&1
  fi
  if [[ ! -L "$output/codebook_shared.npy" ]]; then
    link_shared_vocabulary "$output" "$vocabulary"
  fi
}

run_scene() {
  local scene=$1
  local base candidate output joint vocabulary base_ids candidate_ids mask pruned
  base=$(base_consensus "$scene")
  candidate=$(candidate_consensus "$scene")
  output=$RUN_ROOT/$scene
  joint=$output/joint_vocabulary
  vocabulary=$joint/codebook_shared.npy
  base_ids=$output/base_ids
  candidate_ids=$output/candidate_ids
  mask=$output/candidate_mask.npy
  pruned=$output/pruned_candidate_ids
  mkdir -p "$output"

  echo "[$(date +%FT%T)] scene=$scene joint vocabulary start"
  if [[ ! -f "$joint/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_joint_semantic_vocabulary.py \
      --base_consensus "$base" \
      --candidate_consensus "$candidate" \
      --num_codes 32768 \
      --samples_per_source 262144 \
      --iterations 25 \
      --faiss_gpu --seed 20260716 \
      --output_dir "$joint" \
      > "$LOG_DIR/${scene}_joint_vocab.log" 2>&1
  fi

  assign_mode "$base" "$vocabulary" "$base_ids" "$LOG_DIR/${scene}_base_ids.log"
  assign_mode "$candidate" "$vocabulary" "$candidate_ids" "$LOG_DIR/${scene}_candidate_ids.log"

  if [[ ! -f "${mask%.npy}.json" ]]; then
    "$PYTHON_BIN" -u build_novelty_route_mask.py \
      --base_consensus "$base" \
      --candidate_consensus "$candidate" \
      --base_codebook_dir "$base_ids" \
      --candidate_codebook_dir "$candidate_ids" \
      --noise_ratio 1 \
      --output "$mask" \
      > "$LOG_DIR/${scene}_novelty_mask.log" 2>&1
  fi

  "$PYTHON_BIN" -u prune_gaussian_codebook.py \
    --artifact_dir "$candidate_ids" \
    --keep_mask "$mask" \
    --codebook_path "$vocabulary" \
    --output_dir "$pruned" \
    > "$LOG_DIR/${scene}_prune.log" 2>&1

  if [[ ! -f "$output/eval/metrics.json" ]]; then
    mkdir -p "$output/eval"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
      -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$pruned" \
      --query_route_base_codebook_dir "$base_ids" \
      --codebook_query_route query_positive \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --occupancy_threshold 0.7 \
      --output "$output/eval" \
      > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi
  echo "[$(date +%FT%T)] scene=$scene complete"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker=$1
  shift
  echo "[$(date +%FT%T)] worker=$worker scenes=$*"
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
    "$(base_consensus "$scene")" \
    "$(candidate_consensus "$scene")" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$REFERENCE_ROOT/$scene/eval/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for gpu_index in "${!gpus[@]}"; do
  worker_scenes=()
  for scene_index in "${!scenes[@]}"; do
    if (( scene_index % ${#gpus[@]} == gpu_index )); then
      worker_scenes+=("${scenes[$scene_index]}")
    fi
  done
  [[ "${#worker_scenes[@]}" -gt 0 ]] || continue
  gpu=${gpus[$gpu_index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a14_e8_joint32k.sh" \
      --worker "gpu-$gpu" "${worker_scenes[@]}" \
    > "$LOG_DIR/worker_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

metrics=()
for scene in "${scenes[@]}"; do
  metrics+=("$RUN_ROOT/$scene/eval/metrics.json")
done
summary_args=()
[[ "${#scenes[@]}" -eq 4 ]] || summary_args+=(--allow_partial)
"$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
  "${metrics[@]}" "${summary_args[@]}" \
  --output "$RUN_ROOT/paper_metrics.json" \
  > "$RUN_ROOT/paper_table.md"

"$PYTHON_BIN" - "$RUN_ROOT" "$REFERENCE_ROOT" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, reference_root, *scenes = sys.argv[1:]
threshold = 0.55

def row_at(path):
    with open(path) as source:
        metrics = json.load(source)
    return next(
        row for row in metrics["threshold_summary"]
        if abs(float(row["selection_threshold"]) - threshold) < 1e-8
    )

summary = {"selection_threshold": threshold, "scenes": {}}
for scene in scenes:
    candidate = row_at(os.path.join(root, scene, "eval", "metrics.json"))
    reference = row_at(os.path.join(reference_root, scene, "eval", "metrics.json"))
    mask = json.load(open(os.path.join(root, scene, "candidate_mask.json")))
    base = json.load(open(os.path.join(root, scene, "base_ids", "manifest.json")))
    pruned = json.load(
        open(os.path.join(root, scene, "pruned_candidate_ids", "manifest.json"))
    )
    summary["scenes"][scene] = {
        "e8_3": {key: reference[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
        "a14_e8_joint32k": {
            key: candidate[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
        },
        "delta": {
            key: candidate[key] - reference[key]
            for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
        },
        "candidate_fraction": mask["candidate_fraction"],
        "average_base_ids": base["average_ids_per_valid_gaussian"],
        "average_candidate_ids": pruned["average_ids_per_valid_gaussian"],
    }
for name in ("e8_3", "a14_e8_joint32k", "delta"):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
with open(os.path.join(root, "fixed_threshold_comparison.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A14/E8 joint-32K experiment complete: $RUN_ROOT"
