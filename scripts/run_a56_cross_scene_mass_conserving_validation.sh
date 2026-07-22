#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A56 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A56 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A31_RUN=${A31_RUN:-$ROOT/runs/a31_teatime_equal_four_token_validation_20260718_104609}
A55_RUN=${A55_RUN:-$ROOT/runs/a55_semantic_mass_conserving_group_retrieval_20260721_221930}
A14_DISC=${A14_DISC:-$ROOT/runs/a14_e8_joint32k_20260716}
SCENES=(figurines teatime waldo_kitchen)
CACHE_ROOT=$RUN_ROOT/.cache

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch HF_HOME=$CACHE_ROOT/huggingface
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

cache_dir() {
  case "$1" in
    figurines|teatime) printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/cache_l2_raw" ;;
    waldo_kitchen) printf '%s\n' "$ROOT/runs/split_consistency_heldout/waldo_kitchen/cache_l2" ;;
    *) return 2 ;;
  esac
}

memory_dir() {
  case "$1" in
    figurines|waldo_kitchen) printf '%s\n' "$A33_RUN/$1/equal_four_token_memory" ;;
    teatime) printf '%s\n' "$A31_RUN/teatime/equal_four_token_memory" ;;
    *) return 2 ;;
  esac
}

reference_metrics() {
  case "$1" in
    figurines|waldo_kitchen) printf '%s\n' "$A33_RUN/$1/eval_equal_query_max/metrics.json" ;;
    teatime) printf '%s\n' "$A31_RUN/teatime/eval_equal_query_max/metrics.json" ;;
    *) return 2 ;;
  esac
}

for required in \
  "$SOURCE_DIR/export_raw_sam_proposals.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$SOURCE_DIR/build_query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/build_hierarchical_semantic_memory.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/scripts/run_a47_raw_proposal_identifiability.sh" \
  "$SOURCE_DIR/scripts/run_a56_eval_worker.sh" \
  "$ROOT/scripts/gpu_guard.py" "$A55_RUN/PROBE_COMPLETE"; do
  [[ -e "$required" ]] || { echo "Missing A56 input: $required" >&2; exit 2; }
done

for scene in "${SCENES[@]}"; do
  for required in \
    "$(cache_dir "$scene")/manifest.json" \
    "$(memory_dir "$scene")/manifest.json" \
    "$(reference_metrics "$scene")" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/$scene/images" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC/$scene/base_ids/manifest.json" \
    "$A14_DISC/$scene/pruned_candidate_ids/manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing $scene input: $required" >&2; exit 2; }
  done
done

run_raw_source() {
  local scene=$1
  local gpus=$2
  local root=$RUN_ROOT/a47_source_$scene
  local logs=$LOG_DIR/a47_$scene
  mkdir -p "$root" "$logs"
  if [[ -f "$root/PROBE_COMPLETE" ]]; then
    echo "Reuse completed A47 source: $scene"
    return
  fi
  env ROOT="$ROOT" RUN_ROOT="$root" LOG_DIR="$logs" SOURCE_DIR="$SOURCE_DIR" \
    PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpus" SEED="$SEED" SCENE="$scene" \
    CACHE_DIR="$(cache_dir "$scene")" \
    bash "$SOURCE_DIR/scripts/run_a47_raw_proposal_identifiability.sh"
}

raw_pids=()
run_raw_source figurines "0 1" > "$LOG_DIR/figurines_raw_driver.log" 2>&1 & raw_pids+=("$!")
run_raw_source teatime "2" > "$LOG_DIR/teatime_raw_driver.log" 2>&1 & raw_pids+=("$!")
run_raw_source waldo_kitchen "3 4" > "$LOG_DIR/waldo_kitchen_raw_driver.log" 2>&1 & raw_pids+=("$!")
raw_status=0
for pid in "${raw_pids[@]}"; do wait "$pid" || raw_status=1; done
[[ "$raw_status" -eq 0 ]] || { echo "A56 raw proposal stage failed" >&2; exit 1; }

build_spatial() {
  local scene=$1
  local audit=$RUN_ROOT/a47_source_$scene/$scene/entity_identifiability_audit
  local output=$RUN_ROOT/$scene/query_conditioned_spatial_posterior
  mkdir -p "$RUN_ROOT/$scene"
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u \
    "$SOURCE_DIR/build_query_conditioned_spatial_posterior.py" \
    --a47_audit_dir "$audit" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --output_dir "$output" --seed "$SEED" \
    --coverage_threshold 0.30 --minimum_spatial_jaccard 0.35 \
    --minimum_semantic_cosine 0.75 --minimum_association 0.40 \
    --spatial_weight 0.85 --temporal_neighbors 2 --minimum_persistence_views 3 \
    --merge_jaccard 0.85 --merge_semantic_cosine 0.90 --maximum_slots 4096 \
    --atom_neighbors 8 --minimum_atom_contact 0.05 \
    --core_coverage_threshold 0.30 --boundary_coverage_threshold 0.05 \
    --minimum_owner_membership 0.02
}

spatial_pids=()
for scene in "${SCENES[@]}"; do
  build_spatial "$scene" > "$LOG_DIR/${scene}_spatial_build.log" 2>&1 &
  spatial_pids+=("$!")
done
spatial_status=0
for pid in "${spatial_pids[@]}"; do wait "$pid" || spatial_status=1; done
[[ "$spatial_status" -eq 0 ]] || { echo "A56 spatial build stage failed" >&2; exit 1; }

eval_pids=()
for spec in "figurines:1" "teatime:2" "waldo_kitchen:3"; do
  scene=${spec%%:*}
  gpu=${spec##*:}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
    env ROOT="$ROOT" SOURCE_DIR="$SOURCE_DIR" RUN_ROOT="$RUN_ROOT" LOG_DIR="$LOG_DIR" \
      SCENE="$scene" GPU="$gpu" MEMORY="$(memory_dir "$scene")" \
      SPATIAL="$RUN_ROOT/$scene/query_conditioned_spatial_posterior" \
      PYTHON_BIN="$PYTHON_BIN" A14_DISC="$A14_DISC" \
      bash "$SOURCE_DIR/scripts/run_a56_eval_worker.sh" \
    > "$LOG_DIR/${scene}_eval_driver.log" 2>&1 &
  eval_pids+=("$!")
done
eval_status=0
for pid in "${eval_pids[@]}"; do wait "$pid" || eval_status=1; done
[[ "$eval_status" -eq 0 ]] || { echo "A56 evaluation stage failed" >&2; exit 1; }

"$PYTHON_BIN" - "$RUN_ROOT" "$A33_RUN" "$A31_RUN" "$A55_RUN" <<'PY'
import json, os, sys
root, a33, a31, a55 = sys.argv[1:]
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path):
    item = json.load(open(path))["threshold_summary"][0]
    return {key: item[key] for key in (*metric_names, "per_category")}

references = {
    "figurines": os.path.join(a33, "figurines", "eval_equal_query_max", "metrics.json"),
    "teatime": os.path.join(a31, "teatime", "eval_equal_query_max", "metrics.json"),
    "waldo_kitchen": os.path.join(a33, "waldo_kitchen", "eval_equal_query_max", "metrics.json"),
    "ramen": os.path.join(a33, "ramen", "eval_equal_query_max", "metrics.json"),
}
results = {}
for scene in ("figurines", "teatime", "waldo_kitchen"):
    reference = row(references[scene])
    control = row(os.path.join(root, scene, "eval_control_equal_query_max", "metrics.json"))
    candidate = row(os.path.join(root, scene, "eval_mass_conserving_anchor", "metrics.json"))
    results[scene] = {
        "reference": reference,
        "control": control,
        "a55": candidate,
        "control_delta_miou": control["mIoU"] - reference["mIoU"],
        "a55_delta": {key: candidate[key] - reference[key] for key in metric_names},
        "beats_reference_miou": candidate["mIoU"] > reference["mIoU"],
    }
ramen_reference = row(references["ramen"])
ramen_candidate = row(os.path.join(a55, "ramen", "eval_mass_conserving_anchor", "metrics.json"))
results["ramen"] = {
    "reference": ramen_reference,
    "a55": ramen_candidate,
    "a55_delta": {key: ramen_candidate[key] - ramen_reference[key] for key in metric_names},
    "beats_reference_miou": ramen_candidate["mIoU"] > ramen_reference["mIoU"],
}
summary = {
    "experiment": "A56_cross_scene_mass_conserving_validation",
    "fixed_parameters": {
        "seed": 20260719,
        "semantic_preservation_quantile": 0.75,
        "group_temperature": 0.05,
        "maximum_penalty": 0.08,
        "selection_threshold": 0.55,
        "occupancy_threshold": 0.7,
    },
    "teatime_reference_note": "A31 equal-query-max; A33 did not include teatime",
    "scenes": results,
    "all_four_scenes_beat_reference_miou": all(
        item["beats_reference_miou"] for item in results.values()
    ),
    "mean_reference_miou": sum(item["reference"]["mIoU"] for item in results.values()) / 4,
    "mean_a55_miou": sum(item["a55"]["mIoU"] for item in results.values()) / 4,
}
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A56 cross-scene A55 validation complete: $RUN_ROOT"
