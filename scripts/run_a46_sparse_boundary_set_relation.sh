#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
GPU=${GPU:-1}
SEED=${SEED:-20260719}
SCENE=${SCENE:-ramen}
MEMORY_DIR=${MEMORY_DIR:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/equal_four_token_memory}
CONTROL_METRICS=${CONTROL_METRICS:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/eval_equal_query_max/metrics.json}
RELATION_DIR=${RELATION_DIR:-$ROOT/runs/a46b_boundary_conflict_audit_20260721_173347/$SCENE/multiscale_set_relation_audit}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
LABEL_ROOT=${LABEL_ROOT:-$ROOT/drsplat_data/lerf_ovs/label}
OUTPUT_DIR=$RUN_ROOT/$SCENE/eval_sparse_boundary_set_relation
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$OUTPUT_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" \
  "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
  "$MEMORY_DIR/manifest.json" "$RELATION_DIR/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids/manifest.json" \
  "$LABEL_ROOT/$SCENE" "$CONTROL_METRICS"; do
  [[ -e "$required" ]] || { echo "Missing A46 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$MEMORY_DIR" "$RELATION_DIR" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import json
import os
import sys

memory_dir, relation_dir, raw_seed = sys.argv[1:]
seed = int(raw_seed)
memory = json.load(open(os.path.join(memory_dir, "manifest.json")))
relation = json.load(open(os.path.join(relation_dir, "manifest.json")))
assert memory["representation"] == "hierarchical_independent_group_codebooks"
assert memory["resident_slots_required"] == 4
assert memory["reproducibility"]["seed"] == seed
assert relation["representation"] == "heldout_multiscale_set_relation_diagnostic"
assert relation["source_contract"]["fixed_seed"] == seed
assert relation["source_contract"]["evaluation_queries_or_labels_used"] is False
assert relation["source_contract"]["codebooks_trained"] is False
assert relation["metrics"]["stable_set_ambiguous_directed_edges"] >= 1000
assert relation["train_selected_conflict_metrics"]["relative_nll_improvement"] >= 0.10
print("A46_SPARSE_BOUNDARY_INPUT_CONTRACT_OK")
PY

run_worker() {
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
    "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/$SCENE" -m "$GEOMETRY_ROOT/$SCENE" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY_DIR" --group_topk 4 \
    --group_readout equal_query_multiscale_set_relation \
    --group_query_temperature 0.05 \
    --group_relation_graph_dir "$RELATION_DIR" \
    --group_relation_positive_strength 0.20 \
    --group_relation_negative_strength 0.10 \
    --group_relation_maximum_delta 0.05 \
    --label_dir "$LABEL_ROOT/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.50 0.55 --occupancy_threshold 0.7 \
    --output "$OUTPUT_DIR" > "$LOG_DIR/${SCENE}_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$CONTROL_METRICS" "$OUTPUT_DIR/metrics.json" "$SEED" <<'PY'
import json
import os
import sys

root, control_path, candidate_path, raw_seed = sys.argv[1:]


def selected(path, threshold=0.55):
    payload = json.load(open(path))
    row = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return payload, {key: float(row[key]) for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}


control_payload, control = selected(control_path)
candidate_payload, candidate = selected(candidate_path)
delta = {key: candidate[key] - control[key] for key in control}
diagnostics = [
    value.get("multiscale_set_relation", {})
    for value in candidate_payload.get("route_diagnostics", {}).values()
]
summary = {
    "method": "A46 sparse boundary set-valued relation residual",
    "scene": "ramen",
    "seed": int(raw_seed),
    "selection_threshold": 0.55,
    "control_equal_four_token": control,
    "candidate": candidate,
    "delta": delta,
    "mean_corrected_token_slots_per_query": sum(
        float(item.get("corrected_token_slots", 0)) for item in diagnostics
    ) / max(len(diagnostics), 1),
    "decision": {
        "miou_gain_at_least_0p15_points": delta["mIoU"] >= 0.0015,
        "strict_accuracy_not_lower": delta["mAcc@0.5"] >= 0.0,
    },
    "contract": {
        "a33_codebooks_reused_without_training": True,
        "four_tokens_remain_peer_slots": True,
        "relations_selected_without_evaluation_queries_or_labels": True,
        "only_double_split_stable_set_ambiguous_edges_active": True,
    },
}
summary["decision"]["proceed_to_raw_overlapping_proposals"] = all(
    summary["decision"].values()
)
with open(os.path.join(root, "ramen_sparse_boundary_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A46 sparse boundary relation probe complete: $RUN_ROOT"
