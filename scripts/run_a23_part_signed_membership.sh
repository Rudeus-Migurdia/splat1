#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
A22_ROOT=${A22_ROOT:-$ROOT/runs/a22_dual_code_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a23_part_signed_membership_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a23_part_signed_membership_20260716}
SCENE=${SCENE:-waldo_kitchen}
GPU=${GPU:-1}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$RUN_ROOT/$SCENE" "$LOG_DIR"

CACHE=$ROOT/runs/self_trained_gaussian_codebook/${SCENE}_d512_k4096x4096_p32768_topk45raw/cache
ARTIFACT=$RUN_ROOT/$SCENE/signed_membership
EVAL=$RUN_ROOT/$SCENE/eval

for required in \
  "$CACHE/manifest.json" \
  "$A18_ROOT/$SCENE/interior/part_interior_support.npy" \
  "$A20_ROOT/$SCENE/fine_part_codebook/manifest.json" \
  "$A21_ROOT/$SCENE/atom_codebook/manifest.json" \
  "$A22_ROOT/$SCENE/dual_codebook/manifest.json" \
  "$ROOT/runs/paper_selection_20260714/$SCENE/baseline/metrics.json"; do
  [[ -e "$required" ]] || { echo "Missing input: $required" >&2; exit 2; }
done

run_probe() {
  if [[ ! -f "$ARTIFACT/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_part_conditioned_signed_membership.py \
      --source_artifact_dir "$A20_ROOT/$SCENE/fine_part_codebook" \
      --cache_dir "$CACHE" \
      --feature_dir "$ROOT/drsplat_data/lerf_ovs/$SCENE/language_features_multiscale" \
      --feature_levels 1 2 \
      --part_interior_support "$A18_ROOT/$SCENE/interior/part_interior_support.npy" \
      --boundary_threshold 0.75 --interior_distance 4 --interior_floor 0.25 \
      --min_split_views 3 --output_dir "$ARTIFACT" \
      > "$LOG_DIR/${SCENE}_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$ARTIFACT" --required base part fine \
    > "$LOG_DIR/${SCENE}_contract.log" 2>&1

  if [[ ! -f "$EVAL/metrics.json" ]]; then
    mkdir -p "$EVAL"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$SCENE/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$SCENE/base_ids" \
      --codebook_query_route query_positive --group_hierarchy_dir "$ARTIFACT" \
      --group_topk 2 --group_readout hypothesis_blend \
      --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$EVAL" > "$LOG_DIR/${SCENE}_eval.log" 2>&1
  fi

  "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
    --fine_consensus "$A20_ROOT/$SCENE/l1_signed_split2/consensus.pt" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
    --pq_checkpoint "$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --a14_base_dir "$A14_ROOT/$SCENE/base_ids" \
    --a14_candidate_dir "$A14_ROOT/$SCENE/pruned_candidate_ids" \
    --a20_group_dir "$A20_ROOT/$SCENE/fine_part_codebook" \
    --a21_group_dir "$A21_ROOT/$SCENE/atom_codebook" \
    --a22_group_dir "$A22_ROOT/$SCENE/dual_codebook" \
    --a23_group_dir "$ARTIFACT" --samples 100000 \
    --output "$RUN_ROOT/$SCENE/query_consistency.json" \
    > "$LOG_DIR/${SCENE}_consistency.log" 2>&1

  "$PYTHON_BIN" analyze_small_object_metrics.py \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
    --metrics \
      "baseline@${BASELINE_THRESHOLD}=$ROOT/runs/paper_selection_20260714/$SCENE/baseline/metrics.json" \
      "A14@${SELECTION_THRESHOLD}=$A14_ROOT/$SCENE/eval/metrics.json" \
      "A20@${SELECTION_THRESHOLD}=$A20_ROOT/$SCENE/eval_fine_part/metrics.json" \
      "A23@${SELECTION_THRESHOLD}=$EVAL/metrics.json" \
    --output "$RUN_ROOT/$SCENE/small_object_analysis.json" \
    > "$LOG_DIR/${SCENE}_small_object.log" 2>&1
}

if [[ "${1:-}" == --worker ]]; then
  run_probe
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
  bash "$ROOT/scripts/run_a23_part_signed_membership.sh" --worker \
  > "$LOG_DIR/worker_${SCENE}_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A20_ROOT" \
  "$ROOT/runs/paper_selection_20260714" "$SCENE" \
  "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" <<'PY'
import json
import os
import sys

root, a14, a20, baseline_root, scene, raw_t, raw_bt = sys.argv[1:]
t, bt = float(raw_t), float(raw_bt)
names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, threshold):
    payload = json.load(open(path))
    selected = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(selected[name]) for name in names}

manifest = json.load(open(os.path.join(root, scene, "signed_membership", "manifest.json")))
consistency = json.load(open(os.path.join(root, scene, "query_consistency.json")))["representations"]
small = json.load(open(os.path.join(root, scene, "small_object_analysis.json")))["small_category_mean_iou"]
result = {
    "protocol": "waldo_preregistered_raw_top45_probe",
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": t,
    "baseline_threshold": bt,
    "occupancy_threshold": 0.7,
    "scene": scene,
    "drsplat_pq_baseline": row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), bt),
    "a14": row(os.path.join(a14, scene, "eval", "metrics.json"), t),
    "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), t),
    "a23": row(os.path.join(root, scene, "eval", "metrics.json"), t),
    "consistency": consistency,
    "small_object": small,
    "membership": manifest["membership"],
}
result["delta_a23_vs_a20"] = {
    name: result["a23"][name] - result["a20"][name] for name in names
}
result["expand_to_three_scenes"] = bool(
    result["delta_a23_vs_a20"]["mIoU"] >= 0.0015
    and result["delta_a23_vs_a20"]["mAcc@0.25"] >= 0.0
    and result["delta_a23_vs_a20"]["mAcc@0.5"] >= 0.0
    and small["A23"] >= small["A20"]
)
with open(os.path.join(root, "waldo_probe_summary.json"), "w") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
