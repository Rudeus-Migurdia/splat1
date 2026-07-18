#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
A22_ROOT=${A22_ROOT:-$ROOT/runs/a22_dual_code_20260716}
A24_ROOT=${A24_ROOT:-$ROOT/runs/a24_multiscale_micro_identity_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a24_multiscale_micro_identity_20260716}
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
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$LOG_DIR"

run_scene() {
  local scene=$1
  local artifact=$A24_ROOT/$scene/micro_interior
  local output=$A24_ROOT/$scene/eval_micro_interior
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" build_micro_interior_membership.py \
      --source_artifact_dir "$A24_ROOT/$scene/micro_codebook" \
      --part_interior_support "$A18_ROOT/$scene/interior/part_interior_support.npy" \
      --output_dir "$artifact" > "$LOG_DIR/${scene}_micro_interior_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine micro \
    > "$LOG_DIR/${scene}_micro_interior_contract.log" 2>&1
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive --group_hierarchy_dir "$artifact" \
      --group_topk 3 --group_readout hypothesis_blend \
      --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_micro_interior_eval.log" 2>&1
  fi
  "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
    --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --a14_base_dir "$A14_ROOT/$scene/base_ids" \
    --a14_candidate_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
    --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
    --a21_group_dir "$A21_ROOT/$scene/atom_codebook" \
    --a22_group_dir "$A22_ROOT/$scene/dual_codebook" \
    --a24_group_dir "$artifact" --samples 100000 \
    --output "$A24_ROOT/$scene/query_consistency_micro_interior.json" \
    > "$LOG_DIR/${scene}_micro_interior_consistency.log" 2>&1
}

if [[ "${1:-}" == --worker ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a24_micro_interior_ablation.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_micro_interior_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"
date +%FT%T > "$A24_ROOT/MICRO_INTERIOR_COMPLETE"
