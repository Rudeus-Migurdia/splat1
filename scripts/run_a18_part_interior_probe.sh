#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A17_ROOT=${A17_ROOT:-$ROOT/runs/a17_multi_id_group_hierarchy_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a18_hierarchical_group_codebook_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
cd "$ROOT"; source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline

run_scene() {
  local scene=$1
  local root=$A18_ROOT/$scene/interior
  if [[ ! -f "$root/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_part_interior_group_artifacts.py \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --hierarchy_dir "$A17_ROOT/$scene/hierarchy" \
      --exact_artifact_dir "$A18_ROOT/$scene/exact_part_extension" \
      --output_root "$root" --neighbors 8 --minimum_same_neighbors 4 \
      --interior_fraction 0.75 --knn_workers 4 --faiss_gpu \
      > "$LOG_DIR/${scene}_interior_build.log" 2>&1
  fi
  local variant out
  for variant in soft hard; do
    out=$A18_ROOT/$scene/eval_part_interior_$variant
    [[ -f "$out/metrics.json" ]] && continue
    mkdir -p "$out"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$root/$variant" --group_topk 1 \
      --group_readout hypothesis_blend --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection --selection_thresholds 0.55 \
      --occupancy_threshold 0.7 --output "$out" \
      > "$LOG_DIR/${scene}_interior_${variant}_eval.log" 2>&1
  done
}
if [[ "${1:-}" == "--worker" ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"; read -r -a gpus <<< "$GPU_LIST"; pids=()
for i in "${!scenes[@]}"; do s=${scenes[$i]}; g=${gpus[$((i%${#gpus[@]}))]};
  "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$g" --hold-mb 384 --max-used-mb 256 \
    --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a18_part_interior_probe.sh" --worker "$s" \
    > "$LOG_DIR/worker_interior_${s}_gpu_${g}.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=$?; done; exit "$status"
