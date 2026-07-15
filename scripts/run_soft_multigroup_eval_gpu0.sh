#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/anlanfan/Dr-Splat}"
GPU="${GPU:-0}"
DATASET="${DATASET:-${ROOT}/drsplat_data/lerf_ovs/figurines}"
LABEL_DIR="${LABEL_DIR:-${ROOT}/drsplat_data/lerf_ovs/label/figurines}"
START_CKPT="${START_CKPT:-${ROOT}/runs/3dgs/figurines/chkpnt30000.pth}"
PQ_INDEX="${PQ_INDEX:-${ROOT}/ckpts/pq_index.faiss}"
OUT="${OUT:-${ROOT}/runs/prototypes/mask_group_lift/group_soft_topk10_tokens}"
LOG="${LOG:-${ROOT}/logs/soft_multigroup_eval_gpu${GPU}.log}"

mkdir -p "${ROOT}/logs" "$(dirname "${OUT}")"
cd "${ROOT}"

source scripts/drsplat_env.sh
SITE="${ROOT}/.venv/lib/python3.9/site-packages"
export PYTHONPATH="${SITE}:${SITE}/setuptools/_vendor:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${ROOT}/.cache/xdg"
export TORCH_HOME="${ROOT}/.cache/torch"
export HF_HOME="${ROOT}/.cache/huggingface"
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
export ROOT GPU DATASET LABEL_DIR START_CKPT PQ_INDEX OUT LOG

run_all() {
  rm -rf "${OUT}"
  .venv/bin/python -u prototype_mask_group_lift.py \
    -s "${DATASET}" \
    -m "${OUT}" \
    --start_checkpoint "${START_CKPT}" \
    --pq_index "${PQ_INDEX}" \
    --output_model "${OUT}" \
    --label_dir "${LABEL_DIR}" \
    --feature_level 1 \
    --topk 10 \
    --mask_keep_gaussians 2048 \
    --group_keep_gaussians 8192 \
    --candidate_seed_gaussians 64 \
    --index_group_gaussians 160 \
    --min_group_observations 1 \
    --assignment_mode soft \
    --keep_point_groups 4 \
    --soft_score_power 1.0

  .venv/bin/python summarize_multigroup_artifact.py --artifact_dir "${OUT}"

  for AGG in max weighted score_max; do
    .venv/bin/python -u eval_lerf_ovs_multigroup_miou.py \
      -s "${DATASET}" \
      -m "${OUT}" \
      --checkpoint "${OUT}/chkpnt0.pth" \
      --label_dir "${LABEL_DIR}" \
      --group_features "${OUT}/group_features.npy" \
      --assignments "${OUT}/point_group_assignments.npz" \
      --aggregation "${AGG}" \
      --score_power 1.0 \
      --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
      --output "${OUT}/eval/lerf_ovs_multigroup_${AGG}"
  done
}

.venv/bin/python -u scripts/gpu_guard.py \
  --gpu "${GPU}" \
  --hold-mb 512 \
  --max-used-mb 256 \
  --max-utilization 5 \
  -- bash -lc "$(declare -f run_all); run_all" 2>&1 | tee "${LOG}"
