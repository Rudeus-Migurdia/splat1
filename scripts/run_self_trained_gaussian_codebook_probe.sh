#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENES=${SCENES:-"figurines"}
CODES_PER_LEVEL=${CODES_PER_LEVEL:-"8192 8192"}
MAX_PIXELS_PER_VIEW=${MAX_PIXELS_PER_VIEW:-32768}
TOPK=${TOPK:-16}
RUN_SUFFIX=${RUN_SUFFIX:-}
RAW_CONTRIBUTION_WEIGHTS=${RAW_CONTRIBUTION_WEIGHTS:-0}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-language_features}
FEATURE_LEVEL=${FEATURE_LEVEL:-1}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-5000}
TRAIN_MODE=${TRAIN_MODE:-lovo_kl}
QUERY_KL_WEIGHT=${QUERY_KL_WEIGHT:-0.1}
LOVO_QUERY_KL_WEIGHT=${LOVO_QUERY_KL_WEIGHT:-0.1}
LOVO_WEIGHT=${LOVO_WEIGHT:-0.5}
NUISANCE_RANK=${NUISANCE_RANK:-4}
RGR_ALPHA=${RGR_ALPHA:-0.75}
FAISS_GPU=${FAISS_GPU:-1}
LOG_DIR=${LOG_DIR:-$ROOT/logs/self_trained_gaussian_codebook}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

scene_group_dir() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
  else
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256"
  fi
}

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

level_tag=${CODES_PER_LEVEL// /x}
  faiss_args=()
if [[ "$FAISS_GPU" == "1" ]]; then
  faiss_args+=(--faiss_gpu)
fi

prepare_weight_args=()
if [[ "$RAW_CONTRIBUTION_WEIGHTS" == "1" ]]; then
  prepare_weight_args+=(--raw_contribution_weights)
fi

for scene in $SCENES; do
  log_scene="${scene}${RUN_SUFFIX}"
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  labels="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  geometry="$ROOT/runs/3dgs/$scene/chkpnt30000.pth"
  feature_dir="$dataset/$FEATURE_DIR_NAME"
  group_dir=$(scene_group_dir "$scene")
  group_codebook="$group_dir/group_features.npy"
  group_assignments="$group_dir/point_group_assignments.npz"
  run_root="$ROOT/runs/self_trained_gaussian_codebook/${scene}_d512_k${level_tag}_p${MAX_PIXELS_PER_VIEW}${RUN_SUFFIX}"
  cache="$run_root/cache"
  initial_codebook="$run_root/initial_codebook"
  group_hierarchy="$run_root/group_hierarchy_top3"
  query_bank="$run_root/query_bank_256.npy"
  trained="$run_root/${TRAIN_MODE}"
  IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$scene")"

  for path in "$dataset" "$labels" "$geometry" "$feature_dir" "$group_codebook" "$group_assignments"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  mkdir -p "$run_root"

  if [[ ! -f "$group_hierarchy/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_compact_group_hierarchy.py \
      --group_features "$group_codebook" \
      --assignments "$group_assignments" \
      --top_m 3 \
      --output_dir "$group_hierarchy" \
      > "$LOG_DIR/${log_scene}_group_hierarchy_top3.log" 2>&1
  fi

  if [[ ! -f "$cache/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$cache" \
      --geometry_checkpoint "$geometry" \
      --feature_dir "$feature_dir" \
      --feature_level "$FEATURE_LEVEL" \
      --semantic_dim 512 \
      --identity_codec \
      --topk "$TOPK" \
      --max_pixels_per_view "$MAX_PIXELS_PER_VIEW" \
      "${prepare_weight_args[@]}" \
      > "$LOG_DIR/${log_scene}_prepare_d512.log" 2>&1
  fi

  if [[ ! -f "$initial_codebook/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
      --consensus "$cache/consensus.pt" \
      --codes_per_level $CODES_PER_LEVEL \
      --train_samples 262144 \
      --iterations 25 \
      --assignment_chunk 16384 \
      "${faiss_args[@]}" \
      --output_dir "$initial_codebook" \
      > "$LOG_DIR/${log_scene}_initialize_k${level_tag}.log" 2>&1
  fi

  if [[ ! -f "$query_bank" ]]; then
    "$PYTHON_BIN" -u build_semantic_query_bank.py \
      --feature_dir "$feature_dir" \
      --num_queries 256 \
      --max_features 200000 \
      --iterations 25 \
      "${faiss_args[@]}" \
      --output "$query_bank" \
      > "$LOG_DIR/${log_scene}_query_bank.log" 2>&1
  fi

  train_args=(
    --cache_dir "$cache"
    --initial_codebook_dir "$initial_codebook"
    --output "$trained"
    --iterations "$TRAIN_ITERATIONS"
    --batch_pixels 4096
    --codebook_lr 0.001
    --query_bank "$query_bank"
  )
  case "$TRAIN_MODE" in
    direct)
      train_args+=(--lovo_weight 0.0 --nuisance_rank 0)
      ;;
    lovo)
      train_args+=(--lovo_weight "$LOVO_WEIGHT" --nuisance_rank "$NUISANCE_RANK")
      ;;
    lovo_kl)
      train_args+=(
        --lovo_weight "$LOVO_WEIGHT"
        --nuisance_rank "$NUISANCE_RANK"
        --query_kl_weight "$QUERY_KL_WEIGHT"
        --lovo_query_kl_weight "$LOVO_QUERY_KL_WEIGHT"
      )
      ;;
    lovo_kl_svi)
      train_args+=(
        --lovo_weight "$LOVO_WEIGHT"
        --nuisance_rank "$NUISANCE_RANK"
        --query_kl_weight "$QUERY_KL_WEIGHT"
        --lovo_query_kl_weight "$LOVO_QUERY_KL_WEIGHT"
        --view_sampling segment_importance
        --importance_max_base_kl 0.25
      )
      ;;
    *)
      echo "Unknown TRAIN_MODE=$TRAIN_MODE" >&2
      exit 2
      ;;
  esac
  if [[ ! -f "$trained/artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u train_gaussian_multilevel_codebook.py "${train_args[@]}" \
      > "$LOG_DIR/${log_scene}_${TRAIN_MODE}_train.log" 2>&1
  fi

  if [[ ! -f "$trained/query_consistency.json" ]]; then
    "$PYTHON_BIN" -u eval_semantic_field_consistency.py \
      --cache_dir "$cache" \
      --codebook_dir "$trained/artifact" \
      --label_dir "$labels" \
      --output "$trained/query_consistency.json" \
      --samples_per_view 256 \
      --lovo_topk 4 \
      > "$LOG_DIR/${log_scene}_${TRAIN_MODE}_query_consistency.log" 2>&1
  fi

  for variant in codebook_only codebook_rgr; do
    rgr_alpha=0.0
    group_args=()
    if [[ "$variant" == "codebook_rgr" ]]; then
      rgr_alpha=$RGR_ALPHA
      group_args+=(
        --group_hierarchy_dir "$group_hierarchy"
        --group_aggregation weighted
        --point_gate_floor 0.1
        --point_gate_power 1.0
      )
    fi
    output="$trained/eval_${variant}_${calibration}_l${calibration_low}_h${calibration_high}"
    if [[ -f "$output/metrics.json" ]]; then
      echo "reuse scene=$scene mode=$TRAIN_MODE variant=$variant"
      continue
    fi
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$run_root" \
      --geometry_checkpoint "$geometry" \
      --codebook_dir "$trained/artifact" \
      --label_dir "$labels" \
      --rgr_alpha "$rgr_alpha" \
      "${group_args[@]}" \
      --score_calibration "$calibration" \
      --calibration_low "$calibration_low" \
      --calibration_high "$calibration_high" \
      --thresholds $THRESHOLDS \
      --output "$output" \
      > "$LOG_DIR/${log_scene}_${TRAIN_MODE}_${variant}.log" 2>&1
  done
done

echo "self-trained Gaussian codebook probe complete: scenes=$SCENES mode=$TRAIN_MODE"
