#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-ramen}
CODES_PER_LEVEL=${CODES_PER_LEVEL:-"4096 4096"}
QUERY_KL_WEIGHT=${QUERY_KL_WEIGHT:-0.1}
REFINE_ITERATIONS=${REFINE_ITERATIONS:-2000}
RGR_ALPHA=${RGR_ALPHA:-0.75}
LOG_DIR=${LOG_DIR:-$ROOT/logs/query_kl_codebook_probe}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

level_tag=${CODES_PER_LEVEL// /x}
weight_tag=${QUERY_KL_WEIGHT/./p}
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
checkpoint="$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
pq_index="$ROOT/ckpts/pq_index.faiss"
initial_root="$ROOT/runs/gaussian_multilevel_codebook/${SCENE}_k${level_tag}_seed0"
initial_artifact="$initial_root/artifact"
group_hierarchy="$initial_root/group_hierarchy_top3"
query_bank_root="$ROOT/runs/gaussian_multilevel_codebook/query_banks"
query_bank="$query_bank_root/${SCENE}_anchors256.npy"
output_root="$ROOT/runs/gaussian_multilevel_codebook/${SCENE}_k${level_tag}_qkl${weight_tag}_seed0"
output_artifact="$output_root/artifact"
IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$SCENE")"

for path in "$dataset" "$labels" "$checkpoint" "$pq_index" "$initial_artifact/manifest.json" "$group_hierarchy/manifest.json"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$query_bank_root" "$output_root"

if [[ ! -f "$query_bank" ]]; then
  "$PYTHON_BIN" -u build_semantic_query_bank.py \
    --feature_dir "$dataset/language_features" \
    --num_queries 256 \
    --max_features 200000 \
    --iterations 25 \
    --faiss_gpu \
    --output "$query_bank" \
    > "$LOG_DIR/${SCENE}_query_bank.log" 2>&1
fi

if [[ ! -f "$output_artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u refine_gaussian_codebook_query_kl.py \
    --initial_codebook_dir "$initial_artifact" \
    --drsplat_checkpoint "$checkpoint" \
    --pq_index "$pq_index" \
    --query_bank "$query_bank" \
    --iterations "$REFINE_ITERATIONS" \
    --batch_gaussians 8192 \
    --learning_rate 0.001 \
    --cosine_weight 1.0 \
    --query_kl_weight "$QUERY_KL_WEIGHT" \
    --query_temperature 0.07 \
    --output_dir "$output_artifact" \
    > "$LOG_DIR/${SCENE}_k${level_tag}_qkl${weight_tag}_refine.log" 2>&1
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
  output="$output_root/eval_${variant}_${calibration}_l${calibration_low}_h${calibration_high}"
  if [[ -f "$output/metrics.json" ]]; then
    echo "reuse scene=$SCENE variant=$variant output=$output"
    continue
  fi
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$output_root" \
    --geometry_checkpoint "$checkpoint" \
    --codebook_dir "$output_artifact" \
    --label_dir "$labels" \
    --rgr_alpha "$rgr_alpha" \
    "${group_args[@]}" \
    --score_calibration "$calibration" \
    --calibration_low "$calibration_low" \
    --calibration_high "$calibration_high" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${SCENE}_k${level_tag}_qkl${weight_tag}_${variant}.log" 2>&1
done

echo "query KL codebook probe complete: scene=$SCENE levels=$CODES_PER_LEVEL weight=$QUERY_KL_WEIGHT"
