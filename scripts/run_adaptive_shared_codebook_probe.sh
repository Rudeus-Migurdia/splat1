#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENE=${SCENE:-waldo_kitchen}
NUM_CODES=${NUM_CODES:-16384}
MAX_IDS=${MAX_IDS:-1}
MIN_COSINE_GAIN=${MIN_COSINE_GAIN:-0.002}
TARGET_COSINE=${TARGET_COSINE:-0.995}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-5000}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-262144}
KMEANS_ITERATIONS=${KMEANS_ITERATIONS:-25}
ASSIGNMENT_CHUNK=${ASSIGNMENT_CHUNK:-4096}
SOURCE_CODEBOOK=${SOURCE_CODEBOOK:-}
RUN_TAG=${RUN_TAG:-}
FAISS_GPU=${FAISS_GPU:-1}
LOG_DIR=${LOG_DIR:-$ROOT/logs/adaptive_shared_codebook}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

if [[ "$SCENE" == "ramen" || "$SCENE" == "waldo_kitchen" ]]; then
  calibration=category_percentile
  calibration_low=1
  calibration_high=99
else
  calibration=frame_minmax
  calibration_low=0
  calibration_high=100
fi

base_root="$ROOT/runs/self_trained_gaussian_codebook/${SCENE}_d512_k4096x4096_p32768"
cache="$base_root/cache"
query_bank="$base_root/query_bank_256.npy"
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
gain_tag=${MIN_COSINE_GAIN/./p}
target_tag=${TARGET_COSINE/./p}
run_root="$ROOT/runs/adaptive_shared_codebook/${SCENE}_k${NUM_CODES}_m${MAX_IDS}_g${gain_tag}_t${target_tag}${RUN_TAG}"
initial="$run_root/initial"
trained="$run_root/lovo_kl"

for path in "$cache/manifest.json" "$query_bank" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$run_root"

faiss_args=()
if [[ "$FAISS_GPU" == "1" ]]; then
  faiss_args+=(--faiss_gpu)
fi
codebook_args=()
if [[ -n "$SOURCE_CODEBOOK" ]]; then
  [[ -f "$SOURCE_CODEBOOK" ]] || { echo "Missing reused codebook: $SOURCE_CODEBOOK" >&2; exit 1; }
  codebook_args+=(--codebook "$SOURCE_CODEBOOK")
fi

if [[ ! -f "$initial/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
    --consensus "$cache/consensus.pt" \
    "${codebook_args[@]}" \
    --num_codes "$NUM_CODES" \
    --max_ids "$MAX_IDS" \
    --min_cosine_gain "$MIN_COSINE_GAIN" \
    --target_cosine "$TARGET_COSINE" \
    --train_samples "$TRAIN_SAMPLES" \
    --iterations "$KMEANS_ITERATIONS" \
    --assignment_chunk "$ASSIGNMENT_CHUNK" \
    "${faiss_args[@]}" \
    --output_dir "$initial" \
    > "$LOG_DIR/${SCENE}_k${NUM_CODES}_m${MAX_IDS}${RUN_TAG}_build.log" 2>&1
fi

if [[ ! -f "$trained/artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u train_gaussian_multilevel_codebook.py \
    --cache_dir "$cache" \
    --initial_codebook_dir "$initial" \
    --output "$trained" \
    --iterations "$TRAIN_ITERATIONS" \
    --batch_pixels 4096 \
    --codebook_lr 0.001 \
    --lovo_weight 0.5 \
    --nuisance_rank 4 \
    --query_bank "$query_bank" \
    --query_kl_weight 0.1 \
    --lovo_query_kl_weight 0.1 \
    > "$LOG_DIR/${SCENE}_k${NUM_CODES}_m${MAX_IDS}${RUN_TAG}_train.log" 2>&1
fi

output="$trained/eval_codebook_only_${calibration}_l${calibration_low}_h${calibration_high}"
if [[ ! -f "$output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$run_root" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$trained/artifact" \
    --label_dir "$labels" \
    --rgr_alpha 0 \
    --score_calibration "$calibration" \
    --calibration_low "$calibration_low" \
    --calibration_high "$calibration_high" \
    --thresholds $THRESHOLDS \
    --output "$output" \
    > "$LOG_DIR/${SCENE}_k${NUM_CODES}_m${MAX_IDS}${RUN_TAG}_eval.log" 2>&1
fi

echo "adaptive shared codebook complete: scene=$SCENE K=$NUM_CODES max_ids=$MAX_IDS"
