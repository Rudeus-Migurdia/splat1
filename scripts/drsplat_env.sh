#!/usr/bin/env bash
# Shared-server isolation defaults for Dr.Splat.

DRSPLAT_CACHE_DIR="${DRSPLAT_CACHE_DIR:-${PWD}/.cache/drsplat}"

mkdir -p "${DRSPLAT_CACHE_DIR}"/{pip,torch,huggingface,xdg,nv}

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DRSPLAT_CACHE_DIR}/pip}"
export TORCH_HOME="${TORCH_HOME:-${DRSPLAT_CACHE_DIR}/torch}"
export HF_HOME="${HF_HOME:-${DRSPLAT_CACHE_DIR}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DRSPLAT_CACHE_DIR}/xdg}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${DRSPLAT_CACHE_DIR}/nv}"
export WANDB_MODE="${WANDB_MODE:-offline}"
