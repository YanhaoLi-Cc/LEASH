#!/usr/bin/env bash
# LEASH environment setup
# Prerequisites: conda, CUDA 12.4+

set -e

ENV_NAME=${1:-leash}

echo "Creating conda environment: ${ENV_NAME} (Python 3.10)"
conda create -n ${ENV_NAME} python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Installing PyTorch (CUDA 12.4)"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing verl with vLLM support"
cd "$(dirname "$0")/verl"
pip install -e .[vllm]

echo "Installing additional dependencies"
pip install flash-attn==2.7.3 --no-build-isolation
pip install vllm==0.8.4 math-verify==0.6.0

echo "Done. Activate with: conda activate ${ENV_NAME}"
