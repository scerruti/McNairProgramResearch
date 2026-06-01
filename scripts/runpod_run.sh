#!/usr/bin/env bash
set -euo pipefail

# Run this on a RunPod Linux instance with NVIDIA GPU support.
# Recommended: Ubuntu 22.04 + A100/RTX 4090/A5000 with CUDA drivers installed.

cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch for CUDA. Adjust cu121/cu122 if needed based on RunPod image.
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu122
python -m pip install -q transformers>=4.51.0 accelerate pillow pandas tqdm

# Optional helper packages
python -m pip install -q safetensors
python -m pip install -q qwen-vl-utils || true

python3 phase2/lego_moe_expert_analysis.py
