#!/usr/bin/env bash
set -euo pipefail

# Run this on a RunPod pod after uploading files to /workspace.
# Assumes: PyTorch image with CUDA already installed (runpod/pytorch:2.4.0-py3.11-cuda12.4.1)
#
# Upload these files before running:
#   /workspace/phase3_data/test_200.json
#   /workspace/phase3_data/step4_meta.json
#   /workspace/ablation_run.py
#
# Steps this script runs:
#   1. Install Python dependencies
#   2. Download LEGO.tsv
#   3. Extract images from TSV base64 data
#   4. Download Qwen3-VL-30B model (skip if already cached)
#   5. Run ablation experiment (Parts A, B, C)

WORKSPACE=/workspace
LEGO_DATA_DIR=$WORKSPACE/lego_data
LEGO_IMAGES_DIR=$WORKSPACE/lego_images
MODEL_DIR=$WORKSPACE/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct
PHASE3_DATA_DIR=$WORKSPACE/phase3_data
RESULTS_DIR=$WORKSPACE/phase3_results

mkdir -p "$LEGO_DATA_DIR" "$LEGO_IMAGES_DIR" "$PHASE3_DATA_DIR" "$RESULTS_DIR"

echo "=== Step 1: Installing dependencies ==="
pip install -q transformers accelerate bitsandbytes pandas tqdm pillow huggingface_hub safetensors qwen-vl-utils

echo "=== Step 2: Downloading LEGO.tsv ==="
if [ ! -f "$LEGO_DATA_DIR/LEGO.tsv" ]; then
    wget -q "https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv" -O "$LEGO_DATA_DIR/LEGO.tsv"
    echo "Downloaded LEGO.tsv"
else
    echo "LEGO.tsv already exists, skipping."
fi

echo "=== Step 3: Extracting images from TSV ==="
python3 - <<'EOF'
import pandas as pd
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

df = pd.read_csv('/workspace/lego_data/LEGO.tsv', sep='\t')
out = Path('/workspace/lego_images')
out.mkdir(exist_ok=True)

existing = set(p.stem for p in out.glob('*.png'))
extracted = 0

for _, row in df.iterrows():
    idx = str(row['index'])
    if idx in existing:
        continue
    img_col = row.get('image', '')
    if isinstance(img_col, str) and len(img_col) > 100:
        try:
            img_data = base64.b64decode(img_col)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img.save(out / f"{idx}.png")
            extracted += 1
        except Exception as e:
            print(f"Failed to extract image {idx}: {e}")

print(f"Extracted {extracted} images ({len(list(out.glob('*.png')))} total)")
EOF

echo "=== Step 4: Downloading Qwen3-VL-30B model ==="
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Model already cached at $MODEL_DIR, skipping download."
else
    echo "Downloading model (this takes 30-45 minutes)..."
    python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
    local_dir="/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct"
)
print("Model download complete.")
EOF
fi

echo "=== Step 5: Running ablation experiment ==="
python3 /workspace/ablation_run.py

echo "=== Done. Results saved to $RESULTS_DIR/ablation_results.json ==="
