# McNairResearch

This repository contains the LEGO-Puzzles spatial reasoning benchmark workflow, including RunPod Qwen3-VL-30B experiments, MoE expert activation analysis, and category-level evaluation for height, rotation, position, and ordering.

## What is included

- `code/LEGO-Puzzles/`: LEGO-Puzzles benchmark framework and evaluation code
- `data/`: Benchmark results and analysis outputs, organized by phase
  - `data/phase1/`: Early benchmark runs and analysis
  - `data/phase2/`: RunPod experiment analysis outputs (runpod_first, runpod_second)
- `scripts/`: Analysis and utility scripts for MoE expert analysis and RunPod execution
- `docs/`: Research notes and logs
- `notebooks/`: Analysis notebook for MoE expert activation
- `reports/`: Generated summary reports

## Key workflows

### RunPod experiment

The main RunPod experiment is centered on `scripts/runpod_run.sh`, which sets up a Python virtual environment, installs dependencies, and runs the MoE expert analysis script.

### Spatial expert analysis

The `scripts/phase1/lego_moe_expert_analysis.py` script loads `Qwen/Qwen3-VL-30B-A3B-Instruct`, runs the LEGOLite benchmark subset (height, position, rotation, ordering), and logs which MoE experts activate for each question.

## Requirements

- Python 3.10+
- CUDA-enabled GPU if running the Qwen3 model locally
- `pip` for package installation

## Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
python -m pip install transformers accelerate pillow pandas tqdm safetensors
python -m pip install qwen-vl-utils || true
```

> Adjust the PyTorch install URL if running on a different CUDA version.

## Running the analysis

Use the `runpod_run.sh` helper to run the experiment on a GPU-enabled machine:

```bash
cd scripts
./runpod_run.sh
```

This will create and activate a virtual environment, install dependencies, and run the phase 2 MoE analysis.

To run the analysis script directly:

```bash
cd scripts/phase1
source ../../venv/bin/activate
python lego_moe_expert_analysis.py
```

## Repository structure

See `PROJECT_STRUCTURE.md` for the full directory breakdown.

## Notes

- `data/phase2/runpod_second/` contains the second RunPod experiment outputs
- `docs/LEGO_Research_Log.md` documents experiment updates and project progress
- `scripts/phase1/lego_moe_expert_analysis.py` is the primary script for tracing Qwen3 MoE expert activations

## License

This project is released under the MIT License. See the `LICENSE` file for full details.

## Contact

For more details, see `docs/LEGO_Research_Log.md`, the notebook in `notebooks/`, or `PROJECT_STRUCTURE.md`.
