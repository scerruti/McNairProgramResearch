# McNairResearch

This repository contains the full workflow for testing `Qwen/Qwen3-VL-30B-A3B-Instruct` on the LEGO-Puzzles / LEGOLite spatial reasoning benchmark (height, position, rotation, ordering), from a baseline accuracy check through MoE expert routing analysis and layer/expert ablation experiments.

See `PROJECT_STRUCTURE.md` for the full directory breakdown and a phase-by-phase summary of methods and results.

## Phases

| Phase | Where it ran | What it did |
|-------|-------------|-------------|
| 0 | Fireworks AI API | Baseline accuracy run on LEGOLite to get a reference score |
| 1 | Google Colab | Tracked which MoE experts handle image tokens vs. text tokens |
| 2 | RunPod (A100 80GB) | Two full inference passes logging expert routing across all 48 layers |
| 3 | Local + RunPod (A100 40GB) | Z-score layer scoring, k-medoids expert clustering, ablation experiments, statistical validation |

Phase 3 asked whether specific layers/experts were causing wrong answers, and whether disabling them improves accuracy. Short answer: no. See the Phase 3 Summary in `PROJECT_STRUCTURE.md` for the full statistical writeup (McNemar's test, random-ablation control, holdout validation).

## What is included

- `code/LEGO-Puzzles/`: LEGO-Puzzles benchmark framework and evaluation code (VLMEval-based)
- `data/`: Benchmark results and analysis outputs, organized by phase (`phase0` through `phase3`)
- `scripts/`: Analysis and execution scripts, organized by phase, including RunPod bootstrap scripts
- `docs/`: Sprint plans, the Phase 3 ADR, and the research log

## Requirements

- Python 3.10+
- CUDA-enabled GPU if running the Qwen3 model (Phases 1-3 inference steps)
- `pip` for package installation

## Running the Phase 3 ablation experiment on RunPod

The Phase 3 ablation and holdout runs need a GPU. Upload the repo (or just `scripts/phase3/`) to a RunPod pod, then:

```bash
cd scripts/phase3/step5/runpod
./runpod_run.sh
```

This installs dependencies, downloads `LEGO.tsv`, extracts `lego_images/` from the TSV's embedded base64 images, downloads the Qwen3-VL-30B model if not already cached, and runs the ablation experiment (`ablation_run.py`).

Steps 1-4 of Phase 3 (data split, layer scoring, fingerprinting, clustering) are local, CPU-only Python scripts under `scripts/phase3/step1` through `step4`. Step 6 (`scripts/phase3/step6/analyze_results.py`, `holdout_run.py`) computes the statistical analysis and holdout validation.

## Notes

- `data/phase2/runpod_second/results.json` is the primary source data for Phase 3 (per-question expert routing + correctness for all 400 questions)
- `docs/LEGO_Research_Log.md` documents experiment updates and project progress
- `docs/phase3_adr.md` explains the rationale behind each Phase 3 methodology decision

## License

This project is released under the MIT License. See the `LICENSE` file for full details.

## Contact

For more details, see `docs/LEGO_Research_Log.md`, `docs/phase3_adr.md`, or `PROJECT_STRUCTURE.md`.
