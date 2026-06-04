# McNairResearch Project Structure

## About

This repository contains the full workflow for testing Qwen3-VL-30B on the LEGO-Puzzles spatial reasoning benchmark. The project is organized into 4 phases: a Fireworks AI baseline (Phase 0), Google Colab image vs text token routing analysis (Phase 1), RunPod expert activation analysis (Phase 2), and K-means clustering on layer activation patterns (Phase 3, in progress).

---

## Directory Layout

```
McNairResearch/
│
├── code/                               # Benchmark framework and model outputs
│   ├── LEGO-Puzzles/                  # LEGO-Puzzles evaluation framework (VLMEval-based)
│   │   ├── vlmeval/                   # Core evaluation library
│   │   ├── scripts/                   # Benchmark utility scripts
│   │   ├── images/                    # Sample and result images
│   │   ├── outputs/                   # Raw model outputs, organized by model
│   │   │   ├── Fireworks_Qwen3-VL-30B/   # Phase 0 Fireworks API outputs (LEGOLite)
│   │   │   │   ├── T20260313_G58d6fb3e/  # Early test run
│   │   │   │   └── T20260322_G58d6fb3e/  # Phase 0 full LEGOLite run (3/22/2026)
│   │   │   ├── GeminiFlash2-0/           # Gemini baseline outputs (full benchmark)
│   │   │   ├── GPT4o_MINI/               # GPT-4o Mini baseline outputs (full benchmark)
│   │   │   └── GPT4o/                    # GPT-4o outputs
│   │   ├── run.py
│   │   └── requirements.txt
│
├── data/                               # Processed results and analysis outputs
│   ├── phase0/                        # Phase 0 - Fireworks AI baseline results
│   │   ├── phase0_model_comparison.csv    # Side-by-side accuracy for all models
│   │   ├── phase0_baseline_results.csv    # Per-category results (full 11-category benchmark)
│   │   └── phase0_run_debug_log.txt       # Run and debug log from Phase 0 setup
│   ├── phase1/                        # Early runs and MoE analysis outputs
│   │   ├── runs/                      # VADAR program execution outputs (Feb 2026, pre-phase-0)
│   │   │   └── lego_2026-02-16_*/    # Per-question HTML outputs from program_generator and api_generator
│   │   └── analysis/                  # MoE expert analysis outputs (same run as runpod_first)
│   │       ├── results.json           # Per-question predictions and layer activation data (400 questions)
│   │       ├── expert_success_rates.csv
│   │       ├── spatial_expert_leaderboard.csv
│   │       ├── heatmap_height.csv
│   │       ├── heatmap_position.csv
│   │       ├── heatmap_rotation.csv
│   │       ├── heatmap_ordering.csv
│   │       ├── heatmap_height_minus_rotation.csv
│   │       └── report.pdf
│   └── phase2/                        # RunPod inference runs
│       ├── runpod_first/              # Run 1 - 24.25% overall accuracy
│       │   ├── results.json
│       │   ├── expert_success_rates.csv
│       │   ├── spatial_expert_leaderboard.csv
│       │   ├── heatmap_*.csv
│       │   └── report.pdf
│       └── runpod_second/             # Run 2 - 25.25% overall accuracy (primary dataset for Phase 3)
│           ├── results.json
│           ├── expert_success_rates.csv
│           ├── spatial_expert_leaderboard.csv
│           ├── heatmap_*.csv
│           └── report.pdf
│
├── scripts/                            # Analysis and execution scripts
│   ├── phase1/
│   │   ├── lego_moe_expert_analysis.py    # Google Colab MoE analysis script
│   │   └── lego_moe_expert_analysis.ipynb # Interactive Colab-compatible notebook
│   ├── phase2/
│   │   ├── lego_moe_expert_analysis.py    # Wrapper script
│   │   ├── lego_moe_expert_analysis.ipynb
│   │   ├── lego_moe_expert_analysis.json  # Intermediate output (early run, had image errors)
│   │   ├── lego_moe_expert_analysis.log
│   │   ├── runpod_first/
│   │   │   ├── lego_lite_moe_analysis.py  # Run 1 inference + hook script
│   │   │   ├── generate_report.py
│   │   │   └── lego_lite_run.log
│   │   └── runpod_second/
│   │       ├── lego_lite_moe_analysis.py  # Run 2 inference + hook script (NF4, offline mode)
│   │       ├── generate_report.py
│   │       └── run.log
│   ├── method_diagram.py
│   └── runpod_run.sh                  # RunPod setup and execution script
│
├── docs/                               # Documentation and sprint plans
│   ├── LEGO_Research_Log.md           # Running log of experiments and findings
│   ├── phase0_sprint_plan.md          # Phase 0: Fireworks AI baseline evaluation
│   ├── phase1_sprint_plan.md          # Phase 1: Google Colab image vs text token routing
│   ├── phase2_sprint_plan.md          # Phase 2: RunPod expert activation analysis
│   ├── phase3_sprint_plan.md          # Phase 3: K-means clustering (in progress)
│   └── Notes on proposal intro drafts .pages
│
```

---

## Phase Summary

| Phase | Where | What we did | Status |
|-------|-------|-------------|--------|
| 0 | Fireworks AI API | Ran Qwen3-VL-30B on LEGOLite via API to get a baseline accuracy score | Done |
| 1 | Google Colab | Ran the model locally and tracked which experts handle image tokens vs text tokens separately | Done |
| 2 | RunPod (A100 80GB) | Ran two full inference passes with expert routing hooks across all 48 layers; analyzed per-category routing patterns | Done |
| 3 | RunPod | K-means clustering on layer activation profiles to find natural question groupings | In Progress |

---

## Key Data Files

| File | What it contains |
|------|-----------------|
| `data/phase2/runpod_second/results.json` | Primary dataset for Phase 3 - per-question layer activation data, predictions, and accuracy for all 400 LEGOLite questions |
| `data/phase0/phase0_baseline_results.csv` | Full 11-category benchmark accuracy for all models (used for baseline comparisons) |
| `data/phase0/phase0_model_comparison.csv` | Side-by-side accuracy table across all tested models |
| `data/phase2/runpod_second/spatial_expert_leaderboard.csv` | Top experts ranked by correlation with correct answers |

---

## Accuracy Summary (LEGOLite, 4 categories)

| Model | Overall | height | position | rotation | ordering |
|-------|---------|--------|----------|----------|---------|
| GeminiFlash2-0 | 44.25% | 35% | 47% | 49% | 46% |
| Qwen3-VL-30B Run 2 (ours) | 25.25% | 32% | 25% | 24% | 20% |
| Qwen3-VL-30B Run 1 (ours) | 24.25% | 32% | 22% | 23% | 20% |
| GPT-4o Mini | 13.75% | 29% | 10% | 12% | 4% |
