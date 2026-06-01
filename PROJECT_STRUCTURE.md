# McNairResearch Project Structure

## About
This repository captures the LEGO-Puzzles spatial reasoning benchmark workflow, including RunPod phase 2 results for the Qwen3-VL-30B model, mixture-of-experts activation analysis, and category-level evaluation for height, rotation, position, and ordering. The current update reflects the second RunPod Qwen3 experiment and the associated spatial expert routing analysis.

## Directory Organization

```
McNairResearch/
├── code/                          # Main codebase
│   └── LEGO-Puzzles/             # LEGO Puzzles benchmark framework
│       ├── vlmeval/              # VLM evaluation code
│       ├── scripts/              # Utility scripts
│       ├── images/               # Dataset images
│       ├── outputs/              # Model outputs by provider
│       └── requirements.txt
│
├── data/                          # All data files, organized by phase
│   ├── phase1/                   # Phase 1 - early LEGO benchmark runs
│   │   ├── runs/                 # Raw execution outputs (timestamped)
│   │   │   └── lego_2026-02-*/   # program_generator, api_generator, program_execution, signature_generator
│   │   └── analysis/             # Processed MoE expert analysis outputs
│   │       ├── expert_success_rates.csv
│   │       ├── heatmap_*.csv     # Heatmap analysis (height, rotation, position, ordering)
│   │       ├── report.pdf
│   │       ├── results.json
│   │       └── spatial_expert_leaderboard.csv
│   └── phase2/                   # Phase 2 - RunPod Qwen3-VL-30B experiments
│       ├── runpod_first/         # First RunPod run analysis outputs
│       │   ├── expert_success_rates.csv
│       │   ├── heatmap_*.csv
│       │   ├── report.pdf
│       │   ├── results.json
│       │   └── spatial_expert_leaderboard.csv
│       └── runpod_second/        # Second RunPod run analysis outputs
│           ├── expert_success_rates.csv
│           ├── heatmap_*.csv
│           ├── report.pdf
│           ├── results.json
│           └── spatial_expert_leaderboard.csv
│
├── scripts/                       # Analysis & utility scripts
│   ├── phase1/                      # Phase 1 analysis scripts
│   │   └── lego_moe_expert_analysis.py    # MoE expert activation analysis (Colab)
│   ├── phase2/                      # Phase 2 RunPod scripts
│   │   ├── lego_moe_expert_analysis.py    # Wrapper - delegates to phase1 script
│   │   ├── lego_moe_expert_analysis.ipynb # Analysis notebook
│   │   ├── lego_moe_expert_analysis.json  # Shared output data
│   │   ├── lego_moe_expert_analysis.log   # Early wrapper run log
│   │   ├── runpod_first/                  # First RunPod run
│   │   │   ├── lego_lite_moe_analysis.py  # MoE analysis script
│   │   │   ├── generate_report.py         # PDF report generator
│   │   │   └── lego_lite_run.log          # Execution log
│   │   └── runpod_second/                 # Second RunPod run
│   │       ├── lego_lite_moe_analysis.py  # MoE analysis script (offline mode)
│   │       ├── generate_report.py         # PDF report generator (w/ run comparison)
│   │       └── run.log                    # Execution log
│   ├── method_diagram.py              # Diagram generation
│   └── runpod_run.sh                  # RunPod execution script
│
├── notebooks/                     # Jupyter notebooks
│   └── lego_moe_expert_analysis.ipynb  # MOE analysis notebook
│
├── reports/                       # Generated reports
│
├── docs/                          # Documentation
│   ├── LEGO_Research_Log.md
│   └── Notes on proposal intro drafts.pages
│
└── media/                         # Images, diagrams, media files
    └── method_diagram.png
```

## Key Components

### code/
- **LEGO-Puzzles**: Main benchmark framework for spatial reasoning on LEGO puzzles
- Contains benchmark output and analysis code for the current Qwen3-VL RunPod experiment

### data/
- **phase1/runs/**: Raw timestamped execution outputs from early LEGO benchmark runs
- **phase1/analysis/**: Processed MoE expert analysis - heatmaps, leaderboards, report
- **phase2/runpod_first/** and **phase2/runpod_second/**: Analysis outputs from RunPod Qwen3-VL-30B experiments

### scripts/
- **phase1/**: MoE expert activation analysis, designed to run on Google Colab
- **phase2/**: RunPod-specific scripts organized by run; `lego_moe_expert_analysis.py` is a thin wrapper delegating to the phase1 script; `runpod_first/` and `runpod_second/` each contain their own analysis and report generation scripts

### reports/
- Generated PDF reports summarizing findings and analysis

## Notes
- All paths have been reorganized for better discoverability and maintainability
- Original functionality remains unchanged; only file locations have been updated
