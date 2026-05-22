# CSE 199 Project Structure

## Directory Organization

```
CSE 199/
├── code/                          # Main codebase
│   └── LEGO-Puzzles/             # LEGO Puzzles benchmark framework
│       ├── vlmeval/              # VLM evaluation code
│       ├── scripts/              # Utility scripts
│       ├── images/               # Dataset images
│       ├── outputs/              # Model outputs by provider
│       └── requirements.txt
│
├── data/                          # All data files
│   ├── analysis/                 # RunPod analysis results
│   │   ├── expert_results.pdf
│   │   ├── expert_success_rates.csv
│   │   ├── heatmap_*.csv         # Heatmap analysis (height, rotation, position, ordering)
│   │   ├── spatial_expert_leaderboard.csv
│   │   └── report.pdf
│   │
│   └── results/                  # Execution & test results
│       └── lego_2026-02-*/       # Timestamped result directories
│
├── scripts/                       # Analysis & utility scripts
│   ├── lego_moe_expert_analysis.py    # MOE expert analysis
│   ├── generate_report.py             # Report generation
│   ├── method_diagram.py              # Diagram generation
│   └── runpod_run.sh                  # RunPod execution script
│
├── notebooks/                     # Jupyter notebooks
│   └── lego_moe_expert_analysis.ipynb  # MOE analysis notebook
│
├── reports/                       # Generated reports
│   └── VADAR_LEGO_Report.pdf
│
├── docs/                          # Documentation
│   ├── LEGO_Research_Log.md
│   ├── vadar_on_lego_via_aws_967ad135.plan.md
│   └── Notes on proposal intro drafts.pages
│
└── media/                         # Images, diagrams, media files
    └── method_diagram.png
```

## Key Components

### code/
- **LEGO-Puzzles**: Main benchmark framework for spatial reasoning on LEGO puzzles
- Contains model outputs from: GPT-4o, GPT-4o Mini, Gemini Flash 2.0, Qwen2.5-VL, Qwen3-VL, SmolVLM, IDEFICS

### data/
- **analysis/**: Results from RunPod phase 2 analysis including expert performance, heatmaps, and leaderboards
- **results/**: Execution results with timestamped directories containing test outputs

### scripts/
- Analysis tools for processing results and generating visualizations
- `lego_moe_expert_analysis.py`: Analyzes mixture-of-experts performance across models

### reports/
- Generated PDF reports summarizing findings and analysis

## Notes
- All paths have been reorganized for better discoverability and maintainability
- Original functionality remains unchanged; only file locations have been updated
