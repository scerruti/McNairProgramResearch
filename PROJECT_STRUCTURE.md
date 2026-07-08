# McNairResearch Project Structure

## About

This repository contains the full workflow for testing Qwen3-VL-30B-A3B-Instruct on the LEGOLite spatial reasoning benchmark. The project is organized into four phases: a Fireworks AI baseline (Phase 0), Google Colab image vs text token routing analysis (Phase 1), RunPod expert activation analysis (Phase 2), and layer/expert ablation to find internal components responsible for spatial reasoning errors (Phase 3).

---

## Directory Layout

```
McNairResearch/
│
├── code/                               # Benchmark framework and model outputs
│   └── LEGO-Puzzles/                  # LEGO-Puzzles evaluation framework (VLMEval-based)
│       ├── vlmeval/                   # Core evaluation library
│       ├── images/                    # Sample and result images
│       └── outputs/                   # Raw model outputs, organized by model
│           ├── Fireworks_Qwen3-VL-30B/
│           ├── GeminiFlash2-0/
│           ├── GPT4o_MINI/
│           └── GPT4o/
│
├── data/                              # All processed results and analysis outputs
│   ├── phase0/
│   │   ├── phase0_model_comparison.csv
│   │   └── phase0_baseline_results.csv
│   ├── phase1/
│   │   ├── runs/                      # VADAR program execution outputs
│   │   └── analysis/                  # MoE expert analysis (results.json, heatmaps, report.pdf)
│   ├── phase2/
│   │   ├── runpod_first/              # Run 1 - 24.25% overall accuracy
│   │   └── runpod_second/             # Run 2 - 25.25% accuracy (primary Phase 3 source data)
│   └── phase3/
│       ├── step1/
│       │   ├── test_200.json          # 200 test questions (stratified split)
│       │   ├── holdout_200.json       # 200 holdout questions (locked until Step 6)
│       │   └── step1_meta.json
│       ├── step2/
│       │   ├── binarized_routing.json # Top-8 expert per question per layer (binary)
│       │   ├── layer_scores.json      # Z-score per layer
│       │   ├── layer_scores.png       # Bar chart of all 48 layer z-scores
│       │   └── step2_meta.json        # Base error rate, top 5 worst layers
│       ├── step3/
│       │   ├── fingerprint_layer_*.json  # 128x200 binary expert fingerprint matrices
│       │   └── step3_meta.json
│       ├── step4/
│       │   ├── cluster_results_layer_*.json
│       │   ├── silhouette_layer_*.png    # Silhouette score curves per layer
│       │   ├── mds_layer_*.png           # MDS cluster scatter plots per layer
│       │   └── step4_meta.json           # Bad expert indices per layer (z-score ranked)
│       ├── step5/
│       │   └── ablation_results.json     # All 40 ablation run results
│       └── step6/
│           ├── step6_meta.json           # McNemar stats, CIs, z-scores vs random
│           ├── holdout_results.json      # Holdout 200 validation results
│           └── plots/
│               ├── accuracy_vs_layers.png
│               └── category_breakdown.png
│
├── scripts/                           # All analysis and execution scripts
│   ├── phase1/
│   │   ├── lego_moe_expert_analysis.py
│   │   └── lego_moe_expert_analysis.ipynb
│   ├── phase2/
│   │   ├── runpod_first/
│   │   └── runpod_second/
│   ├── method_diagram.py
│   └── phase3/
│       ├── step1/                     # Data split
│       ├── step2/                     # Z-score layer scoring
│       ├── step3/                     # Expert fingerprint matrix
│       ├── step4/                     # K-medoids clustering (Jaccard distance)
│       ├── step5/
│       │   ├── ablation_run.py        # 40-run ablation experiment (RunPod)
│       │   └── runpod/
│       │       ├── runpod_run.sh      # End-to-end RunPod bootstrap script
│       │       ├── requirements.txt
│       │       └── validate_pipeline.py
│       └── step6/
│           ├── analyze_results.py     # McNemar, CIs, z-scores, plots
│           └── holdout_run.py         # Holdout 200 validation (RunPod)
│
└── docs/
    ├── phase0_sprint_plan.md
    ├── phase1_sprint_plan.md
    ├── phase2_sprint_plan.md
    ├── phase3_sprint_plan.md
    └── phase3_adr.md                  # Architecture decision record for Phase 3 methodology
```

---

## Phase Summary

| Phase | Where | What we did | Status |
|-------|-------|-------------|--------|
| 0 | Fireworks AI API | Ran Qwen3-VL-30B on LEGOLite via API to get a baseline accuracy score | Done |
| 1 | Google Colab | Ran the model locally and tracked which experts handle image tokens vs text tokens | Done |
| 2 | RunPod (A100 80GB) | Two full inference passes with expert routing hooks across all 48 layers | Done |
| 3 | Local + RunPod (A100 40GB) | Z-score layer scoring, k-medoids clustering, ablation experiments, statistical analysis | Done |

---

## Phase 3 Summary

Phase 3 asked: are there specific layers or experts that cause the model to get spatial reasoning questions wrong, and can disabling them improve accuracy?

**Step 1:** Split 400 questions into 200 test / 200 holdout (stratified by category). Base error rate: 76.5%.

**Step 2:** Binarized routing data (top-8 experts per question per layer). Scored all 48 layers using z-scores. Worst layers by z-score: 41, 42, 43, 19, 7. Signal was weak (max z-score 0.055).

**Step 3:** Built 128x200 binary expert fingerprint matrices for each of the 5 worst layers.

**Step 4:** K-medoids clustering with Jaccard distance on each layer's experts. All silhouette scores below 0.25 (weak clusters). Identified bad expert lists per layer.

**Step 5:** 40 ablation runs on RunPod: baseline, 3 ablation modes on worst layer (Part A), moe_only and expert mode scaled to 1/3/5 layers (Part B), 30 random control trials (Part C). Best result: moe_only on 3 layers, 23% to 26.5%.

**Step 6:** McNemar's test (p=0.25, not significant). Z-score vs random controls (z=1.33, not significant). Holdout 200 validation: baseline 29%, ablation 27% (-2%). Conclusion: errors are distributed broadly across layers, not concentrated in specific components.

---

## Accuracy Summary (LEGOLite, 4 categories)

| Model | Overall | height | position | rotation | ordering |
|-------|---------|--------|----------|----------|---------|
| GeminiFlash2-0 | 44.25% | 35% | 47% | 49% | 46% |
| Qwen3-VL-30B baseline (Phase 3 test set) | 23.0% | - | - | - | - |
| Qwen3-VL-30B Run 2 (Phase 2) | 25.25% | 32% | 25% | 24% | 20% |
| Qwen3-VL-30B Run 1 (Phase 2) | 24.25% | 32% | 22% | 23% | 20% |
| GPT-4o Mini | 13.75% | 29% | 10% | 12% | 4% |

---

## Key Data Files

| File | What it contains |
|------|-----------------|
| `data/phase2/runpod_second/results.json` | Source data for Phase 3 - per-question routing weights and predictions for all 400 questions |
| `data/phase3/step5/ablation_results.json` | All 40 ablation run results with per-question predictions |
| `data/phase3/step6/step6_meta.json` | Full statistical analysis: McNemar tables, CIs, z-scores |
| `data/phase3/step6/holdout_results.json` | Holdout 200 validation results |
| `data/phase3/step4/step4_meta.json` | Bad expert indices per layer, z-score ranked |
