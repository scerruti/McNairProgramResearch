# Phase 1 — Repeat Plan: Image vs Text Expert Routing (Qwen3-VL-30B)

Purpose
- Reproduce Sprint 1 (Phase 1): run Qwen3-VL-30B locally on the LEGOLite dataset to measure which MoE experts activate for image tokens vs text tokens, then produce the same summary outputs and JSON used in the project.

Quick summary
- Script to run: `scripts/phase1/lego_moe_expert_analysis.py` (notebook `lego_moe_expert_analysis.ipynb` also provided).
- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Dataset: `LEGO.tsv` (script downloads if missing)
- Expected output: JSON summary with per-question expert frequencies, per-layer expert data, and category-level top experts. Canonical location used in repo: `data/phase1/analysis/results.json`.

Prerequisites
- Hardware: single GPU with large memory (recommended 80–96 GB VRAM). Colab (NVIDIA G4 Blackwell 96GB) used originally.
- Software: Python 3.10+, packages: transformers, accelerate, torch, pillow, pandas, tqdm, qwen-vl-utils. Optional: flash-attn for speed.
- Disk: tens of GB for model cache.

Quick commands (from a Colab / Ubuntu session)
```bash
# clone
git clone https://github.com/scerruti/McNairProgramResearch.git
cd McNairProgramResearch

# install minimal deps
pip install transformers accelerate torch pillow pandas tqdm qwen-vl-utils
# optional speedups
pip install flash-attn

# run (interactive: open the notebook; headless:)
python scripts/phase1/lego_moe_expert_analysis.py
# after run, copy the output into project layout if desired:
cp /content/lego_moe_expert_analysis.json data/phase1/analysis/results.json
```

Step-by-step reproduction plan
1. Environment check
   - Start a Colab or other GPU instance and confirm GPU: `import torch; torch.cuda.is_available()`.
   - Ensure GPU memory is sufficient. If not, plan fp16/offload or a larger machine.

2. Prepare repository & packages
   - Clone the repo and install packages (see Quick commands).
   - Optionally test a small Python snippet to confirm the transformers + model can load.

3. Dry run with a small subset
   - Edit the script to loop only the first N questions (e.g., N=5) to validate hooks and outputs quickly.
   - Confirm model loads, hooks register successfully, and a small JSON is produced.

4. Full run
   - Run the full script (400 questions). The script downloads LEGO.tsv, extracts images to `IMG_DIR`, loads the model, registers router hooks, runs inference, and writes `/content/lego_moe_expert_analysis.json`.
   - Monitor runtime and VRAM; script calls `torch.cuda.empty_cache()` between questions.

5. Verify outputs
   - Confirm JSON contains:
     - metadata (model, categories, num_questions)
     - `category_expert_summary` with `top_15_experts`
     - `per_question_results` with `expert_frequency` and `per_layer_experts`
   - Move/copy JSON to `data/phase1/analysis/results.json` to match repo conventions.

Token-level image vs text separation (important)
- The sprint aims to compare image-token vs text-token expert activations. The provided script aggregates expert activations per question but does not explicitly separate image vs text token activations.
- To produce modality-separated counts you must map router outputs back to token positions:
  - Modify the forward hook to record (token_position, expert_indices) rather than flattening; or
  - Use the processor output to identify positions corresponding to image patches vs text tokens (tokenizer output or processor may provide masks or lengths).
- Recommended modification:
  - In the hook, capture both router_indices and their token indices (router often returns indices aligned to token positions). Save them per token, then build two counters per question: `image_expert_freq` and `text_expert_freq`.

Validation checklist (what to check after the run)
- JSON contains 400 per-question entries.
- `category_expert_summary.top_15_experts` exists for each category.
- Overall and per-category accuracy printed by the script match earlier Phase 1 output within reasonable variance.
- If modality separation implemented: produce heatmaps (image vs text) per category and compare top experts.
- Re-run a small random subset to ensure results are reproducible (no major divergence).

Common pitfalls & fixes
- OOM: switch dtype to `torch.float16`, enable model offload, or use a bigger GPU.
- Hook errors: the hook assumes gate output shape `(..., router_indices)`; test the gate output once and adjust indexing if model internals differ.
- Token mapping: if the processor does not expose token-alignment info, create a text-only tokenization and compute token-offsets to identify text token indices.

Optional enhancements (recommended)
- Patch the script to log per-token expert indices and a modality mask so the analysis JSON directly includes `image_expert_frequency` and `text_expert_frequency` per question.
- Produce and save the category heatmaps into `data/phase1/analysis/plots/`.
- Add a small automated test-run script (e.g., `scripts/phase1/run_quick_test.sh`) to validate environment quickly.

Estimated time & resources
- Small test (10 questions): minutes.
- Full run (400 questions): typically 2–6 hours on the recommended GPU (varies by machine and model generation settings).
- Disk: tens of GB used by HuggingFace model cache.

Next steps (choose one)
- I can patch `scripts/phase1/lego_moe_expert_analysis.py` to record per-token expert activations and produce explicit `image_expert_frequency` / `text_expert_frequency` in the JSON.
- I can commit this markdown file into the repository at `docs/phase1_repeat_plan_cerruti.md` or `docs/phase1_repeat_plan/README.md`.
- I can prepare a small 10-question test-run notebook with the modified hook to validate modality mapping.

Contact / notes
- If you want further changes, tell me whether to create a branch or commit to the default branch.
