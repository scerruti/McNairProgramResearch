# Phase 2 Sprint Plan
## Running the Model Locally on RunPod to Track Expert Activations

| Field | Detail |
|-------|--------|
| **Date** | 2026-05-28 |
| **Phase** | 2 |
| **Model** | Qwen3-VL-30B-A3B-Instruct (run locally with 4-bit compression) |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Run Qwen3-VL-30B on a cloud GPU server and record which internal experts the model activates for each question, then look for patterns by question category |

---

## Background

Phase 0 gave us accuracy scores via the Fireworks AI API. Phase 1 ran on Google Colab and tracked which experts handled image vs text tokens. Phase 2 moves to a more powerful cloud GPU (RunPod A100) for two full runs with better settings, producing the expert activation dataset that Phase 3 will use for clustering.

The model has 128 internal experts spread across 48 processing layers. For each question, we attach monitoring hooks to the model's routing gates - these record which 8 experts get chosen at each of the 48 layers for every token. After running all 400 questions, we have a detailed map of how the model distributed its processing.

**GPU:** NVIDIA A100 80GB (RunPod)
**Compression:** 4-bit NF4 quantization - shrinks the model's memory usage so it fits on one GPU without losing much accuracy
**Router:** `Qwen3VLMoeTextTopKRouter` - the internal component that decides which experts to use

---

## Sprint Tasks

### Step 1 - Set Up RunPod
- [x] Start an A100 80GB instance on RunPod
- [x] Create a Python environment and install all required packages
- [x] Move the HuggingFace model cache to `/workspace` (the main drive was full)
- [x] Download and save the Qwen3-VL-30B model weights

### Step 2 - Add Expert Monitoring
- [x] Find the right place in the model code to attach monitoring hooks (the routing gate)
- [x] Write hooks that record which experts are selected during each forward pass
- [x] Attach hooks to all 48 processing layers
- [x] Test on one question to confirm the hooks are working

### Step 3 - Prepare the LEGOLite Questions
- [x] Download the LEGO question dataset
- [x] Decode the images from the dataset format and save them to disk
- [x] Format each question correctly for the model

### Step 4 - Run 1: First Full Inference Pass
- [x] Run all 400 questions through the model with monitoring active
- [x] Collect expert activation counts per question across all 48 layers
- [x] Save predictions, correct answers, and expert data to a results file

### Step 5 - Run 2: Improved Settings
- [x] Enable 4-bit compression to make the model more stable
- [x] Turn off unnecessary network calls during inference
- [x] Re-run all 400 questions and save to a separate results folder

### Step 6 - Analyze Expert Patterns
- [x] Create heatmaps showing which experts activate most for each question category
- [x] Calculate which experts correlate with correct answers
- [x] Build a leaderboard of the most useful experts for spatial reasoning
- [x] Export summary files: `expert_success_rates.csv`, `spatial_expert_leaderboard.csv`

---

## Results

| Model | Overall | height | position | rotation | ordering |
|-------|---------|--------|----------|----------|---------|
| GeminiFlash2-0 | 44.25% | 35% | 47% | 49% | 46% |
| Qwen3-VL-30B Run 2 (ours) | 25.25% | 32% | 25% | 24% | 20% |
| Qwen3-VL-30B Run 1 (ours) | 24.25% | 32% | 22% | 23% | 20% |
| GPT-4o Mini | 13.75% | 29% | 10% | 12% | 4% |

Run 2 improved on position (+3%) and rotation (+1%) over Run 1. Our model beats GPT-4o Mini on 3 of 4 categories but still trails Gemini - the biggest gap is ordering (20% vs 46%).

---

## Outputs

| File | What it contains |
|------|-----------------|
| `data/phase2/runpod_second/results.json` | Per-question expert activations, predictions, and accuracy |
| `data/phase2/runpod_second/expert_success_rates.csv` | How often each expert activates on correct vs incorrect answers |
| `data/phase2/runpod_second/spatial_expert_leaderboard.csv` | Top experts ranked by how much they contribute to correct answers |

---

## Key Takeaways

- Both runs completed all 400 questions with no errors; Run 2 is the cleaner dataset and feeds into Phase 3
- Run 2's small accuracy gains suggest 4-bit compression helps stabilize the model on spatial questions
- Some experts consistently correlate with correct answers - early evidence that routing patterns encode spatial reasoning quality
- The monitoring hooks worked cleanly across all 48 layers with no slowdown
