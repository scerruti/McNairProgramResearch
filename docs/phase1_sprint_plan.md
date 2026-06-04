# Phase 1 Sprint Plan
## Comparing Image vs Text Expert Routing on Google Colab

| Field | Detail |
|-------|--------|
| **Phase** | 1 |
| **Environment** | Google Colab (NVIDIA G4 Blackwell, 96GB VRAM) |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Run the model on Google Colab and track which internal specialists (experts) handle the image parts of a question separately from the text parts |

---

## Background

Phase 0 told us how accurate the model is, but not how it thinks. In Phase 1 we run the model locally on Google Colab so we can look inside it while it works.

Qwen3-VL-30B is a Mixture-of-Experts (MoE) model. Instead of using all of its parameters every time, it has 128 internal specialists called **experts**, and it picks 8 of them for each piece of input. This is what makes it efficient - only a small portion of the model is active at once.

When the model processes a question with an image, two types of input get fed in:
- **Image tokens** - the visual content of the LEGO image, broken into small patches
- **Text tokens** - the question text and answer choices

The model routes each token through different experts. By tracking which experts get chosen for image tokens vs text tokens separately, we can ask: does the model have dedicated visual specialists? Do different spatial question types (height vs rotation) use different experts?

**Source script:** `scripts/phase1/lego_moe_expert_analysis.py`
**Notebook:** `notebooks/lego_moe_expert_analysis.ipynb`

---

## Sprint Tasks

### Step 1 - Set Up Google Colab
- [x] Confirm GPU is available (G4 Blackwell, 96GB VRAM)
- [x] Install required packages: `transformers`, `accelerate`, `torch`, `pillow`, `pandas`, `tqdm`, `qwen-vl-utils`
- [x] Load Qwen3-VL-30B onto the GPU

### Step 2 - Find Image vs Text Token Positions
- [x] Look at how the model tokenizes a sample question to find where image tokens appear in the sequence
- [x] Build a mask that labels each token as either an image token or a text token
- [x] Check that the mask works correctly across different question types

### Step 3 - Track Experts Separately for Image vs Text
- [x] Add monitoring code to the model that records which experts activate for each token
- [x] Keep separate counters for image-token expert activations vs text-token expert activations
- [x] Confirm both counters are filling in correctly on a test question

### Step 4 - Run All 400 Questions
- [x] Run all 400 LEGOLite questions through the model with monitoring active
- [x] Save each question's image-expert and text-expert activation counts separately
- [x] Record whether the model answered each question correctly
- [x] Export results to JSON and download from Colab

### Step 5 - Analyze the Differences
- [x] For each category, find the top 15 most-used experts for image tokens vs text tokens
- [x] Check which experts show up in both lists (general purpose) vs only one (specialized)
- [x] Check if different question types (height, rotation, position, ordering) use different image experts
- [x] Check if questions the model answered correctly used different experts than ones it got wrong

### Step 6 - Summarize Findings
- [x] Plot heatmaps showing image-expert vs text-expert usage side by side for each category
- [x] Identify experts that are heavily used for images across all categories (likely general visual processors)
- [x] Identify experts that only activate for images in specific categories (likely task-specific)
- [x] Write up findings in `docs/LEGO_Research_Log.md`

---

## Key Questions to Answer

| Question | What it tells us |
|----------|-----------------|
| Do image tokens and text tokens use different experts? | Whether the model has separate visual vs language specialists |
| Do image experts differ across height / rotation / position / ordering? | Whether expert routing reflects the type of spatial task |
| Do correct answers show stronger image expert activation? | Whether better visual processing leads to better answers |
| Is ordering different from the other 3 categories in its routing? | Follows up on ordering's 0% score from Phase 0 |

---

## Open Questions

- Results exported to `data/phase1/analysis/results.json` (400 questions, all valid)
