# Phase 3 Sprint Plan
## Finding and Ablating Bad Layers via K-Means Clustering

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-05 |
| **Phase** | 3 |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Identify layers that correlate most strongly with wrong answers, cluster questions by those bad-layer binary expert activation patterns, disable the worst layers, and re-run to see if accuracy improves |

---

## Background

Phases 1 and 2 gave us per-question expert activation data across all 48 layers, plus ground-truth correct/incorrect labels. The key insight we want to exploit: if a particular layer is consistently heavily activated on questions the model gets *wrong*, that layer may be actively hurting performance: routing to the wrong experts, introducing noise, or overriding good earlier-layer reasoning.

The plan is three stages:

1. **Score each layer by how bad it is** - compute a "wrongness score" per layer based on how strongly its activation correlates with incorrect answers.
2. **Cluster questions by their bad-layer expert patterns** - for each of the top-K bad layers, represent each question as a binary vector of which experts fired (1) vs did not (0). Run K-means on these vectors so clusters represent distinct "failure modes."
3. **Ablate the worst layers and re-run** - zero out the top-K worst layers during inference and measure whether accuracy improves.

We use **200 questions** for the test run and hold out the other 200 to validate any findings.

**Source data:** `data/phase2/runpod_second/results.json` (400 questions, all valid)

**Categories:** height, rotation, ordering, position

---

## Sprint Tasks

### Step 1 - Verify and Split the Data

- [x] Confirm Phase 2 Run 2 has all 400 questions with valid expert activation data
- [ ] Confirm each record has both `expert_activations` (per-layer, per-expert) and a correct/incorrect label
- [ ] Split the 400 questions into two halves of 200 each - stratify by category so each half has ~50 per category
- [ ] Save `test_200.json` and `holdout_200.json` - all clustering and ablation work runs on the test half first

---

### Step 2 - Score Layers by "Badness"

Using the 200 test questions and the same binary expert data we use in Step 3, compute a single wrongness score per layer:

For each expert in a layer, compute its **wrong-answer rate**: the fraction of times it fired where the answer was wrong.

```
wrong_rate(expert j, layer i) = count(fired AND wrong) / count(fired)
```

Score the layer by averaging this rate across all its experts:

```
layer_score(i) = mean over j of wrong_rate(expert j, layer i)
```

A score above 0.5 means that layer's experts, on average, fire more often on wrong answers than correct ones. Rank all 48 layers by this score. The top layers are the **bad layers**.

- [ ] Write a script to compute per-expert wrong-answer rates and average them per layer
- [ ] Output a ranked table: layer index, layer score, number of experts with rate > 0.5
- [ ] Plot a bar chart of all 48 layer scores so the worst layers are visually obvious
- [ ] Select the single worst layer for the first test run, then compare K=3, K=5

---

### Step 3 - Build the Binary Expert Activation Feature Matrix

For each question, record which experts fired (1) vs did not (0) in each bad layer. This gives a binary vector that captures the exact routing pathway the model took, not just how active it was.

**Dimensions:**
- Each bad layer has N = 200 experts
- One bad layer = **200-dim binary vector** per question
- K bad layers concatenated = **K * 200 dims** per question
- Start with K=1 (the single worst layer) for the first test run - 200 dims is enough for K-means to find structure with 200 questions

**Feature matrix shape:** `(200 questions, K * 200 experts)`

- [ ] For each of the 200 test questions, extract the binary expert activation vector for the worst layer (K=1)
- [ ] Verify each row sums to the model's top-K routing count (should be constant - a sanity check)
- [ ] If K=1 clustering is weak, concatenate the next worst layer and re-run (K=2 gives 400 dims)
- [ ] Save the feature matrix for use on RunPod

---

### Step 4 - Run K-Means on Binary Expert Profiles

- [ ] Run K-means with K=2, 4, 6, 8 clusters on the binary feature matrix
- [ ] Use elbow plot + silhouette scores to find the natural number of "failure modes"
- [ ] For each cluster, record:
  - Accuracy (what fraction were answered correctly)
  - Category mix (which question types dominate)
  - Which experts in which bad layers are most consistently activated in that cluster
- [ ] A cluster that is almost entirely wrong answers = a clean failure mode driven by specific expert routing in specific layers

---

### Step 5 - Set Up RunPod

K-means does not need a GPU. Use a CPU instance.

- [ ] Start a RunPod instance
- [ ] Upload the feature matrix and all scripts
- [ ] Confirm `scikit-learn`, `matplotlib`, `numpy`, `pandas` are installed

---

### Step 6 - Ablation Run (Turn Off the Worst Layers)

This is the key experiment: disable the top-K worst layers during inference and re-run on the 200 test questions.

**How to ablate a layer in a MoE model:**
- Zero out the MLP/expert output for that layer (pass residual stream through unchanged)
- OR set the router logits for that layer to uniform so no single expert gets selected
- OR skip the layer entirely (residual passthrough)

- [ ] Implement layer ablation in the inference script (zero-out approach is simplest)
- [ ] Run ablation for K=1 (worst single layer), K=3, K=5 bad layers disabled
- [ ] Record accuracy per category for each ablation level
- [ ] Compare against baseline (all layers active, Phase 2 results)

---

### Step 7 - Analyze Results

- [ ] Plot accuracy vs number of ablated layers - does it go up, plateau, or drop?
- [ ] Check if ablating bad layers helps more for specific categories (e.g., rotation gets better, height stays same)
- [ ] Look at which clusters from Step 4 shrink or disappear after ablation (fewer wrong-answer clusters = good)
- [ ] Plot the clusters before and after ablation on a 2D PCA chart, colored by correct/incorrect
- [ ] Validate the best ablation config on the holdout 200 to confirm the finding generalizes
- [ ] Write up the main finding: which layers hurt the model and by how much?

---

## Open Questions

> **Causation vs correlation:** A layer being active on wrong answers might mean it's causing the errors, or it might just mean hard questions activate it more. The ablation run in Step 6 is the test - if accuracy goes up when the layer is off, it's causal.

> **How many layers to ablate:** Ablating too many layers will hurt accuracy regardless. Start conservative (K=1, 3, 5) and stop if accuracy starts dropping below baseline.

> **Layer zeroing vs routing change:** Zeroing the MLP output is the cleanest ablation. Changing router logits risks side effects. Try zeroing first.

> **Sparse binary vectors:** With ~200 experts per layer but only top-K routing active per question, each row of the feature matrix is very sparse (mostly 0s). K-means handles this fine but cosine distance may outperform Euclidean - worth comparing both.
