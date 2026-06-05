# Phase 3 Sprint Plan
## Finding and Ablating Bad Layers via K-Means Clustering

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-05 |
| **Phase** | 3 |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Identify layers that correlate most strongly with wrong answers, cluster questions by those bad-layer activation patterns, disable the worst layers, and re-run to see if accuracy improves |

---

## Background

Phases 1 and 2 gave us per-question expert activation data across all 48 layers, plus ground-truth correct/incorrect labels. The key insight we want to exploit: if a particular layer is consistently heavily activated on questions the model gets *wrong*, that layer may be actively hurting performance: routing to the wrong experts, introducing noise, or overriding good earlier-layer reasoning.

The plan is three stages:

1. **Score each layer by how bad it is** - compute a "wrongness score" per layer based on how strongly its activation correlates with incorrect answers.
2. **Cluster questions by their bad-layer profiles** - run K-means using only the worst-N layers as features, so clusters represent "ways of being wrong."
3. **Ablate the worst layers and re-run** - zero out / skip the top-K worst layers during inference and measure whether accuracy goes up.

**Source data:** `data/phase2/runpod_second/results.json` (400 questions, all valid)

**Categories:** height, rotation, ordering, position

---

## Sprint Tasks

### Step 1 - Verify the Data
- [x] Confirm Phase 2 Run 2 has all 400 questions with valid expert activation data
- [ ] Confirm each record has both `expert_activations` (per-layer) and a correct/incorrect label
- [ ] Load and inspect the data - count correct vs incorrect per category

---

### Step 2 - Score Layers by "Badness"

For each of the 48 layers, compute a wrongness score that measures how much that layer's activation level predicts an incorrect answer.

| Method | Description |
|--------|-------------|
| **Mean activation delta** | `mean(activation[wrong]) - mean(activation[correct])` - positive = layer is more active on wrong answers |
| **Point-biserial correlation** | Correlation between layer activation magnitude and the binary wrong/correct label |
| **Wrong-answer weight** | Fraction of total wrong-question activation that falls in this layer |

Rank all 48 layers by these scores. The top layers are the **bad layers**.

- [ ] Write a script to compute all three scores per layer
- [ ] Output a ranked table: layer index, wrongness scores, category breakdown
- [ ] Plot a bar chart of per-layer wrongness scores so the worst layers are visually obvious
- [ ] Choose a cutoff: select the top-K bad layers (start with K=5, K=10, compare)

---

### Step 3 - Build the Bad-Layer Feature Matrix

Rather than using all 48 layers as features (which buries the bad-layer signal), we build a feature matrix using only the bad layers identified in Step 2.

- [ ] For each of the 400 questions, extract activation values for the top-K bad layers only
- [ ] Resulting matrix: 400 rows × K columns
- [ ] Normalize each column (z-score) so no single layer dominates by scale
- [ ] Save the matrix to a file for use on RunPod

---

### Step 4 - Run K-Means on Bad-Layer Profiles

- [ ] Run K-means with K=2, 4, 6, 8 clusters on the bad-layer feature matrix
- [ ] Use elbow plot + silhouette scores to find the natural number of "failure modes"
- [ ] For each cluster, record:
  - Accuracy (what fraction were answered correctly)
  - Category mix (which question types dominate)
  - Which bad layers are most active in that cluster
- [ ] A cluster that is almost entirely wrong answers = a clean failure mode driven by specific bad layers

---

### Step 5 - Set Up RunPod

K-means does not need a GPU. Use a CPU instance.

- [ ] Start a RunPod instance
- [ ] Upload the feature matrix and all scripts
- [ ] Confirm `scikit-learn`, `matplotlib`, `numpy`, `pandas` are installed

---

### Step 6 - Ablation Run (Turn Off the Worst Layers)

This is the key experiment: disable the top-K worst layers during inference and re-run the benchmark.

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
- [ ] Write up the main finding: which layers hurt the model and by how much?

---

## Open Questions

> **Causation vs correlation:** A layer being active on wrong answers might mean it's causing the errors, or it might just mean hard questions activate it more. The ablation run in Step 6 is the test - if accuracy goes up when the layer is off, it's causal.

> **How many layers to ablate:** Ablating too many layers will hurt accuracy regardless. Start conservative (K=1, 3, 5) and stop if accuracy starts dropping below baseline.

> **Layer zeroing vs routing change:** Zeroing the MLP output is the cleanest ablation. Changing router logits risks side effects. Try zeroing first.
