# Phase 3 Sprint Plan
## Finding and Turning Off Bad Layers

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-05 (updated 2026-06-21) |
| **Phase** | 3 |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Find which internal layers contain experts that cause wrong answers, group those experts by their activation patterns, then turn off the bad experts and re-run to see if accuracy improves |
| **ADR** | See `docs/phase3_adr.md` for full rationale behind each methodology decision |

---

## What We Are Doing and Why

The model has 48 layers. Each layer has 128 experts - small sub-networks the router picks from. For every question, the router selects exactly 8 of those 128 experts to do the work in each layer.

From Phase 2 we have a log of routing weights for all 128 experts across all 48 layers for each of the 400 questions, plus whether each question was answered correctly.

The question we are asking:

> Are there specific experts in specific layers that keep getting selected on questions the model gets wrong? If so, those experts (and their layers) may be hurting performance. Turn off those experts and see if accuracy improves.

**The three-step logic:**
1. Score all 48 layers using z-scores - which ones have experts that activate on wrong answers more than chance predicts?
2. For the worst layer(s), cluster the experts by their activation fingerprints using Jaccard distance - do the bad experts share a pattern?
3. Turn off those specific experts (not the whole layer), re-run the model, compare accuracy to baseline using McNemar's test and a random ablation control

**How we cluster:**

Each expert in a bad layer gets a **200-dim binary vector** - one bit per test question:
- `1` if that expert was selected for that question
- `0` if it was not

K-medoids with Jaccard distance groups the 128 experts by these fingerprints. A cluster where most activated questions were answered wrong = a group of bad experts. That tells us those experts are actively hurting performance.

**Note on the Phase 2 data:**
The Phase 2 results store raw router logit scores for all 128 experts per layer per question (all non-zero, values ~5.2-5.8). Since the router picks the top-8 by logit score and softmax preserves rank order, we binarize by taking the top-8 experts per question per layer: those 8 get `1`, the other 120 get `0`. No re-run needed.

We use **200 questions** for all experiments and hold out the other 200 to double-check findings at the end.

**Source data:** `data/phase2/runpod_second/results.json`

**Categories:** height, rotation, ordering, position

---

## Steps

### Step 1 - Split the Data

Take the 400 questions and split them into two equal groups of 200:
- `test_200.json` - all experiments run on this
- `holdout_200.json` - locked away, only used at the very end to verify results hold

Split by stratified sampling so each half has roughly 50 questions per category.

- [x] Confirm Phase 2 data has routing weights and a correct/incorrect label for all 400 questions
- [x] Confirm each question has `visual_routing`: a dict of layer index -> list of 128 floats
- [x] Split into `test_200.json` and `holdout_200.json`, balanced by category
- [x] Quick check: print category counts for each split to confirm balance
- [x] Compute and record the base error rate for the 200 test questions (total wrong / 200)

---

### Step 2 - Find the Bad Layers (Z-Score Method)

Using only the 200 test questions, score each of the 48 layers using z-scores that account for the model's overall error rate and each expert's activation count.

First, binarize the routing data: for each question and each layer, mark the top-8 experts by logit score as selected (1), all others as not selected (0). This gives the actual routing decisions the model made.

Then for each expert in each layer, compute:

```
expected_wrong = n * base_error_rate
std_dev = sqrt(n * base_error_rate * (1 - base_error_rate))
z = (observed_wrong - expected_wrong) / std_dev
```

Where `n` is the number of questions the expert activated on, `base_error_rate` is the model's overall error rate from Step 1, `observed_wrong` is how many of those questions the model got wrong, `expected_wrong` is how many you would expect to be wrong by chance, and `std_dev` is the natural random variation around that expectation.

Average the z-scores across all 128 experts in each layer. A positive layer z-score means the layer's experts lean toward wrong answers more than chance predicts.

- [x] Write a script to binarize routing data: top-8 per question per layer = 1, rest = 0
- [x] Compute z-score per expert per layer, then average per layer
- [x] Print a ranked table of all 48 layers: layer number, average z-score, and number of experts with z > 2
- [x] Plot a bar chart of all 48 layer z-scores to see which layers stand out
- [x] Pick the single worst layer to use first, then plan to try the top 3 and top 5

---

### Step 3 - Build the Expert Fingerprint Matrix

For the worst layer, build a binary matrix that shows which questions each expert was activated on.

**Each expert gets a 200-dim binary row vector:**
- Dimension `q` = `1` if this expert was selected for question `q`, `0` if not

**Matrix shape: 128 rows (experts) x 200 columns (questions)**

This is the fingerprint of each expert: "which questions did I get called on?"

- [x] For each of the 128 experts in the worst layer, extract its 200-dim binary question vector from the binarized data
- [x] Sanity check: each column (question) should sum to exactly 8 - only 8 experts are selected per question per layer
- [x] Repeat for the top 5 worst layers so that expert-level ablation is available for all of them in Step 5
- [ ] If the single worst layer gives weak clustering, stack the next worst layer: 2 layers = 256 rows of experts to cluster
- [x] Save the matrices as files

---

### Step 4 - Cluster the Experts (K-Medoids with Jaccard Distance)

Compute a Jaccard distance matrix for the 128 experts, then run k-medoids clustering. Jaccard distance only looks at questions where at least one expert was active, ignoring the ~93% of dimensions that are zero for both experts. This prevents shared inactivity from inflating similarity.

```
Jaccard similarity(A, B) = |A intersection B| / |A union B|
Jaccard distance(A, B) = 1 - Jaccard similarity(A, B)
```

Where `A` and `B` are the sets of questions each expert activates on, `|A intersection B|` is the count of questions where both experts are active, and `|A union B|` is the count of questions where at least one is active.

K-medoids works like k-means but uses an actual expert as each cluster center (instead of a fractional average that does not correspond to any real expert) and accepts Jaccard distance directly.

We want to find clusters where most of the questions they activate on were answered wrong - those are bad experts.

- [ ] Compute the 128x128 Jaccard distance matrix from the fingerprint matrix for each of the top 5 worst layers
- [ ] Run k-medoids with 2, 4, 6, and 8 clusters on each layer
- [ ] Use silhouette scores to pick the best number of clusters (higher = tighter clusters)
- [ ] For each cluster, check: what fraction of the questions those experts activate on were wrong? What categories dominate?
- [ ] If one cluster is mostly wrong-answer questions, those experts are a confirmed failure pattern
- [ ] Record the list of expert indices in each "bad" cluster per layer for use in Step 5

> Steps 1-4 are all local Python scripts. No GPU needed.

---

### Step 5 - Ablation Runs

This is the main experiment. Go back to RunPod and run the 200 test questions under multiple ablation conditions. The experiment has three parts: targeted ablation, ablation type comparison, and random ablation controls.

**Part A: Ablation type comparison (worst layer only)**

Test three ways of disabling the worst layer to isolate what is causing the errors:

| Run | What is disabled | What it tests |
|-----|-----------------|---------------|
| Baseline | Nothing | Reference accuracy |
| Full block ablation | Entire layer (attention + MoE) | Is this layer harmful overall? |
| MoE-only ablation | Just the MoE sublayer, attention stays active | Are the experts specifically harmful? |
| Expert ablation | Only the bad-cluster experts, rest of layer stays active | Are the clustered experts specifically harmful? |

For full block ablation, the residual stream passes through unchanged (as if the layer did nothing). For MoE-only ablation, the attention sublayer still runs but the MoE output is zeroed. For expert ablation, bad experts' gate weights are set to zero and remaining weights are renormalized to sum to 1.

**Part B: Scaling (MoE-only or expert ablation, whichever worked better in Part A)**

- [ ] Run three versions: apply the best ablation type to the 1 worst layer, the 3 worst layers, and the 5 worst layers
- [ ] Record accuracy per category for each version

**Part C: Random ablation control**

For each ablation count (1, 3, 5 layers), run 10 trials where the same number of randomly chosen layers are ablated. If expert ablation won Part A, use MoE-only ablation for the random trials instead, since random layers have not been clustered and there is no way to identify which experts to disable. This is the only way to confirm that our scoring method found specifically harmful layers, not just that the model tolerates layer removal in general.

- [ ] Set up a RunPod instance (GPU needed - this runs the actual model)
- [ ] Implement three ablation modes in the inference script: full block skip, MoE-only skip, and expert-level masking with gate weight renormalization
- [ ] Run Part A: baseline + 3 ablation types on the worst layer
- [ ] Decide which ablation type to use for Parts B and C based on Part A results
- [ ] Run Part B: skip 1, 3, 5 worst layers with the chosen ablation type
- [ ] Run Part C: 10 random trials for each of 1, 3, 5 layer counts (30 runs total)
- [ ] Record accuracy per category for every run

---

### Step 6 - Analyze and Wrap Up

Use McNemar's test to check if accuracy differences are real, not just noise. Use the random ablation control to check if our scoring method found real signal.

**McNemar's test (for each ablation vs. baseline):**

For each question, record whether it flipped:
- `b` = baseline right, ablation wrong (regressions)
- `c` = baseline wrong, ablation right (improvements)

```
chi_squared = (|b - c| - 1)^2 / (b + c)
```

If chi_squared > 3.84, the difference is statistically significant (p < 0.05).

**Random ablation comparison (for each ablation count):**

```
z = (targeted_accuracy - random_mean) / random_std
```

If z > 2, our targeted ablation is significantly better than random layer removal.

**Confidence intervals (for all accuracy numbers):**

```
CI = p +/- 1.96 * sqrt(p * (1 - p) / n)
```

Where `p` is accuracy as a decimal and `n` is the number of questions (200 for overall, 50 for per-category). Note: per-category intervals will be wide (~+/- 14%) due to the small sample size of 50, so be cautious about per-category claims.

- [ ] Build McNemar's 2x2 tables for each ablation run vs. baseline
- [ ] Compute chi_squared and p-values; report which differences are significant
- [ ] Compute random_mean and random_std for each layer count; compute z-scores
- [ ] Report 95% confidence intervals for all accuracy numbers
- [ ] Plot accuracy vs. number of skipped layers - does it go up, peak, then drop?
- [ ] Check per-category: does skipping bad layers help some types more than others?
- [ ] Re-run the best config on the holdout 200 to confirm the result generalizes
- [ ] Write one paragraph: which layers were worst, did removing them help, by how much, and is it statistically significant?

---

## Things to Watch Out For

> **The binarization assumption:** We binarize by taking the top-8 per question per layer because the router selects exactly 8. Since softmax preserves rank order, top-8 by logit = top-8 actually selected. But verify the sanity check in Step 3 - every column must sum to exactly 8. If any column sums to something other than 8, the binarization is wrong.

> **Did we actually fix it or just get lucky?** The holdout 200 in Step 6 is the check. If accuracy only goes up on the test set but not the holdout, the result does not generalize. McNemar's test on the holdout should also be significant.

> **Turning off too many layers will always hurt.** If you skip enough layers the model breaks regardless. Stop adding layers to the skip list when accuracy starts dropping below baseline.

> **A bad layer might just mean hard questions.** If the hardest questions happen to activate a certain layer more, that layer looks bad but is not causing the errors. The ablation run tests this - if accuracy goes up when the layer is off, it was actually hurting things. The random ablation control adds a second check: if random layers give the same improvement, the "bad" layer was not special.

> **Statistical significance is not optional at n=200.** With 50 questions per category, a swing of 3 questions changes accuracy by 6%. Always report McNemar's p-value and confidence intervals. Do not claim a result is real without them.

> **Expert ablation edge case:** If all 8 selected experts for a given question happen to be in the bad set, the MoE output is zero for that token. This is equivalent to full MoE ablation for that specific input. This should be rare if the bad cluster is small, but log how often it happens.
