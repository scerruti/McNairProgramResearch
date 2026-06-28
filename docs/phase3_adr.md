# ADR: Phase 3 Methodology Decisions
## How We Score Layers, Cluster Experts, and Validate Results

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-25 |
| **Phase** | 3 |
| **Status** | Accepted |
| **Applies to** | `docs/phase3_sprint_plan.md` |

---

## Context

The original Phase 3 sprint plan proposed a six-step pipeline: split data, score layers by wrong-answer rate, cluster experts with K-means, ablate bad layers, and analyze results. After review, we identified several places where the methodology could silently produce misleading results. This document records which improvements we adopted, which we rejected, and why.

We evaluated eight proposed improvements and sorted them into three tiers:

- **Adopted (6):** These fix real problems or fill gaps that would otherwise make results uninterpretable.
- **Rejected as redundant (1):** This solves a problem that is already handled by an adopted improvement.
- **Rejected as impractical (1):** The idea is theoretically sound but not worth the cost at our sample size.

---

## Decisions

### ADOPTED: Use Z-Scores Instead of Raw Wrong-Answer Rates (Layer Scoring)

**What changed:** Instead of computing each expert's raw wrong-answer rate and averaging across the layer, we now compute a z-score per expert that accounts for (a) the model's overall error rate and (b) how many questions the expert activated on.

**The problem with the old approach:**

The old formula was:

```
wrong_rate(expert) = wrong_activations / total_activations
layer_score = average of wrong_rate across all 128 experts
```

Then we said "a layer scoring above 0.5 is bad." But this ignores how often the model is wrong in general. If the model gets 70% of questions wrong overall, then every expert will have a wrong-rate near 0.70 just by chance. The 0.5 threshold would flag almost every layer as bad, even though none of them are doing anything unusual.

**The new formula:**

For each expert `e` in a layer, compute a z-score:

```
expected_wrong = n * base_error_rate
std_dev = sqrt(n * base_error_rate * (1 - base_error_rate))
z = (observed_wrong - expected_wrong) / std_dev
```

Where:
- `n` = number of questions this expert was selected on (its activation count)
- `base_error_rate` = total wrong answers / total questions across the whole test set (for example, if the model gets 140 out of 200 wrong, this is 0.70)
- `observed_wrong` = number of questions this expert was selected on where the model answered wrong
- `expected_wrong` = how many wrong answers you would expect if this expert had no relationship to correctness at all. It is just `n` times the base error rate. If an expert activates on 20 questions and the model is wrong 70% of the time, you would expect 14 of those 20 to be wrong by pure chance.
- `std_dev` = the standard deviation of a binomial distribution. It measures how much the wrong count naturally bounces around due to randomness. With more activations (`n`), the absolute bounce gets bigger, but relative to `n` it gets smaller. This is why large samples give more stable estimates.
- `z` = how many standard deviations above or below the expected value this expert lands. A z of 0 means average. A z of +2 means the expert activates on wrong answers way more than chance. A z of -2 means it activates on correct answers more than chance.

Then the layer score is:

```
layer_z_score = average of z across all 128 experts in the layer
```

A positive layer z-score means the layer's experts lean toward wrong answers more than chance predicts. A negative score means they lean toward correct answers.

**Why z-scores also replace activation-frequency weighting:**

The z-score formula already handles the problem of experts with few activations. The standard deviation in the denominator scales with `sqrt(n)`. An expert that activates on only 2 questions has a large standard deviation relative to any deviation from expected, so its z-score collapses toward zero automatically. It cannot dominate the layer average. An expert that activates on 100 questions has a smaller relative standard deviation, so real deviations show up as large z-scores. No additional weighting needed.

**Example:**

Base error rate = 0.60 (the model gets 120 out of 200 questions wrong).

Expert A activates on 100 questions. 75 were wrong.

```
expected_wrong = 100 * 0.60 = 60
std_dev = sqrt(100 * 0.60 * 0.40) = sqrt(24) = 4.90
z = (75 - 60) / 4.90 = 15 / 4.90 = +3.06
```

Expert A is 3 standard deviations above expected. Very suspicious.

Expert B activates on 4 questions. 3 were wrong.

```
expected_wrong = 4 * 0.60 = 2.4
std_dev = sqrt(4 * 0.60 * 0.40) = sqrt(0.96) = 0.98
z = (3 - 2.4) / 0.98 = 0.6 / 0.98 = +0.61
```

Expert B is less than 1 standard deviation above expected. Not suspicious at all. Getting 3 out of 4 wrong when the base rate is 60% is totally normal.

Both experts have a raw wrong-rate of 75%. Without z-scores, they look equally bad. With z-scores, Expert A is flagged and Expert B is correctly ignored.

---

### ADOPTED: Use Jaccard Distance Instead of Euclidean for Clustering

**What changed:** When clustering the 128 experts by their activation fingerprints, use Jaccard distance with k-medoids (or hierarchical clustering) instead of Euclidean distance with k-means.

**The problem with Euclidean distance on binary vectors:**

Each expert's fingerprint is a 200-dimensional binary vector (one bit per question: 1 if the expert was selected, 0 if not). The router picks 8 out of 128 experts per question, so each vector is about 93.7% zeros. Any two experts will share roughly 180+ zeros out of 200 dimensions just because most entries are zero for everyone.

Euclidean distance counts those shared zeros as similarity. Two experts that activate on completely different questions still "agree" on 180+ dimensions (the ones where neither was active). Euclidean distance thinks they are similar. This washes out the real differences between experts.

**How Jaccard distance works:**

Jaccard only looks at questions where at least one of the two experts was active. It completely ignores shared zeros.

```
Jaccard similarity = |A intersection B| / |A union B|
Jaccard distance = 1 - Jaccard similarity
```

Where:
- `A` = the set of questions expert A activates on
- `B` = the set of questions expert B activates on
- `|A intersection B|` = number of questions where both A and B are active
- `|A union B|` = number of questions where at least one of A or B is active

**Example:**

Expert X activates on questions: {1, 2, 3, 4, 5, 10, 15, 20, 25, 30}
Expert Y activates on questions: {1, 2, 3, 50, 51, 52, 60, 70, 80, 90}

They share questions {1, 2, 3}.

```
|X intersection Y| = 3
|X union Y| = 17 (the 10 from X plus the 7 unique to Y)

Jaccard similarity = 3 / 17 = 0.176
Jaccard distance = 1 - 0.176 = 0.824
```

These experts are very different. They only overlap on 3 out of 17 relevant questions.

Euclidean distance on the full 200-dim vectors: they agree on 186 out of 200 positions (3 shared ones + 183 shared zeros). Euclidean sees 93% agreement. Jaccard correctly sees 82.4% disagreement.

**Why k-medoids instead of k-means:**

K-means computes cluster centers as the mean of all points in the cluster. The mean of a bunch of binary vectors is a vector of fractions (like 0.3, 0.7, 0.1...), which is not a real expert. K-medoids picks an actual data point as the cluster center, so the center is always a real expert with a real activation pattern. K-medoids also accepts any distance metric, including Jaccard.

---

### ADOPTED: Expert-Level Ablation Instead of Layer-Level Only

**What changed:** Instead of only testing "turn off the entire layer," also test "turn off only the bad experts within a layer." This makes the clustering step (Steps 3-4) directly useful instead of just diagnostic.

**The problem with layer-level ablation alone:**

A layer has 128 experts. Your clustering might find that 30 of them form a "bad" cluster (they activate mostly on wrong answers). But the other 98 experts might be fine or even helpful. If you turn off the entire layer, you lose the good experts along with the bad ones. The improvement from removing bad experts gets partially canceled by the loss of good experts.

**How expert-level ablation works:**

In a Mixture-of-Experts layer, the output is a weighted sum of the selected experts:

```
MoE_output = sum of (weight_i * expert_i(input)) for each selected expert i
```

The router picks 8 experts and assigns each a weight (from softmax). The weights sum to 1.

To ablate specific experts, you set their weights to zero and renormalize the remaining weights so they still sum to 1:

```
1. Identify bad experts from clustering (for example: {3, 17, 42, 91})
2. For each question, check which of the 8 selected experts are in the bad set
3. Set those experts' weights to 0
4. Divide each remaining expert's weight by the sum of all remaining weights
```

**Renormalization example:**

8 experts are selected for a question with these weights:

| Expert | Weight | Bad? |
|--------|--------|------|
| E3     | 0.20   | Yes  |
| E17    | 0.15   | No   |
| E42    | 0.18   | Yes  |
| E56    | 0.12   | No   |
| E71    | 0.10   | No   |
| E85    | 0.09   | No   |
| E91    | 0.08   | Yes  |
| E100   | 0.08   | No   |

Bad experts (E3, E42, E91) are zeroed out. Remaining weights sum to:

```
0.15 + 0.12 + 0.10 + 0.09 + 0.08 = 0.54
```

Renormalize each remaining weight by dividing by 0.54:

| Expert | Old weight | New weight |
|--------|-----------|------------|
| E17    | 0.15      | 0.15 / 0.54 = 0.278 |
| E56    | 0.12      | 0.12 / 0.54 = 0.222 |
| E71    | 0.10      | 0.10 / 0.54 = 0.185 |
| E85    | 0.09      | 0.09 / 0.54 = 0.167 |
| E100   | 0.08      | 0.08 / 0.54 = 0.148 |

New weights sum to 1.000. The relative ranking is preserved (E17 is still strongest), but each expert contributes more to compensate for the removed ones.

**What this means for the experiment:**

We now run four comparisons instead of one:

| Run | What is disabled | What it tests |
|-----|-----------------|---------------|
| Baseline | Nothing | Reference accuracy |
| Layer ablation | Entire worst layer (attention + MoE) | Is this layer harmful overall? |
| MoE-only ablation | Just the MoE sublayer | Are the experts specifically harmful? |
| Expert ablation | Only bad-cluster experts in worst layer | Are the clustered experts specifically harmful? |

If expert ablation gives a bigger improvement than full layer ablation, the clustering found real structure and the good experts in that layer were worth keeping.

---

### ADOPTED: Separate Attention vs. MoE Ablation

**What changed:** When ablating a layer, test two versions: (a) skip the entire transformer block (attention + MoE) and (b) skip only the MoE sublayer while keeping attention active.

**Why this matters:**

Each transformer block has two sublayers that modify the residual stream:

```
x = x + Attention(Norm(x))    (sublayer 1: self-attention)
x = x + MoE(Norm(x))          (sublayer 2: mixture of experts)
```

Our scoring is based entirely on expert routing data from the MoE component. We have no evidence that the attention sublayer in a "bad" layer is also bad. It might be doing useful work like attending to the right parts of the image.

If we skip the entire block and accuracy goes up, we cannot tell if the improvement came from removing the bad MoE experts, from removing the attention heads, or from some combination. If we skip only the MoE sublayer and accuracy goes up, the result directly tests our hypothesis that the experts are causing errors.

**MoE-only ablation** means:

```
x = x + Attention(Norm(x))    (still runs)
x = x + 0                      (MoE output zeroed)
```

**Full block ablation** means:

```
x = x + 0                      (attention zeroed)
x = x + 0                      (MoE zeroed)
```

Which simplifies to: `x` passes through unchanged. The layer does nothing.

---

### ADOPTED: Random Ablation Control

**What changed:** For each ablation experiment (skip 1 layer, skip 3, skip 5), also run 10 trials where we skip the same number of randomly chosen layers. Compare the targeted result against the distribution of random results.

**Why this is necessary (and not redundant with McNemar's test):**

McNemar's test and the random control answer two different questions:

| Question | What answers it |
|----------|----------------|
| Did ablation change accuracy compared to baseline? | McNemar's test |
| Did ablation change accuracy more than random ablation? | Random control |
| Did our scoring method find real signal? | Random control |

McNemar's can tell you "yes, removing these 3 layers significantly changed accuracy." But it cannot tell you whether removing any 3 layers would have done the same thing. Research on large language models has shown that you can remove significant fractions of layers with minimal performance drop. If the model just tolerates layer removal in general, McNemar's would show significance for random ablation too. The random control is the only way to establish that our scoring method identified specifically harmful layers.

**How to run it:**

For each ablation count `k` (1, 3, or 5 layers), run 10 random trials. Each trial randomly picks `k` layers to skip and measures accuracy on the 200 test questions.

Compute:

```
random_mean = sum of all 10 random accuracies / 10
random_std = sqrt( sum of (accuracy_r - random_mean)^2 / 9 )
z = (targeted_accuracy - random_mean) / random_std
```

Where:
- `random_mean` = the average accuracy you get from random layer removal
- `random_std` = how much the random trials vary (we divide by 9 instead of 10 because of Bessel's correction, which adjusts for the fact that we are estimating variability from a small sample)
- `z` = how many standard deviations above the random mean our targeted result lands

If `z > 2`, our targeted ablation is significantly better than random. Our scoring found real signal. If `z < 1`, our targeted ablation is no better than randomly picking layers. The scoring method did not work.

**Example:**

Targeted ablation (3 worst layers): 62% accuracy.
10 random trials: 54%, 53%, 56%, 52%, 55%, 53%, 54%, 56%, 51%, 54%.

```
random_mean = 53.8%
random_std = sqrt(23.6 / 9) = 1.62%
z = (62.0 - 53.8) / 1.62 = 5.07
```

Z = 5.07. Our targeted ablation is over 5 standard deviations above random. Very strong evidence.

**Compute cost:** 10 random trials per ablation count is 30 extra inference runs total. We are already running at least 3 targeted runs plus baseline. This roughly quadruples the GPU time but is the only way to validate the entire scoring pipeline.

---

### ADOPTED: McNemar's Test for Statistical Significance

**What changed:** When comparing baseline accuracy to ablation accuracy, use McNemar's test instead of just comparing two percentages. Also report 95% confidence intervals for all accuracy numbers.

**Why raw accuracy comparison is not enough:**

With 200 test questions (50 per category), small accuracy changes are hard to distinguish from noise. If baseline gets 110 right and ablation gets 118 right, that is a difference of 8 questions. Could easily be luck. We need a formal test.

**How McNemar's test works:**

Both the baseline and ablated model answer the same 200 questions. For each question, there are four possibilities:

```
                    Ablation correct    Ablation wrong
Baseline correct         a                   b
Baseline wrong           c                   d
```

- `a` = both got it right. Not interesting, they agree.
- `d` = both got it wrong. Not interesting, they agree.
- `b` = baseline got it right but ablation got it wrong. These are regressions.
- `c` = baseline got it wrong but ablation got it right. These are improvements.

Only `b` and `c` matter. The test statistic is:

```
chi_squared = (|b - c| - 1)^2 / (b + c)
```

Where:
- `|b - c|` = the absolute difference between regressions and improvements. If ablation helped more questions than it hurt, `c > b` and this is positive.
- The `-1` is a continuity correction. The chi-squared distribution is smooth but `b` and `c` are whole numbers. Subtracting 1 makes the test slightly more conservative and prevents false positives when counts are small.
- `(b + c)` = total number of questions that flipped either direction. This normalizes the result. If 100 questions flipped and the gap is 10, that is less impressive than if 15 questions flipped and the gap is 10.

The result follows a chi-squared distribution with 1 degree of freedom. Look up the p-value:

| chi_squared value | p-value | Interpretation |
|-------------------|---------|---------------|
| 2.71              | 0.10    | Weak evidence |
| 3.84              | 0.05    | Standard threshold |
| 6.63              | 0.01    | Strong evidence |

**Example:**

200 questions. Baseline: 110 correct. Ablation: 118 correct.

Question-level breakdown:
- a = 105 (both right)
- b = 5 (baseline right, ablation wrong)
- c = 13 (baseline wrong, ablation right)
- d = 77 (both wrong)

```
chi_squared = (|5 - 13| - 1)^2 / (5 + 13) = (8 - 1)^2 / 18 = 49 / 18 = 2.72
```

p-value is about 0.099. Not significant at the 0.05 level. Even though ablation improved 8 net questions (4% accuracy gain), the evidence is not strong enough to rule out chance.

**Confidence intervals:**

For any accuracy measurement, report a 95% confidence interval:

```
CI = p +/- 1.96 * sqrt(p * (1 - p) / n)
```

Where:
- `p` = observed accuracy as a decimal (for example, 118/200 = 0.59)
- `n` = number of questions (200 for the full set, 50 for a single category)
- `1.96` = the number of standard deviations that captures 95% of a normal distribution
- `sqrt(p * (1 - p) / n)` = the standard error. It measures how much the observed accuracy would bounce around if you repeated the experiment with different questions.

For 118/200 = 0.59:

```
CI = 0.59 +/- 1.96 * sqrt(0.59 * 0.41 / 200) = 0.59 +/- 0.068 = [0.522, 0.658]
```

The true accuracy is plausibly anywhere from 52.2% to 65.8%. For per-category results with only 50 questions, the interval is much wider (roughly +/- 14%), so per-category claims need to be made cautiously.

---

### REJECTED: Activation-Frequency Weighting (Redundant)

**What was proposed:** Weight each expert's wrong-answer rate by how many questions it activated on, so that rarely-activated experts do not dominate the layer score.

**Why we rejected it:** The z-score formula (adopted above) already handles this. The standard deviation in the denominator scales with `sqrt(n)`, which automatically shrinks the z-score of experts with few activations. An expert that activates on 2 questions cannot produce a large z-score no matter what happens on those 2 questions. Applying an additional frequency weight on top of the z-score would double-count the sample size effect and artificially bias results toward high-frequency experts. It could bury genuinely toxic low-frequency experts whose z-scores are legitimately large.

---

### REJECTED: Greedy Sequential Ablation (Impractical)

**What was proposed:** Instead of ranking all 48 layers once and then ablating the top N, ablate the single worst layer, re-score all remaining layers, ablate the next worst, re-score again, and repeat.

**Why the idea has merit:** When you remove a layer, the residual stream changes for all downstream layers. A layer that looks bad with all 48 active might behave differently after an upstream layer is removed. Greedy ablation accounts for these interaction effects.

**Why we rejected it:** With only 200 test questions, greedy search over layer combinations will almost certainly overfit. Each round of re-scoring optimizes for the specific 200 questions in the test set, and multiple rounds of optimization compound the overfitting risk. The holdout set catches this at the end, but by then you have spent significant GPU compute. The full greedy approach requires 230 inference runs. Even the hybrid approach (greedy over the top 10 candidates) requires 27 runs per ablation target. This is a lot of GPU time for a marginal improvement in layer ordering that may not generalize. Static scoring with the z-score method is sufficient for our purposes.

---

## Summary Table

| Improvement | Decision | Reason |
|------------|----------|--------|
| Z-scores for layer scoring | **Adopted** | Fixes base-rate bias and handles sample size |
| Jaccard distance for clustering | **Adopted** | Correct metric for sparse binary vectors |
| Expert-level ablation | **Adopted** | Makes clustering actionable, more targeted intervention |
| Attention vs. MoE ablation | **Adopted** | Isolates the variable we actually measured |
| Random ablation control | **Adopted** | Only way to validate the scoring method found real signal |
| McNemar's test | **Adopted** | Correct significance test for paired binary outcomes |
| Activation-frequency weighting | **Rejected** | Redundant with z-scores |
| Greedy sequential ablation | **Rejected** | Overfitting risk and compute cost at n=200 |
