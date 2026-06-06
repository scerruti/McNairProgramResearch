# Phase 3 Sprint Plan
## Finding and Turning Off Bad Layers

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-05 |
| **Phase** | 3 |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Find which internal layers of the model cause the most wrong answers, group questions by how those layers behaved, then turn off the bad layers and re-run to see if accuracy improves |

---

## What We Are Doing and Why

The model has 48 layers. Inside each layer are ~200 "experts" - small sub-networks the model can route a question through. For every question, the model picks a handful of experts per layer to do the work.

From Phase 2 we already have a log of which experts fired for each of the 400 questions, and whether each question was answered correctly or not.

The idea here is simple:

> If certain experts in a certain layer keep firing on questions the model gets wrong, that layer might be hurting performance. Turn it off and see if the model does better.

**The three-step logic:**
1. Score all 48 layers - which ones have experts that mostly fire on wrong answers?
2. Cluster the questions by which experts fired in those bad layers - do the wrong answers share a pattern?
3. Turn off the worst layers, re-run the model, compare accuracy to baseline

We use **200 questions** for all experiments and save the other 200 to double-check our findings at the end.

**Source data:** `data/phase2/runpod_second/results.json`

**Categories:** height, rotation, ordering, position

---

## Steps

### Step 1 - Split the Data

Take the 400 questions and split them into two equal groups of 200:
- `test_200.json` - this is what we run all experiments on
- `holdout_200.json` - locked away, only used at the very end to verify results

When splitting, make sure each half has roughly 50 questions per category (height, rotation, ordering, position) so neither group is skewed.

- [ ] Confirm Phase 2 data has expert activation info and a correct/incorrect label for all 400 questions
- [ ] Split into `test_200.json` and `holdout_200.json`, balanced by category
- [ ] Quick check: print category counts for each split to confirm balance

---

### Step 2 - Find the Bad Layers

For each of the 48 layers, compute one score that answers: "when experts in this layer fire, how often is the model wrong?"

For every expert in a layer, compute:
```
wrong-answer rate = (times it fired on a wrong answer) / (total times it fired)
```

Then average that rate across all experts in the layer. A layer scoring above 0.5 means its experts fire more on wrong answers than right ones - that is a bad layer.

Rank all 48 layers by this score. The highest scores are the layers we care about.

- [ ] Write a script to compute wrong-answer rates per expert, then average per layer
- [ ] Print a ranked table of all 48 layers: layer number and score
- [ ] Plot a bar chart of all 48 scores to visually see which layers stand out
- [ ] Pick the single worst layer to use first, then plan to try the top 3 and top 5

---

### Step 3 - Build the Question Feature Table

Now that we know which layer is worst, we turn each question into a row of numbers that describes exactly which experts fired in that layer.

Each row is a **200-number binary vector** (one number per expert):
- `1` = that expert fired for this question
- `0` = it did not

The result is a table with shape: **200 rows (questions) x 200 columns (experts)**

Each row is a fingerprint of how the model processed that question through the bad layer.

- [ ] For each of the 200 test questions, extract the 200-dim binary vector for the worst layer
- [ ] Sanity check: every row should sum to the same small number (only a few experts fire per question - this is the model's routing setting)
- [ ] If you want to include more than one bad layer, stack them side by side: 2 layers = 400 columns, 3 layers = 600 columns. Start with 1.
- [ ] Save the table as a file

---

### Step 4 - Cluster the Questions (K-Means)

Run K-means on the 200x200 table. K-means will group the 200 questions into clusters based on which experts fired. Questions in the same cluster share a routing pattern through the bad layer.

We want to find clusters where most questions were answered wrong - that is a clear failure mode.

- [ ] Run K-means with 2, 4, 6, and 8 clusters
- [ ] Use an elbow plot to pick the best number of clusters
- [ ] For each cluster, check: what fraction of questions were wrong? What categories dominate?
- [ ] If one cluster is mostly wrong answers, that routing pattern is a confirmed failure mode

> Steps 1-4 are all local Python scripts. No GPU needed.

---

### Step 5 - Turn Off the Bad Layers (Ablation Run)

This is the main experiment. We go back to RunPod, modify the inference script to skip the worst layer(s), and re-run the 200 test questions.

**How to turn off a layer:** zero out the layer's output so the residual stream passes through unchanged - as if the layer did nothing.

- [ ] Set up a RunPod instance (GPU needed - this runs the actual model)
- [ ] Add a "layers to skip" option to the inference script
- [ ] Run three versions: skip the 1 worst layer, skip the 3 worst, skip the 5 worst
- [ ] Record accuracy per category for each version
- [ ] Compare all three against the Phase 2 baseline (no layers skipped)

---

### Step 6 - Analyze and Wrap Up

- [ ] Plot accuracy vs number of skipped layers - does it go up, peak, then drop?
- [ ] Check per-category: does skipping bad layers help some types more than others?
- [ ] Re-run the best config on the holdout 200 to confirm the result holds
- [ ] Write one paragraph: which layers were worst, did removing them help, and by how much?

---

## Things to Watch Out For

> **Did we actually fix it or just get lucky?** The holdout 200 in Step 6 is the check. If accuracy only goes up on the test set but not the holdout, the result does not generalize.

> **Turning off too many layers will always hurt.** If you skip enough layers, the model breaks regardless. Stop adding layers to the skip list when accuracy starts dropping below baseline.

> **A bad layer might just mean hard questions.** If the hardest questions also happen to activate a certain layer more, that layer looks "bad" but is not causing the errors. The ablation run tests this - if accuracy goes up when the layer is off, it was actually hurting things.
