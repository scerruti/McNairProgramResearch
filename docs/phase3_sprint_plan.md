# Phase 3 Sprint Plan
## Grouping Questions by How the Model Thinks (K-Means Clustering)

| Field | Detail |
|-------|--------|
| **Date** | 2026-06-04 |
| **Phase** | 3 |
| **Model** | Qwen3-VL-30B-A3B-Instruct |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Use K-means clustering to find natural groupings in the expert activation data from Phase 2, and see if the model's internal routing aligns with the question categories |

---

## Background

Phases 1 and 2 told us which internal experts the model activates for each question, broken down by layer. Now we want to ask: if you group questions purely by their layer activation patterns, do those groups match up with the 4 question categories (height, rotation, position, ordering)?

The model has 48 processing layers. Early layers tend to handle low-level visual details, while later layers handle more abstract reasoning. If height questions consistently light up different layers than ordering questions, that tells us the model is using genuinely different processing pathways for different types of spatial reasoning.

If the groups don't align with categories, it might mean something else is driving the routing - like question difficulty, image complexity, or whether the model got it right or wrong.

We use **K-means clustering** to find these groups automatically, without telling the algorithm anything about the categories ahead of time.

**Source data:** `data/phase2/runpod_second/results.json` (400 questions, all valid)

**Categories:** height, rotation, ordering, position

---

## Sprint Tasks

### Step 1 - Verify the Data
- [x] Confirm Phase 2 Run 2 has all 400 questions with valid expert activation data
- [ ] Load and inspect the data before building feature vectors

---

### Step 2 - Decide Which Features to Use

We need to turn each question's expert activation data into a set of numbers the clustering algorithm can work with. We start with 3 features:

| # | Feature | What it captures |
|---|---------|-----------------|
| 1 | **Layer activation profile** | A list of 48 numbers - how much total expert activity happened at each of the 48 layers for this question |
| 2 | **Processing depth** | A single number (0 to 47) - on average, how deep into the network was the heaviest routing? Early layers handle low-level perception, late layers handle higher reasoning |
| 3 | **Expert focus** | A single number (0 to 1) - did a few experts dominate, or was activity spread across many? Higher = more concentrated |

**Other features to add later:**
- Whether the question is multiple choice vs ordering type
- How consistent expert choices were across adjacent layers
- Length of the question text

- [ ] Confirm the 3 starting features make sense for what we want to measure
- [ ] Note any features to add before the first run

---

### Step 3 - Build the Feature Matrix

- [ ] Write a script that reads the Phase 2 results and computes the 3 features for each of the 400 questions
- [ ] For each question, sum up all expert activations per layer to get a 48-number layer activity profile
- [ ] Combine all features into a single table (400 rows, one per question)
- [ ] Save the table as a file for use on RunPod

---

### Step 4 - Set Up RunPod

K-means clustering does not need a GPU - a basic CPU instance works fine here.

- [ ] Start a RunPod instance
- [ ] Upload the feature table and clustering script
- [ ] Make sure `scikit-learn` and `matplotlib` are installed

---

### Step 5 - Run K-Means

- [ ] Run K-means trying 2 groups, 4 groups, and 8 groups (4 matches our number of question categories)
- [ ] Use an elbow plot and silhouette scores to find the best number of groups
- [ ] Save which group each question was assigned to, along with its category and whether it was answered correctly

---

### Step 6 - Analyze by Layer

Before interpreting the clusters, look at how activity is distributed across the 48 layers for each question category:

- [ ] Plot average layer activity per category - which layers are busiest for height vs rotation vs position vs ordering?
- [ ] Check if categories differ in whether they activate early layers (low-level visual processing) or late layers (higher-level reasoning)
- [ ] Check if correct answers show different layer activity patterns than incorrect answers
- [ ] Identify any layers that are consistently more active across all categories (general purpose) vs layers that spike only for specific categories

---

### Step 7 - Look at the Clustering Results

- [ ] Compare the clusters to the 4 question categories - do they line up, or do they cut across categories?
- [ ] Check if correct and incorrect answers land in different clusters
- [ ] Plot the clusters on a 2D chart, colored by category
- [ ] Plot again colored by correct vs incorrect
- [ ] Write up the main finding: what does the model's layer-by-layer routing reveal about how it handles different types of spatial reasoning?

---

## Open Questions

> **Cluster shape:** K-means works best when groups are round and evenly sized. If the data looks stretched or uneven on the chart, we may want to try a different method like DBSCAN or Gaussian Mixture Models.
