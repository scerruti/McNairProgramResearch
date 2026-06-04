# Phase 0 Sprint Plan
## Getting a Baseline Score with Fireworks AI

| Field | Detail |
|-------|--------|
| **Date** | 2026-03-22 |
| **Phase** | 0 |
| **Model** | Qwen3-VL-30B-A3B-Instruct (via Fireworks AI API) |
| **Benchmark** | LEGOLite (400 questions, 4 categories) |
| **Goal** | Run Qwen3-VL-30B on the LEGO spatial reasoning questions and get a first accuracy score to compare against other models |

---

## Background

The LEGO-Puzzles benchmark is a set of visual questions that test spatial reasoning - things like figuring out the height, position, rotation, or ordering of LEGO bricks in an image. There are 1,100 questions across 11 question types.

Since the model we wanted to test (Qwen3-VL-30B) was not available on our usual platform (Parasail), we used Fireworks AI instead, which lets you call the model through an API over the internet - no GPU setup required.

We also created a smaller version of the benchmark called **LEGOLite** - just 400 questions covering the 4 categories we care about most - to keep costs and runtime low.

---

## Sprint Tasks

### Step 1 - Connect Fireworks AI
- [x] Add Fireworks AI as a model provider in the LEGO-Puzzles evaluation code
- [x] Set up the API key and point the code at Fireworks' servers
- [x] Fix API call formatting issues that Fireworks rejected
- [x] Add delays between requests to stay within Fireworks' free tier rate limits

### Step 2 - Build the LEGOLite Question Set
- [x] Pick the 4 question categories to focus on: height, position, rotation, ordering
- [x] Cut the full 1,100-question set down to 400 questions
- [x] Test a small sample to make sure everything runs before doing the full evaluation

### Step 3 - Fix Setup Issues
- [x] Find replacements for Python packages that did not work on Mac
- [x] Confirm the full pipeline runs correctly on a local machine

### Step 4 - Run the Evaluation
- [x] Send all 400 LEGOLite questions to the model via Fireworks AI and collect answers
- [x] Calculate accuracy for each category and overall

### Step 5 - Record Results
- [x] Save results to `docs/LEGO_Research_Log.md`
- [x] Compare our score against Gemini and GPT-4o Mini

---

## Results

| Category | Accuracy |
|----------|----------|
| **Overall** | **27.5%** |
| position | 42% |
| height | 34% |
| rotation | 34% |
| ordering | 0% |

**Comparison baselines (LEGOLite, 4 categories):**

| Model | Overall | height | position | rotation | ordering |
|-------|---------|--------|----------|----------|---------|
| GeminiFlash2-0 | 44.25% | 35% | 47% | 49% | 46% |
| Qwen3-VL-30B (ours) | 27.5% | 34% | 42% | 34% | 0% |
| GPT-4o Mini | 13.75% | 29% | 10% | 12% | 4% |

Our model lands between Gemini and GPT-4o Mini overall. The biggest surprise is ordering at 0% - these questions ask the model to arrange a sequence of steps in the correct order, which is much harder than picking a single answer.

---

## Key Takeaways

- Fireworks AI worked well as an API provider; the free tier rate limits just required adding wait times between requests
- Ordering (0%) is a fundamentally harder question type and behaves differently from the other 3 categories
- Position (42%) was our strongest category, but Gemini still beat us on all 4 - especially ordering (46% vs our 0%)
- Runtime: ~47 minutes for 400 questions on the free tier
