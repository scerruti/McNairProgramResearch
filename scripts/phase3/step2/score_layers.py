"""
Step 2: Score all 48 layers using z-scores to find which layers have experts
that activate on wrong answers more than chance predicts.

Process:
    1. Load the 200 test questions and the base error rate from Step 1
    2. Binarize the routing data: for each question and layer, the top 8
       experts by logit score are marked as selected (1), the other 120 as not (0)
    3. For each expert in each layer, compute a z-score measuring how many
       more wrong-answer activations it has compared to chance
    4. Average the z-scores across all 128 experts per layer to get a layer score
    5. Rank all 48 layers by score and identify the worst ones

Inputs:
    data/phase3/step1/test_200.json     - the 200 test questions
    data/phase3/step1/step1_meta.json   - contains base_error_rate

Outputs:
    data/phase3/step2/layer_scores.json     - ranked list of all 48 layer scores
    data/phase3/step2/binarized_routing.json - the binarized top-8 routing decisions
    data/phase3/step2/layer_scores.png       - bar chart of all 48 layer z-scores
    data/phase3/step2/step2_meta.json        - metadata including top 5 worst layers
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STEP1_DATA_DIR = PROJECT_ROOT / "data" / "phase3" / "step1"
STEP2_OUTPUT_DIR = PROJECT_ROOT / "data" / "phase3" / "step2"

TEST_QUESTIONS_PATH = STEP1_DATA_DIR / "test_200.json"
STEP1_META_PATH = STEP1_DATA_DIR / "step1_meta.json"

# --- Constants ---

EXPERTS_PER_LAYER = 128
LAYERS_COUNT = 48

# The router selects exactly 8 experts per question per layer.
# Binarization marks these 8 as selected (1) and the rest as not (0).
TOP_K_EXPERTS = 8


def load_test_questions():
    """Load the 200 test questions produced by Step 1."""
    with open(TEST_QUESTIONS_PATH) as file_handle:
        test_questions = json.load(file_handle)
    return test_questions


def load_base_error_rate():
    """
    Load the base error rate from Step 1 metadata.

    The base error rate is the fraction of the 200 test questions
    the model got wrong (e.g., 0.7650 means 76.5% wrong). It serves
    as the null hypothesis: if an expert has no relationship to
    correctness, its wrong-answer rate should be close to this value.
    """
    with open(STEP1_META_PATH) as file_handle:
        step1_meta = json.load(file_handle)
    return step1_meta["test_base_error_rate"]


def binarize_routing_for_question(visual_routing):
    """
    Convert raw logit scores into binary top-8 selection decisions.

    For each layer, the router selects the 8 experts with the highest
    logit scores. Softmax preserves rank order, so the top 8 by raw
    logit are the same 8 the router actually selected during inference.

    Args:
        visual_routing: dict mapping layer index (str or int) to a list
                        of 128 floats (one logit score per expert)

    Returns:
        dict mapping layer index (int) to a list of 128 binary values
        (1 = selected, 0 = not selected)
    """
    binarized_layers = {}

    for layer_index, expert_logits in visual_routing.items():
        layer_index_int = int(layer_index)

        # Find the indices of the top 8 experts by logit score.
        # enumerate pairs each logit with its expert index, then we sort
        # by logit value (descending) and take the first 8.
        indexed_logits = list(enumerate(expert_logits))
        sorted_by_logit = sorted(indexed_logits, key=lambda pair: pair[1], reverse=True)
        top_k_expert_indices = {pair[0] for pair in sorted_by_logit[:TOP_K_EXPERTS]}

        # Build the binary vector: 1 for selected experts, 0 for the rest.
        binary_vector = [
            1 if expert_index in top_k_expert_indices else 0
            for expert_index in range(EXPERTS_PER_LAYER)
        ]

        binarized_layers[layer_index_int] = binary_vector

    return binarized_layers


def binarize_all_questions(test_questions):
    """
    Binarize routing data for every question across all 48 layers.

    Returns a list of dicts, one per question, each mapping layer index
    to a 128-element binary list. Also carries over the 'correct' field
    so we can compute wrong-answer associations downstream.
    """
    binarized_questions = []

    for question in test_questions:
        binarized_routing = binarize_routing_for_question(question["visual_routing"])
        binarized_questions.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "correct": question["correct"],
            "binarized_routing": binarized_routing,
        })

    return binarized_questions


def compute_expert_z_score(
    activation_count, observed_wrong_count, base_error_rate
):
    """
    Compute the z-score for a single expert in a single layer.

    The z-score measures how many standard deviations above (or below)
    the expected wrong-answer count this expert falls.

    Formula:
        expected_wrong = n * base_error_rate
        std_dev = sqrt(n * base_error_rate * (1 - base_error_rate))
        z = (observed_wrong - expected_wrong) / std_dev

    Where n is the number of questions this expert was selected on.

    An expert that never activates (n=0) gets a z-score of 0 because
    there is no data to evaluate it. An expert with very few activations
    (e.g., n=2) will naturally produce a z-score close to 0 because
    the standard deviation is large relative to any deviation from
    expected, preventing noisy low-sample experts from dominating
    the layer score. This is why separate activation-frequency
    weighting is unnecessary (see ADR).

    Args:
        activation_count: number of questions this expert was selected on (n)
        observed_wrong_count: of those, how many the model got wrong
        base_error_rate: overall fraction of test questions answered wrong

    Returns:
        z-score as a float (positive = leans toward wrong answers,
        negative = leans toward correct answers, 0 = average or no data)
    """
    if activation_count == 0:
        return 0.0

    expected_wrong = activation_count * base_error_rate
    variance = activation_count * base_error_rate * (1.0 - base_error_rate)

    # Variance can only be zero if base_error_rate is exactly 0 or 1,
    # meaning the model got everything right or everything wrong.
    # In that case every expert trivially matches the base rate.
    if variance == 0.0:
        return 0.0

    standard_deviation = math.sqrt(variance)
    z_score = (observed_wrong_count - expected_wrong) / standard_deviation

    return z_score


def compute_layer_scores(binarized_questions, base_error_rate):
    """
    Score all 48 layers by averaging z-scores across their 128 experts.

    For each layer, we track two things per expert:
        - How many questions it was selected on (activation count)
        - Of those, how many the model got wrong (wrong activation count)

    Then we compute a z-score per expert and average across the layer.

    Returns:
        dict mapping layer index (int) to a dict containing:
            - 'average_z_score': mean z-score across all 128 experts
            - 'experts_above_z2': count of experts with z > 2
              (individually suspicious experts)
            - 'expert_z_scores': list of all 128 individual z-scores
              (used by Step 3 for deeper analysis)
    """
    # Track activation counts and wrong-activation counts per expert per layer.
    # expert_activation_counts[layer][expert] = number of questions this expert
    # was selected on.
    expert_activation_counts = defaultdict(lambda: defaultdict(int))
    expert_wrong_activation_counts = defaultdict(lambda: defaultdict(int))

    for question in binarized_questions:
        question_is_wrong = not question["correct"]

        for layer_index, binary_vector in question["binarized_routing"].items():
            for expert_index, is_selected in enumerate(binary_vector):
                if is_selected == 1:
                    expert_activation_counts[layer_index][expert_index] += 1
                    if question_is_wrong:
                        expert_wrong_activation_counts[layer_index][expert_index] += 1

    layer_scores = {}

    for layer_index in range(LAYERS_COUNT):
        expert_z_scores = []
        experts_above_z2_count = 0

        for expert_index in range(EXPERTS_PER_LAYER):
            activation_count = expert_activation_counts[layer_index][expert_index]
            wrong_count = expert_wrong_activation_counts[layer_index][expert_index]

            z_score = compute_expert_z_score(
                activation_count, wrong_count, base_error_rate
            )
            expert_z_scores.append(z_score)

            if z_score > 2.0:
                experts_above_z2_count += 1

        average_z_score = sum(expert_z_scores) / len(expert_z_scores)

        layer_scores[layer_index] = {
            "average_z_score": round(average_z_score, 4),
            "experts_above_z2": experts_above_z2_count,
            "expert_z_scores": [round(z, 4) for z in expert_z_scores],
        }

    return layer_scores


def print_ranked_table(layer_scores):
    """
    Print all 48 layers ranked by average z-score (worst first).

    Columns:
        Rank: position in the ranking (1 = worst layer)
        Layer: the layer index (0-47)
        Avg Z-Score: mean z-score across 128 experts (positive = bad)
        Experts z>2: count of experts individually above z=2
    """
    ranked_layers = sorted(
        layer_scores.items(),
        key=lambda item: item[1]["average_z_score"],
        reverse=True,
    )

    print(f"\n{'Rank':<6} {'Layer':<7} {'Avg Z-Score':<13} {'Experts z>2':<13}")
    print("-" * 39)

    for rank, (layer_index, scores) in enumerate(ranked_layers, start=1):
        print(
            f"{rank:<6} {layer_index:<7} {scores['average_z_score']:>+10.4f}"
            f"   {scores['experts_above_z2']:>5}"
        )

    return ranked_layers


def plot_layer_scores(layer_scores, output_path):
    """
    Bar chart of all 48 layer z-scores so outliers are visually obvious.

    Layers with positive z-scores (bar goes up) lean toward wrong answers.
    Layers with negative z-scores (bar goes down) lean toward correct answers.
    The top 5 worst layers are highlighted in red.
    """
    ranked_layers = sorted(
        layer_scores.items(),
        key=lambda item: item[1]["average_z_score"],
        reverse=True,
    )
    top_5_worst_indices = {layer_index for layer_index, _ in ranked_layers[:5]}

    layer_indices = list(range(LAYERS_COUNT))
    z_scores = [layer_scores[i]["average_z_score"] for i in layer_indices]

    bar_colors = [
        "#d32f2f" if i in top_5_worst_indices else "#1976d2"
        for i in layer_indices
    ]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(layer_indices, z_scores, color=bar_colors, edgecolor="none")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Average Z-Score")
    ax.set_title("Layer Z-Scores (red = top 5 worst)")
    ax.set_xticks(range(0, LAYERS_COUNT, 2))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    print("=" * 60)
    print("Step 2: Find the Bad Layers (Z-Score Method)")
    print("=" * 60)

    # --- Load ---
    print(f"\nLoading test questions from:\n  {TEST_QUESTIONS_PATH}")
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} questions.")

    base_error_rate = load_base_error_rate()
    print(f"Base error rate: {base_error_rate:.4f} ({base_error_rate:.2%})")

    # --- Binarize ---
    # Convert raw logit scores to binary top-8 selections.
    # After this, each question's routing data is a simple
    # yes/no for each expert: was it one of the top 8 or not?
    print("\n--- Binarizing Routing Data ---")
    binarized_questions = binarize_all_questions(test_questions)
    print(f"Binarized routing for {len(binarized_questions)} questions across {LAYERS_COUNT} layers.")

    # Sanity check: every question-layer pair should have exactly 8 selected experts.
    for question in binarized_questions:
        for layer_index, binary_vector in question["binarized_routing"].items():
            selected_count = sum(binary_vector)
            if selected_count != TOP_K_EXPERTS:
                print(
                    f"ERROR: Question '{question['question_id']}', layer {layer_index}: "
                    f"expected {TOP_K_EXPERTS} selected experts, got {selected_count}"
                )
                return
    print(f"Sanity check passed: all question-layer pairs have exactly {TOP_K_EXPERTS} selected experts.")

    # --- Score Layers ---
    print("\n--- Computing Z-Scores ---")
    layer_scores = compute_layer_scores(binarized_questions, base_error_rate)

    ranked_layers = print_ranked_table(layer_scores)

    # --- Identify Top 5 ---
    top_5_worst = [
        {"layer_index": layer_index, "average_z_score": scores["average_z_score"],
         "experts_above_z2": scores["experts_above_z2"]}
        for layer_index, scores in ranked_layers[:5]
    ]

    print("\n--- Top 5 Worst Layers ---")
    for entry in top_5_worst:
        print(
            f"  Layer {entry['layer_index']}: "
            f"avg z-score = {entry['average_z_score']:+.4f}, "
            f"experts with z>2 = {entry['experts_above_z2']}"
        )

    # --- Save ---
    print("\n--- Saving Output ---")
    STEP2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full layer scores (including per-expert z-scores for Step 3).
    layer_scores_path = STEP2_OUTPUT_DIR / "layer_scores.json"
    # Convert int keys to strings for JSON serialization.
    serializable_scores = {str(k): v for k, v in layer_scores.items()}
    with open(layer_scores_path, "w") as file_handle:
        json.dump(serializable_scores, file_handle, indent=2)
    print(f"Saved: {layer_scores_path}")

    # Save binarized routing data so Step 3 can build fingerprint
    # matrices without re-running binarization.
    binarized_path = STEP2_OUTPUT_DIR / "binarized_routing.json"
    serializable_binarized = []
    for question in binarized_questions:
        serializable_binarized.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "correct": question["correct"],
            "binarized_routing": {
                str(k): v for k, v in question["binarized_routing"].items()
            },
        })
    with open(binarized_path, "w") as file_handle:
        json.dump(serializable_binarized, file_handle, indent=2)
    print(f"Saved: {binarized_path}")

    # Plot bar chart.
    plot_path = STEP2_OUTPUT_DIR / "layer_scores.png"
    plot_layer_scores(layer_scores, plot_path)

    # Save metadata for downstream steps.
    step2_metadata = {
        "base_error_rate": base_error_rate,
        "test_question_count": len(test_questions),
        "top_5_worst_layers": top_5_worst,
    }
    meta_path = STEP2_OUTPUT_DIR / "step2_meta.json"
    with open(meta_path, "w") as file_handle:
        json.dump(step2_metadata, file_handle, indent=2)
    print(f"Saved: {meta_path}")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
