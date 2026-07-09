"""
Step 3: Build expert fingerprint matrices for the top 5 worst layers.

For each layer, builds a 128x200 binary matrix where row i is expert i's
activation pattern across the 200 test questions. Column q = 1 if expert i
was selected for question q, 0 otherwise.

Inputs:
    data/phase3/step1/step1_meta.json         - test question count
    data/phase3/step2/binarized_routing.json  - binarized top-8 routing decisions
    data/phase3/step2/step2_meta.json         - top 5 worst layers and router top-k

Outputs:
    data/phase3/step3/fingerprint_layer_XX.json - 128x200 matrix per layer
    data/phase3/step3/step3_meta.json           - metadata and sanity check results
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STEP1_DATA_DIR = PROJECT_ROOT / "data" / "phase3" / "step1"
STEP2_DATA_DIR = PROJECT_ROOT / "data" / "phase3" / "step2"
STEP3_OUTPUT_DIR = PROJECT_ROOT / "data" / "phase3" / "step3"

STEP1_META_PATH = STEP1_DATA_DIR / "step1_meta.json"
BINARIZED_PATH = STEP2_DATA_DIR / "binarized_routing.json"
STEP2_META_PATH = STEP2_DATA_DIR / "step2_meta.json"

EXPERTS_PER_LAYER = 128
# The router selects exactly this many experts per question per layer.
ROUTER_TOP_K = 8


def load_binarized_data():
    with open(BINARIZED_PATH) as file_handle:
        return json.load(file_handle)


def load_step1_meta():
    with open(STEP1_META_PATH) as file_handle:
        return json.load(file_handle)


def load_top_layers():
    with open(STEP2_META_PATH) as file_handle:
        step2_meta = json.load(file_handle)
    return [entry["layer_index"] for entry in step2_meta["top_5_worst_layers"]]


def build_fingerprint_matrix(binarized_questions, layer_index, num_questions):
    """
    Transpose the routing data from question-major to expert-major layout.

    The binarized data is stored per question (each question has a dict of
    layer -> 128-bit vector). Step 4 needs the inverse: per expert, which
    questions activated it. Transposing here so clustering can treat each
    expert as a single num_questions-dim point without re-reading all questions
    per expert.
    """
    layer_key = str(layer_index)
    matrix = [[0] * num_questions for _ in range(EXPERTS_PER_LAYER)]

    for question_idx, question in enumerate(binarized_questions):
        binary_vector = question["binarized_routing"][layer_key]
        for expert_idx, is_selected in enumerate(binary_vector):
            if is_selected == 1:
                matrix[expert_idx][question_idx] = 1

    return matrix


def check_column_sums(matrix, num_questions):
    """
    Verify that exactly ROUTER_TOP_K experts are selected per question.

    Each column represents one question. If any column sums to something
    other than ROUTER_TOP_K, the binarization in Step 2 is wrong and
    the fingerprint matrix cannot be trusted for clustering.

    Returns a list of (question_index, actual_sum) for any failing columns.
    """
    errors = []
    for question_idx in range(num_questions):
        col_sum = sum(matrix[expert_idx][question_idx] for expert_idx in range(EXPERTS_PER_LAYER))
        if col_sum != ROUTER_TOP_K:
            errors.append((question_idx, col_sum))
    return errors


def compute_activation_stats(matrix):
    """
    Summarize how activations are distributed across the 128 experts.

    High zero_experts count indicates the router strongly prefers a small
    subset. Those zero-experts must be dropped before Jaccard clustering
    since Jaccard distance is undefined when both vectors are all-zero.
    """
    row_sums = [sum(row) for row in matrix]
    return {
        "avg_activations_per_expert": round(sum(row_sums) / len(row_sums), 1),
        "min_activations": min(row_sums),
        "max_activations": max(row_sums),
        "zero_experts": sum(1 for count in row_sums if count == 0),
    }


def process_layer(layer_index, binarized_questions, num_questions):
    print(f"\n--- Layer {layer_index} ---")

    matrix = build_fingerprint_matrix(binarized_questions, layer_index, num_questions)

    column_errors = check_column_sums(matrix, num_questions)
    if column_errors:
        print(f"  SANITY CHECK FAILED: {len(column_errors)} columns have wrong sums")
        for question_idx, col_sum in column_errors[:5]:
            print(f"    Column {question_idx}: sum = {col_sum} (expected {ROUTER_TOP_K})")
        return None

    print(f"  Sanity check passed: all {num_questions} columns sum to {ROUTER_TOP_K}")

    stats = compute_activation_stats(matrix)
    print(
        f"  Expert activations: avg={stats['avg_activations_per_expert']}, "
        f"min={stats['min_activations']}, max={stats['max_activations']}, "
        f"zero_experts={stats['zero_experts']}"
    )

    out_path = STEP3_OUTPUT_DIR / f"fingerprint_layer_{layer_index}.json"
    with open(out_path, "w") as file_handle:
        json.dump(matrix, file_handle)
    print(f"  Saved: {out_path}")

    return {
        "layer_index": layer_index,
        "matrix_shape": [EXPERTS_PER_LAYER, num_questions],
        "sanity_check_passed": True,
        **stats,
    }


def main():
    print("=" * 60)
    print("Step 3: Build Expert Fingerprint Matrices")
    print("=" * 60)

    step1_meta = load_step1_meta()
    num_questions = step1_meta["test_question_count"]

    binarized_questions = load_binarized_data()
    print(f"Loaded {len(binarized_questions)} binarized questions.")

    if len(binarized_questions) != num_questions:
        print(f"ERROR: expected {num_questions} questions, got {len(binarized_questions)}")
        return

    top_layers = load_top_layers()
    print(f"Top 5 worst layers: {top_layers}")

    STEP3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layer_results = []
    for layer_index in top_layers:
        result = process_layer(layer_index, binarized_questions, num_questions)
        if result:
            layer_results.append(result)

    step3_meta = {
        "layers_processed": top_layers,
        "matrix_shape": [EXPERTS_PER_LAYER, num_questions],
        "layer_results": layer_results,
    }
    meta_path = STEP3_OUTPUT_DIR / "step3_meta.json"
    with open(meta_path, "w") as file_handle:
        json.dump(step3_meta, file_handle, indent=2)
    print(f"\nSaved: {meta_path}")
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
