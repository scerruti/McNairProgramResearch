"""
Step 1: Split the 400 Phase 2 questions into a test set (200) and holdout set (200).

Uses stratified sampling to ensure each split has roughly equal questions per category.
Computes the base error rate for the test set, which Step 2 uses as the expected
wrong-answer probability when computing z-scores for each expert.

Outputs:
    data/phase3/test_200.json      - 200 questions used for all experiments (Steps 2-5)
    data/phase3/holdout_200.json   - 200 questions reserved for final validation (Step 6)
    data/phase3/step1_meta.json    - metadata including base error rate and split config
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Fixing the random seed ensures the same split every time the script runs.
# Without this, re-running the script would produce a different test/holdout
# partition, making results from earlier runs incomparable.
RANDOM_SEED = 42

# The Phase 2 data contains router logit scores for all 128 experts across
# all 48 layers for each of the 400 LEGOLite questions, plus predictions
# and correctness labels.
PHASE2_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "phase2" / "runpod_second" / "results.json"
)

# Step 1 outputs go in data/phase3/step1/, keeping each step's
# outputs isolated so downstream scripts know exactly where to look.
STEP1_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "phase3" / "step1"
)

EXPECTED_TOTAL_QUESTIONS = 400
EXPECTED_LAYER_COUNT = 48
EXPECTED_EXPERTS_PER_LAYER = 128

# The four LEGOLite spatial reasoning categories.
EXPECTED_CATEGORIES = {"height", "rotation", "ordering", "position"}


def load_phase2_questions(filepath):
    """Load the raw Phase 2 results JSON from disk."""
    with open(filepath) as file_handle:
        phase2_questions = json.load(file_handle)
    return phase2_questions


def validate_question_structure(phase2_questions):
    """
    Verify every question has the fields needed for Phase 3:
      - 'correct': boolean label for whether the model answered correctly
      - 'category': one of the four LEGOLite categories
      - 'visual_routing': dict with 48 layers, each containing 128 expert logit scores

    Returns a list of error strings. An empty list means all questions passed.
    The validation checks every question exhaustively rather than failing fast,
    so you can see all problems at once instead of fixing them one at a time.
    """
    validation_errors = []

    for question_index, question in enumerate(phase2_questions):
        question_id = question.get("question_id", f"index_{question_index}")

        if "correct" not in question:
            validation_errors.append(
                f"Question '{question_id}': missing 'correct' field (needed for error rate)"
            )

        if "category" not in question:
            validation_errors.append(
                f"Question '{question_id}': missing 'category' field (needed for stratified split)"
            )
        elif question["category"] not in EXPECTED_CATEGORIES:
            validation_errors.append(
                f"Question '{question_id}': unexpected category '{question['category']}', "
                f"expected one of {EXPECTED_CATEGORIES}"
            )

        if "visual_routing" not in question:
            validation_errors.append(
                f"Question '{question_id}': missing 'visual_routing' field (needed for expert analysis)"
            )
            continue

        visual_routing = question["visual_routing"]

        layer_count = len(visual_routing)
        if layer_count != EXPECTED_LAYER_COUNT:
            validation_errors.append(
                f"Question '{question_id}': expected {EXPECTED_LAYER_COUNT} layers "
                f"in visual_routing, got {layer_count}"
            )

        for layer_index, expert_logits in visual_routing.items():
            expert_count = len(expert_logits)
            if expert_count != EXPECTED_EXPERTS_PER_LAYER:
                validation_errors.append(
                    f"Question '{question_id}', layer {layer_index}: expected "
                    f"{EXPECTED_EXPERTS_PER_LAYER} expert scores, got {expert_count}"
                )

    return validation_errors


def group_questions_by_category(phase2_questions):
    """
    Organize questions into a dict keyed by category name.

    This is the first step of stratified sampling: we need to split within
    each category independently so that both halves end up with roughly
    equal category proportions.
    """
    questions_by_category = defaultdict(list)
    for question in phase2_questions:
        category_name = question["category"]
        questions_by_category[category_name].append(question)
    return dict(questions_by_category)


def perform_stratified_split(questions_by_category):
    """
    Split each category's questions in half, then combine the halves
    into test and holdout sets.

    Stratified splitting guarantees each set gets ~50 questions per category.
    A naive random split across all 400 questions could produce imbalanced
    sets (e.g., 70 rotation in test, 30 in holdout), which would make
    per-category accuracy comparisons between test and holdout unreliable.
    """
    random.seed(RANDOM_SEED)

    test_set_questions = []
    holdout_set_questions = []

    for category_name in sorted(questions_by_category.keys()):
        category_questions = questions_by_category[category_name]
        random.shuffle(category_questions)

        # Integer division: if a category has 100 questions, each half gets 50.
        # If a category has 101, test gets 50 and holdout gets 51. The imbalance
        # is at most 1 question per category, which is acceptable.
        split_point = len(category_questions) // 2
        test_set_questions.extend(category_questions[:split_point])
        holdout_set_questions.extend(category_questions[split_point:])

    return test_set_questions, holdout_set_questions


def compute_split_statistics(split_name, questions):
    """
    Compute and print category counts, accuracy, and error rate for a split.

    Returns the error rate as a float, which becomes the base_error_rate
    when called on the test set. The base_error_rate is the probability
    that any randomly selected question is wrong, used in Step 2 to set
    the expected value in the z-score formula:
        expected_wrong = n * base_error_rate
    """
    category_statistics = {}
    total_correct = 0
    total_wrong = 0

    for category_name in sorted(EXPECTED_CATEGORIES):
        category_questions = [q for q in questions if q["category"] == category_name]
        category_correct = sum(1 for q in category_questions if q["correct"])
        category_wrong = len(category_questions) - category_correct

        category_statistics[category_name] = {
            "total": len(category_questions),
            "correct": category_correct,
            "wrong": category_wrong,
        }
        total_correct += category_correct
        total_wrong += category_wrong

    total_questions = total_correct + total_wrong

    # The error rate is total_wrong / total_questions. For the test set,
    # this becomes the base_error_rate used throughout Steps 2-6.
    # A base_error_rate of 0.75 means the model gets 75% of questions wrong,
    # so any expert's wrong-answer rate needs to be compared against 0.75, not 0.50.
    error_rate = total_wrong / total_questions

    print(f"\n{split_name} ({total_questions} questions):")
    for category_name, stats in sorted(category_statistics.items()):
        print(
            f"  {category_name}: {stats['total']} questions "
            f"({stats['correct']} correct, {stats['wrong']} wrong)"
        )
    print(f"  Overall: {total_correct} correct, {total_wrong} wrong")
    print(f"  Error rate: {error_rate:.4f} ({error_rate:.2%})")

    return error_rate


def save_json(data, filepath):
    """Write data to a JSON file with readable indentation."""
    with open(filepath, "w") as file_handle:
        json.dump(data, file_handle, indent=2)
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Step 1: Data Splitting")
    print("=" * 60)

    # --- Load ---
    print(f"\nLoading Phase 2 data from:\n  {PHASE2_RESULTS_PATH}")
    phase2_questions = load_phase2_questions(PHASE2_RESULTS_PATH)
    print(f"Loaded {len(phase2_questions)} questions.")

    if len(phase2_questions) != EXPECTED_TOTAL_QUESTIONS:
        print(
            f"WARNING: expected {EXPECTED_TOTAL_QUESTIONS} questions, "
            f"got {len(phase2_questions)}"
        )

    # --- Validate ---
    # Check every question before doing any work. If the Phase 2 data
    # is malformed (missing fields, wrong layer count, etc.), we want to
    # know now rather than getting cryptic errors in Step 2.
    print("\n--- Validation ---")
    validation_errors = validate_question_structure(phase2_questions)

    if validation_errors:
        print(f"Validation FAILED with {len(validation_errors)} errors:")
        for error_message in validation_errors[:20]:
            print(f"  {error_message}")
        if len(validation_errors) > 20:
            print(f"  ... and {len(validation_errors) - 20} more errors")
        return
    else:
        print(f"All {len(phase2_questions)} questions passed validation.")

    # --- Split ---
    print("\n--- Stratified Split ---")
    questions_by_category = group_questions_by_category(phase2_questions)
    test_set_questions, holdout_set_questions = perform_stratified_split(
        questions_by_category
    )

    test_error_rate = compute_split_statistics("Test set", test_set_questions)
    holdout_error_rate = compute_split_statistics("Holdout set", holdout_set_questions)

    # --- Save ---
    print("\n--- Saving Output ---")
    STEP1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_output_path = STEP1_OUTPUT_DIR / "test_200.json"
    holdout_output_path = STEP1_OUTPUT_DIR / "holdout_200.json"

    save_json(test_set_questions, test_output_path)
    save_json(holdout_set_questions, holdout_output_path)

    # Save metadata so downstream scripts can read the base error rate
    # and split configuration without re-running this script.
    split_metadata = {
        "random_seed": RANDOM_SEED,
        "source_data_path": str(PHASE2_RESULTS_PATH),
        "test_question_count": len(test_set_questions),
        "holdout_question_count": len(holdout_set_questions),
        "test_base_error_rate": test_error_rate,
        "holdout_error_rate": holdout_error_rate,
    }
    metadata_output_path = STEP1_OUTPUT_DIR / "step1_meta.json"
    save_json(split_metadata, metadata_output_path)

    # --- Summary ---
    print(f"\n--- Base Error Rate (Test Set) ---")
    print(f"base_error_rate = {test_error_rate:.4f}")
    print("This value is used in Step 2 as the expected wrong-answer")
    print("probability when computing z-scores for each expert.")


if __name__ == "__main__":
    main()
