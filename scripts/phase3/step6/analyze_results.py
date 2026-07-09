"""
Step 6: Statistical analysis of ablation results.

For each targeted ablation run:
    1. McNemar's test vs baseline - checks if accuracy changes are real (p < 0.05)
    2. 95% confidence intervals on all accuracy numbers

For each layer count (1, 3, 5):
    3. Z-score comparing targeted moe_only vs 10 random trials
       A z > 2 means our layer scoring found specific signal, not just that
       the model tolerates any layer removal.

Also produces:
    - Per-category accuracy breakdown for best run vs baseline
    - Accuracy vs layers ablated plot
    - Category breakdown bar chart

Inputs:
    /workspace/results/ablation_results.json

Outputs (written to this directory):
    step6_meta.json
    plots/accuracy_vs_layers.png
    plots/category_breakdown.png
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_PATH = Path("/workspace/results/ablation_results.json")
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"

CATEGORIES = ["height", "position", "rotation", "ordering"]
Z_95 = 1.96


# --- Data loading ---

def load_results():
    with open(RESULTS_PATH) as file_handle:
        return json.load(file_handle)


def build_result_map(results_list):
    return {record["question_id"]: record["correct"] for record in results_list}


# --- Statistics ---

def mcnemar_test(baseline_map, ablation_map):
    """
    McNemar's chi-squared test on matched question pairs.

    b = baseline correct, ablation wrong (regression)
    c = baseline wrong, ablation correct (improvement)

    Uses continuity correction (|b-c| - 1)^2 to reduce Type I error at n=200.
    p-value from chi-squared df=1 CDF: P(X > x) = 1 - erf(sqrt(x/2)).
    No scipy needed since df=1 reduces to the error function.
    """
    b, c = 0, 0
    for qid, base_correct in baseline_map.items():
        ablation_correct = ablation_map.get(qid, False)
        if base_correct and not ablation_correct:
            b += 1
        elif not base_correct and ablation_correct:
            c += 1

    if b + c == 0:
        return {"b": 0, "c": 0, "chi_squared": 0.0, "p_value": 1.0, "significant": False}

    chi_squared = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - math.erf(math.sqrt(chi_squared / 2))

    return {
        "b": b,
        "c": c,
        "chi_squared": round(chi_squared, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def confidence_interval(accuracy, num_questions):
    """95% CI using normal approximation. Reliable for n >= 30."""
    margin = Z_95 * math.sqrt(accuracy * (1 - accuracy) / num_questions)
    return {
        "lower": round(max(0.0, accuracy - margin), 4),
        "upper": round(min(1.0, accuracy + margin), 4),
        "margin": round(margin, 4),
    }


def random_z_score(targeted_accuracy, random_accuracies):
    """
    Z-score of targeted ablation vs the 10 random control trials.

    Uses sample std (ddof=1) since the 10 trials are a sample, not the population.
    """
    num_trials = len(random_accuracies)
    mean = sum(random_accuracies) / num_trials
    variance = sum((value - mean) ** 2 for value in random_accuracies) / (num_trials - 1)
    std_dev = math.sqrt(variance) if variance > 1e-12 else 1e-8
    z_score = (targeted_accuracy - mean) / std_dev

    return {
        "random_mean": round(mean, 4),
        "random_std": round(std_dev, 4),
        "z_score": round(z_score, 4),
        "significant": z_score > 2.0,
    }


def category_accuracy(results_list):
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in results_list:
        cat = record["category"]
        counts[cat]["total"] += 1
        if record["correct"]:
            counts[cat]["correct"] += 1
    return {cat: round(counts[cat]["correct"] / counts[cat]["total"], 4) for cat in counts}


# --- Analysis ---

def analyze_targeted_runs(runs, baseline_map, n_questions):
    """McNemar + CI + category breakdown for every non-random run."""
    stats = []
    for run in runs:
        if run["run"].startswith("partC"):
            continue
        run_map = build_result_map(run["results"])
        stats.append({
            "run": run["run"],
            "ablation_mode": run["ablation_mode"],
            "layers_ablated": run["layers_ablated"],
            "accuracy": run["accuracy"],
            "ci": confidence_interval(run["accuracy"], n_questions),
            "mcnemar": mcnemar_test(baseline_map, run_map),
            "category_accuracy": category_accuracy(run["results"]),
        })
    return stats


def analyze_random_controls(runs, targeted_by_count):
    """
    For each layer count, z-score the targeted moe_only accuracy vs the 10 random trials.

    Random trials only used moe_only mode (no expert clustering was done for random layers),
    so this comparison is apples-to-apples for the moe_only targeted runs.
    """
    random_by_count = defaultdict(list)
    for run in runs:
        if not run["run"].startswith("partC"):
            continue
        parts = run["run"].split("_")
        count = int(parts[2].replace("layers", ""))
        random_by_count[count].append(run["accuracy"])

    results = {}
    for count, random_accuracies in sorted(random_by_count.items()):
        if count not in targeted_by_count:
            continue
        results[count] = {
            "targeted_accuracy": targeted_by_count[count],
            "random_trials": random_accuracies,
            "z_score_vs_random": random_z_score(targeted_by_count[count], random_accuracies),
        }
    return results


# --- Plots ---

def plot_accuracy_scaling(targeted_moe, targeted_expert, random_stats, baseline_acc, output_dir):
    """
    Accuracy vs number of layers ablated for targeted and random ablation.

    The shaded band shows random mean +/- 1 std. If targeted lines stay above
    the band, the layer scoring is adding value beyond random removal.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [0, 1, 3, 5]

    ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline ({baseline_acc:.1%})")

    moe_accs = [baseline_acc] + [targeted_moe[layer_count] for layer_count in [1, 3, 5]]
    ax.plot(counts, moe_accs, "b-o", linewidth=1.8, markersize=7, label="Targeted (moe_only)")

    if targeted_expert:
        expert_accs = [baseline_acc] + [targeted_expert[layer_count] for layer_count in [1, 3, 5]]
        ax.plot(counts, expert_accs, "g-s", linewidth=1.8, markersize=7, label="Targeted (expert)")

    rand_means = [baseline_acc] + [random_stats[layer_count]["z_score_vs_random"]["random_mean"] for layer_count in [1, 3, 5]]
    rand_stds = [0.0] + [random_stats[layer_count]["z_score_vs_random"]["random_std"] for layer_count in [1, 3, 5]]
    ax.plot(counts, rand_means, "r--^", linewidth=1.5, markersize=6, label="Random mean")
    ax.fill_between(
        counts,
        [mean_val - std_val for mean_val, std_val in zip(rand_means, rand_stds)],
        [mean_val + std_val for mean_val, std_val in zip(rand_means, rand_stds)],
        alpha=0.15, color="red", label="Random +/- 1 std",
    )

    ax.set_xlabel("Layers ablated")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Layers Ablated")
    ax.set_xticks(counts)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "accuracy_vs_layers.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_category_breakdown(baseline_run, best_run, output_dir):
    """
    Per-category accuracy for baseline vs best ablation run.

    If the improvement concentrates in specific categories, it suggests
    the ablated layers are category-specific rather than generally helpful.
    """
    base_cat = category_accuracy(baseline_run["results"])
    best_cat = category_accuracy(best_run["results"])

    x_positions = list(range(len(CATEGORIES)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([position_idx - width / 2 for position_idx in x_positions], [base_cat.get(cat, 0) for cat in CATEGORIES],
           width, label="Baseline", color="steelblue", alpha=0.8)
    ax.bar([position_idx + width / 2 for position_idx in x_positions], [best_cat.get(cat, 0) for cat in CATEGORIES],
           width, label=best_run["run"], color="seagreen", alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Category Accuracy: Baseline vs Best Run")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "category_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --- Main ---

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ablation_data = load_results()
    runs = ablation_data["runs"]
    n_questions = len(runs[0]["results"])

    baseline_run = next(run for run in runs if run["run"] == "baseline")
    baseline_map = build_result_map(baseline_run["results"])
    baseline_acc = baseline_run["accuracy"]

    print(f"Baseline: {baseline_acc:.1%} over {n_questions} questions")

    # McNemar + CI for all targeted runs
    targeted_stats = analyze_targeted_runs(runs, baseline_map, n_questions)

    print("\n=== Targeted runs (McNemar vs baseline) ===")
    for run_stat in targeted_stats:
        if run_stat["run"] == "baseline":
            continue
        mcnemar_result = run_stat["mcnemar"]
        ci = run_stat["ci"]
        print(
            f"{run_stat['run']}: {run_stat['accuracy']:.1%} "
            f"[{ci['lower']:.1%}, {ci['upper']:.1%}] "
            f"| +{mcnemar_result['c']} improvements, -{mcnemar_result['b']} regressions "
            f"| chi2={mcnemar_result['chi_squared']} p={mcnemar_result['p_value']} sig={mcnemar_result['significant']}"
        )

    # Extract targeted moe_only and expert accuracies by layer count
    def get_acc(run_name):
        candidate_run = next((run for run in runs if run["run"] == run_name), None)
        return candidate_run["accuracy"] if candidate_run else None

    targeted_moe = {layer_count: get_acc(f"partB_moe_only_{layer_count}layers") for layer_count in [1, 3, 5]}
    targeted_expert = {layer_count: get_acc(f"partB_expert_{layer_count}layers") for layer_count in [1, 3, 5]}
    targeted_expert = {layer_count: acc for layer_count, acc in targeted_expert.items() if acc is not None}

    # Z-score vs random controls
    random_stats = analyze_random_controls(runs, targeted_moe)

    print("\n=== Random control comparison (targeted moe_only vs random moe_only) ===")
    for count, count_stats in sorted(random_stats.items()):
        z_stats = count_stats["z_score_vs_random"]
        print(
            f"{count} layers: targeted={count_stats['targeted_accuracy']:.1%} "
            f"random_mean={z_stats['random_mean']:.1%} std={z_stats['random_std']:.1%} "
            f"z={z_stats['z_score']:.2f} sig={z_stats['significant']}"
        )

    # Plots
    plot_accuracy_scaling(targeted_moe, targeted_expert, random_stats, baseline_acc, PLOTS_DIR)

    best_run = next(run for run in runs if run["run"] == "partB_moe_only_3layers")
    plot_category_breakdown(baseline_run, best_run, PLOTS_DIR)

    # Save all stats
    step6_meta = {
        "baseline_accuracy": baseline_acc,
        "n_questions": n_questions,
        "targeted_run_stats": targeted_stats,
        "random_control_stats": {str(layer_count): count_stats for layer_count, count_stats in random_stats.items()},
        "targeted_moe_by_count": targeted_moe,
        "targeted_expert_by_count": targeted_expert,
    }

    meta_path = OUTPUT_DIR / "step6_meta.json"
    with open(meta_path, "w") as file_handle:
        json.dump(step6_meta, file_handle, indent=2)
    print(f"\nSaved: {meta_path}")


if __name__ == "__main__":
    main()
