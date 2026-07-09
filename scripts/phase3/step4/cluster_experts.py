"""
Step 4: Cluster experts in the top 5 worst layers using k-medoids with Jaccard distance.

For each layer:
  1. Drop zero experts (all-zero rows) - Jaccard is undefined for them
  2. Compute pairwise Jaccard distance matrix for the active experts
  3. Run k-medoids with k = 2, 4, 6, 8 and pick the best k by silhouette score
  4. For each cluster, compute its wrong-answer rate vs the 76.5% baseline
  5. Flag clusters above the baseline as bad and record their expert indices for Step 5

Inputs:
    data/phase3/step3/fingerprint_layer_XX.json  - 128x200 binary matrices
    data/phase3/step2/binarized_routing.json      - correctness labels per question
    data/phase3/step2/step2_meta.json             - top 5 worst layers

Outputs:
    data/phase3/step4/cluster_results_layer_XX.json  - cluster assignments and stats
    data/phase3/step4/silhouette_layer_XX.png        - silhouette score vs k plot
    data/phase3/step4/step4_meta.json                - summary and bad expert lists
"""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STEP2_DATA_DIR = PROJECT_ROOT / "data" / "phase3" / "step2"
STEP3_DATA_DIR = PROJECT_ROOT / "data" / "phase3" / "step3"
STEP4_OUTPUT_DIR = PROJECT_ROOT / "data" / "phase3" / "step4"

BINARIZED_PATH = STEP2_DATA_DIR / "binarized_routing.json"
STEP2_META_PATH = STEP2_DATA_DIR / "step2_meta.json"

# k=1 is trivial (everything in one cluster), and k values beyond the number
# of active experts are skipped at runtime.
K_VALUES_TO_TRY = [2, 4, 6, 8]

# PAM converges quickly in practice; 100 iterations is a safe ceiling.
KMEDOIDS_MAX_ITER = 100

RANDOM_SEED = 42


# --- Data loading ---

def load_fingerprint(layer_index):
    path = STEP3_DATA_DIR / f"fingerprint_layer_{layer_index}.json"
    with open(path) as file_handle:
        return json.load(file_handle)


def load_correctness_labels():
    with open(BINARIZED_PATH) as file_handle:
        questions = json.load(file_handle)
    # Only the correctness label is needed here; keeping question metadata
    # out of this module avoids coupling to the binarized_routing schema.
    return [question["correct"] for question in questions]


def load_step2_meta():
    with open(STEP2_META_PATH) as file_handle:
        return json.load(file_handle)


def load_top_layers(step2_meta):
    return [entry["layer_index"] for entry in step2_meta["top_5_worst_layers"]]


# --- Expert filtering ---

def filter_active_experts(matrix):
    """
    Remove experts that were never selected across all 200 questions.

    Jaccard distance is undefined when both vectors are all-zero (union = 0),
    so inactive experts must be excluded before computing the distance matrix.
    An expert that never fires carries no routing signal and tells us
    nothing about which questions the layer handles badly.

    Returns (active_rows, original_indices) so results can be mapped back
    to the original 0-127 expert numbering for Step 5 ablation.
    """
    active_rows = []
    original_indices = []
    for expert_idx, row in enumerate(matrix):
        if any(v == 1 for v in row):
            active_rows.append(row)
            original_indices.append(expert_idx)
    return active_rows, original_indices


# --- Distance computation ---

def compute_jaccard_distance_matrix(active_rows):
    """
    Pairwise Jaccard distances for all active experts.

    Jaccard distance ignores shared zeros, which is why it fits here better
    than cosine or Euclidean: two experts that both happen to be inactive on
    the same 150 questions should not be considered similar. Shared inactivity
    tells us nothing about their routing pattern.
    Only shared activations (both selected on the same question) indicate a
    genuine similarity in routing behavior.
    """
    bool_matrix = np.array(active_rows, dtype=bool)
    num_active_experts = len(bool_matrix)
    dist_matrix = np.zeros((num_active_experts, num_active_experts))

    for expert_i in range(num_active_experts):
        for expert_j in range(expert_i + 1, num_active_experts):
            intersection = np.sum(bool_matrix[expert_i] & bool_matrix[expert_j])
            union = np.sum(bool_matrix[expert_i] | bool_matrix[expert_j])
            distance = 1.0 - intersection / union if union > 0 else 0.0
            dist_matrix[expert_i, expert_j] = distance
            dist_matrix[expert_j, expert_i] = distance

    return dist_matrix


# --- Clustering ---

def run_kmedoids(dist_matrix, k, seed=RANDOM_SEED):
    """
    K-medoids (PAM) clustering on a precomputed distance matrix.

    Medoids are actual data points rather than fractional averages, which
    matters because Step 5 needs real expert indices to disable. A k-means
    centroid has no corresponding expert; a k-medoids medoid does.

    PAM alternates between assigning each point to its nearest medoid and
    swapping each medoid for the cluster member that minimizes total intra-
    cluster distance. It converges when no swap improves the solution.
    """
    num_active_experts = len(dist_matrix)
    rng = random.Random(seed)
    medoids = rng.sample(range(num_active_experts), k)

    for _ in range(KMEDOIDS_MAX_ITER):
        labels = [
            min(range(k), key=lambda cluster_idx: dist_matrix[point_idx][medoids[cluster_idx]])
            for point_idx in range(num_active_experts)
        ]

        new_medoids = []
        for cluster_id in range(k):
            members = [member_idx for member_idx, label in enumerate(labels) if label == cluster_id]
            if not members:
                new_medoids.append(medoids[cluster_id])
            else:
                best = min(members, key=lambda candidate_idx: sum(dist_matrix[candidate_idx][other_idx] for other_idx in members))
                new_medoids.append(best)

        if new_medoids == medoids:
            break
        medoids = new_medoids

    return labels, medoids


def cluster_all_k_values(dist_matrix, n_active):
    """
    Run k-medoids for each candidate k and collect silhouette scores.

    Silhouette score requires at least 2 distinct clusters, so k values that
    collapse to a single cluster (can happen when experts are very similar)
    are skipped. k values >= n_active are also skipped because you cannot
    have more clusters than data points.
    """
    results = {}
    for k in K_VALUES_TO_TRY:
        if k >= n_active:
            print(f"  k={k}: skipped (only {n_active} active experts)")
            continue

        labels, medoids = run_kmedoids(dist_matrix, k)

        if len(set(labels)) < 2:
            print(f"  k={k}: collapsed to 1 cluster, skipping silhouette")
            continue

        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
        results[k] = {"labels": labels, "medoids": medoids, "silhouette": round(sil, 4)}

    return results


def pick_best_k(k_results):
    """
    Select the k with the highest silhouette score.

    Higher silhouette = tighter within-cluster distances relative to between-
    cluster distances. This is the standard unsupervised selection criterion
    when the true number of clusters is unknown.
    """
    return max(k_results, key=lambda k: k_results[k]["silhouette"])


# --- Cluster analysis ---

def score_cluster(member_rows, correctness_labels, base_error_rate):
    """
    Compute wrong-answer rate for a single cluster.

    Counts every (expert, question) activation pair where the expert is in
    this cluster and the question was answered wrong. Divides by total
    activations to get the cluster's wrong-answer rate. If this rate exceeds
    base_error_rate, the cluster's experts are firing on bad questions more
    than chance predicts.
    """
    total = 0
    wrong = 0
    for question_idx, correct in enumerate(correctness_labels):
        for row in member_rows:
            if row[question_idx] == 1:
                total += 1
                if not correct:
                    wrong += 1

    wrong_rate = wrong / total if total > 0 else 0.0
    return {
        "expert_count": len(member_rows),
        "total_activations": total,
        "wrong_activations": wrong,
        "wrong_rate": round(wrong_rate, 4),
        "above_baseline": wrong_rate > base_error_rate,
    }


def score_all_clusters(labels, k, active_rows, correctness_labels, base_error_rate):
    cluster_stats = []
    for cluster_id in range(k):
        member_rows = [active_rows[member_idx] for member_idx, label in enumerate(labels) if label == cluster_id]
        if not member_rows:
            cluster_stats.append(None)
        else:
            cluster_stats.append(score_cluster(member_rows, correctness_labels, base_error_rate))
    return cluster_stats


def identify_bad_experts(labels, cluster_stats, active_indices):
    """
    Collect original expert indices from clusters whose wrong-rate exceeds the baseline.

    These are passed to Step 5 for targeted ablation. Only clusters with
    above_baseline=True are included. Clusters at or below the baseline
    are not doing worse than the model overall and are not candidates for ablation.
    """
    bad_expert_indices = []
    for cluster_id, stats in enumerate(cluster_stats):
        if stats and stats["above_baseline"]:
            members = [active_indices[member_idx] for member_idx, label in enumerate(labels) if label == cluster_id]
            bad_expert_indices.extend(members)
    return bad_expert_indices


# --- Output ---

def plot_silhouette_curve(k_results, best_k, layer_index):
    ks = sorted(k_results.keys())
    sils = [k_results[k]["silhouette"] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sils, marker="o", color="#1976d2")
    ax.axvline(best_k, color="#d32f2f", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Layer {layer_index}: Silhouette vs k")
    ax.legend()
    fig.tight_layout()

    plot_path = STEP4_OUTPUT_DIR / f"silhouette_layer_{layer_index}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")


def plot_mds_clusters(dist_matrix, labels, cluster_stats, active_indices, layer_index, best_k):
    """
    Project experts into 2D using MDS and color by cluster assignment.

    MDS preserves pairwise Jaccard distances as faithfully as possible in 2D,
    so clusters that are visually separated here are genuinely far apart in the
    original 200-dim activation space. If bad-cluster experts (marked with X)
    overlap with the rest, the clustering has not found a real structural split
    and the weak z-scores from Step 2 likely reflect genuine noise rather than
    a hidden pattern.
    """
    coords = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        random_state=RANDOM_SEED,
        normalized_stress="auto",
    ).fit_transform(dist_matrix)

    bad_expert_set = set(
        active_indices[member_idx]
        for member_idx, label in enumerate(labels)
        if cluster_stats[label] and cluster_stats[label]["above_baseline"]
    )

    cluster_ids = sorted(set(labels))
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in cluster_ids:
        member_positions = [(coords[member_idx, 0], coords[member_idx, 1]) for member_idx, label in enumerate(labels) if label == cluster_id]
        xs = [position[0] for position in member_positions]
        ys = [position[1] for position in member_positions]
        color = colors[cluster_id % len(colors)]

        stats = cluster_stats[cluster_id]
        wrong_rate_str = f"{stats['wrong_rate']:.2f}" if stats else "n/a"
        cluster_label = f"Cluster {cluster_id} (wr={wrong_rate_str})"

        ax.scatter(xs, ys, c=[color], label=cluster_label, s=60, zorder=2)

        # Mark bad-cluster experts with X so they stand out without a separate legend entry.
        bad_xs = [coords[member_idx, 0] for member_idx, label in enumerate(labels) if label == cluster_id and active_indices[member_idx] in bad_expert_set]
        bad_ys = [coords[member_idx, 1] for member_idx, label in enumerate(labels) if label == cluster_id and active_indices[member_idx] in bad_expert_set]
        if bad_xs:
            ax.scatter(bad_xs, bad_ys, c="black", marker="x", s=80, linewidths=1.5, zorder=3)

    ax.set_title(f"Layer {layer_index}: MDS of Expert Activations (k={best_k}, X = above baseline)")
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    plot_path = STEP4_OUTPUT_DIR / f"mds_layer_{layer_index}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")


def save_layer_results(layer_index, active_indices, best_k, k_results, bad_expert_indices):
    result = {
        "layer_index": layer_index,
        "active_expert_count": len(active_indices),
        "active_expert_indices": active_indices,
        "best_k": best_k,
        "best_silhouette": k_results[best_k]["silhouette"],
        "bad_expert_indices": bad_expert_indices,
        "all_k_results": {
            str(k): {
                "silhouette": cluster_result["silhouette"],
                "medoid_expert_indices": [active_indices[medoid_idx] for medoid_idx in cluster_result["medoids"]],
                "cluster_stats": cluster_result["cluster_stats"],
            }
            for k, cluster_result in k_results.items()
        },
    }
    out_path = STEP4_OUTPUT_DIR / f"cluster_results_layer_{layer_index}.json"
    with open(out_path, "w") as file_handle:
        json.dump(result, file_handle, indent=2)
    print(f"  Saved: {out_path}")
    return result


# --- Per-layer orchestration ---

def process_layer(layer_index, correctness_labels, base_error_rate):
    print(f"\n{'='*50}")
    print(f"Layer {layer_index}")
    print(f"{'='*50}")

    matrix = load_fingerprint(layer_index)
    active_rows, active_indices = filter_active_experts(matrix)
    n_active = len(active_rows)
    print(f"Active experts: {n_active} / {len(matrix)}")

    if n_active < 2:
        print("Not enough active experts to cluster. Skipping.")
        return None

    print("Computing Jaccard distance matrix...")
    dist_matrix = compute_jaccard_distance_matrix(active_rows)

    k_results = cluster_all_k_values(dist_matrix, n_active)
    if not k_results:
        print("No valid clustering found.")
        return None

    best_k = pick_best_k(k_results)

    # Score all clusters for every k so the output file has the full picture,
    # but only use the best_k results for the bad-expert list.
    for k, result in k_results.items():
        stats = score_all_clusters(result["labels"], k, active_rows, correctness_labels, base_error_rate)
        result["cluster_stats"] = stats
        silhouette = result["silhouette"]
        print(f"  k={k}: silhouette={silhouette:.4f}")
        for cluster_id, cluster_stat in enumerate(stats):
            if cluster_stat:
                flag = " <<< BAD" if cluster_stat["above_baseline"] else ""
                print(f"    cluster {cluster_id}: {cluster_stat['expert_count']} experts, wrong_rate={cluster_stat['wrong_rate']:.3f}{flag}")

    best_labels = k_results[best_k]["labels"]
    best_cluster_stats = k_results[best_k]["cluster_stats"]
    bad_expert_indices = identify_bad_experts(best_labels, best_cluster_stats, active_indices)

    print(f"\nBest k: {best_k} (silhouette={k_results[best_k]['silhouette']:.4f})")
    print(f"Bad experts: {bad_expert_indices}")

    plot_silhouette_curve(k_results, best_k, layer_index)
    plot_mds_clusters(dist_matrix, best_labels, best_cluster_stats, active_indices, layer_index, best_k)
    return save_layer_results(layer_index, active_indices, best_k, k_results, bad_expert_indices)


# --- Entry point ---

def main():
    print("=" * 60)
    print("Step 4: Cluster Experts (K-Medoids + Jaccard)")
    print("=" * 60)

    step2_meta = load_step2_meta()
    base_error_rate = step2_meta["base_error_rate"]
    top_layers = load_top_layers(step2_meta)

    correctness_labels = load_correctness_labels()
    print(f"Base error rate: {base_error_rate}")
    print(f"Layers to process: {top_layers}")

    STEP4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for layer_index in top_layers:
        result = process_layer(layer_index, correctness_labels, base_error_rate)
        if result:
            all_results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    step4_meta = {"layers": []}
    for layer_result in all_results:
        print(
            f"Layer {layer_result['layer_index']}: best_k={layer_result['best_k']}, "
            f"silhouette={layer_result['best_silhouette']:.4f}, "
            f"bad_experts={layer_result['bad_expert_indices']}"
        )
        step4_meta["layers"].append({
            "layer_index": layer_result["layer_index"],
            "best_k": layer_result["best_k"],
            "best_silhouette": layer_result["best_silhouette"],
            "bad_expert_indices": layer_result["bad_expert_indices"],
        })

    meta_path = STEP4_OUTPUT_DIR / "step4_meta.json"
    with open(meta_path, "w") as file_handle:
        json.dump(step4_meta, file_handle, indent=2)
    print(f"\nSaved: {meta_path}")
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
