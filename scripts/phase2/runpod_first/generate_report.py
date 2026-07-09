"""
generate_report.py
==================
Generates a PDF report of the LEGOLite MoE expert analysis results.
Output: /workspace/lego_lite_analysis/report.pdf
"""

import json
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path("/workspace/lego_lite_analysis")
RESULTS_JSON = ANALYSIS_DIR / "results.json"
RANK_CSV     = ANALYSIS_DIR / "expert_success_rates.csv"
BOARD_CSV    = ANALYSIS_DIR / "spatial_expert_leaderboard.csv"
OUTPUT_PDF   = ANALYSIS_DIR / "report.pdf"

CATEGORIES   = ["height", "position", "rotation", "ordering"]
CAT_COLORS   = {"height": "#4C72B0", "position": "#55A868",
                "rotation": "#C44E52", "ordering": "#DD8452"}
MODEL_NAME   = "Qwen3-VL-30B-A3B-Instruct"
BENCHMARK    = "LEGOLite (400 questions)"

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "grid.color":        "white",
    "grid.linewidth":    1.2,
})

DARK   = "#1A1A2E"
ACCENT = "#4C72B0"
LIGHT  = "#F8F9FA"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def page_header(fig, title, subtitle=""):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=DARK)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top",
                 fontsize=9, color="#555555")


def _add_table(ax, df, col_labels, col_widths, row_colors=None, fontsize=8):
    ax.axis("off")
    n_rows, n_cols = len(df), len(col_labels)
    cell_text = df.values.tolist()
    colors = row_colors if row_colors else [["white"] * n_cols] * n_rows
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        cellColours=colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#DDDDDD")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading results…")
with open(RESULTS_JSON) as results_file:
    results = json.load(results_file)

df_rank  = pd.read_csv(RANK_CSV)
df_board = pd.read_csv(BOARD_CSV)

category_accuracy = {}
for cat in CATEGORIES:
    category_results = [record for record in results if record["category"] == cat]
    category_accuracy[cat] = sum(record["correct"] for record in category_results) / len(category_results) if category_results else 0.0
overall_acc = sum(record["correct"] for record in results) / len(results)

# Heatmap arrays  [num_layers × num_experts]
NUM_LAYERS  = 48
NUM_EXPERTS = 128

category_heatmaps = {}
for cat in CATEGORIES:
    heatmap_sum = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    num_category_questions = 0
    for record in results:
        if record["category"] != cat:
            continue
        num_category_questions += 1
        for layer_idx_str, weights in record["visual_routing"].items():
            heatmap_sum[int(layer_idx_str)] += np.asarray(weights)
    category_heatmaps[cat] = heatmap_sum / num_category_questions if num_category_questions > 0 else heatmap_sum

height_rotation_diff = category_heatmaps["height"] - category_heatmaps["rotation"]

# ─────────────────────────────────────────────────────────────────────────────
# Page 1 - Cover / Summary
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Dark header band
    ax.add_patch(FancyBboxPatch((0, 0.78), 1, 0.22, linewidth=0,
                                facecolor=DARK, boxstyle="square,pad=0"))
    fig.text(0.5, 0.955, "MoE Expert Routing Analysis", ha="center",
             fontsize=22, fontweight="bold", color="white")
    fig.text(0.5, 0.895, MODEL_NAME, ha="center",
             fontsize=13, color="#AAAACC")
    fig.text(0.5, 0.855, BENCHMARK, ha="center",
             fontsize=11, color="#AAAACC")

    # Accuracy summary cards
    card_w, card_h = 0.18, 0.13
    xs = [0.07, 0.295, 0.52, 0.745]
    ys = 0.60
    for i, cat in enumerate(CATEGORIES):
        card_x = xs[i]
        ax.add_patch(FancyBboxPatch((card_x, ys), card_w, card_h, linewidth=1.5,
                                    edgecolor=CAT_COLORS[cat],
                                    facecolor=LIGHT,
                                    boxstyle="round,pad=0.01"))
        fig.text(card_x + card_w/2, ys + card_h - 0.025, cat.upper(),
                 ha="center", fontsize=9, fontweight="bold",
                 color=CAT_COLORS[cat])
        fig.text(card_x + card_w/2, ys + card_h/2 - 0.008,
                 f"{category_accuracy[cat]:.1%}", ha="center",
                 fontsize=20, fontweight="bold", color=DARK)
        fig.text(card_x + card_w/2, ys + 0.015, "accuracy",
                 ha="center", fontsize=8, color="#777777")

    # Overall
    ax.add_patch(FancyBboxPatch((0.35, 0.46), 0.30, 0.10, linewidth=2,
                                edgecolor=ACCENT, facecolor=LIGHT,
                                boxstyle="round,pad=0.01"))
    fig.text(0.5, 0.535, "OVERALL ACCURACY", ha="center",
             fontsize=9, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.488, f"{overall_acc:.1%}", ha="center",
             fontsize=22, fontweight="bold", color=DARK)

    # Key findings box
    findings = [
        f"• {NUM_LAYERS} MoE layers × {NUM_EXPERTS} experts × Top-8 routing analysed",
        "• 400 LEGOLite questions (100 per spatial category)",
        "• Router activations captured for visual tokens only",
        "• Top node: Layer 17 Expert 36  (global Δ = +0.0100)",
        "• Ordering suspects: L4_E13, L2_E37, L10_E3 (negative Δ)",
        "• Height vs Rotation show distinct mid-layer expert clusters",
    ]
    fig.text(0.07, 0.415, "Key Findings", fontsize=11,
             fontweight="bold", color=DARK)
    for j, line in enumerate(findings):
        fig.text(0.07, 0.375 - j * 0.033, line, fontsize=9, color="#333333")

    # Footer
    fig.text(0.5, 0.03, "Generated from /workspace/lego_lite_analysis/",
             ha="center", fontsize=8, color="#999999")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 1: cover")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 - Accuracy breakdown + expert activation bar charts
# ─────────────────────────────────────────────────────────────────────────────

def page_accuracy(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Accuracy & Per-Category Expert Activation",
                "How often the model answers correctly, and which layers are most active per category")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.55, wspace=0.35)

    # ── Bar chart: category accuracy ──────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, :])
    cats   = CATEGORIES + ["overall"]
    vals   = [category_accuracy[c] for c in CATEGORIES] + [overall_acc]
    colors = [CAT_COLORS[c] for c in CATEGORIES] + [DARK]
    bars   = ax_acc.bar(cats, vals, color=colors, width=0.55, zorder=3)
    ax_acc.set_ylim(0, 0.55)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy by Category")
    ax_acc.axhline(0.25, color="#AAAAAA", linestyle="--", linewidth=1,
                   label="random (4-choice)")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", zorder=0)
    for bar, val in zip(bars, vals):
        ax_acc.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")

    # ── Per-category layer activation profiles ────────────────────────────────
    for i, cat in enumerate(CATEGORIES):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row + 1, col])
        layer_means = category_heatmaps[cat].mean(axis=1)   # avg over 128 experts per layer
        ax.fill_between(range(NUM_LAYERS), layer_means,
                        color=CAT_COLORS[cat], alpha=0.35)
        ax.plot(range(NUM_LAYERS), layer_means,
                color=CAT_COLORS[cat], linewidth=1.4)
        peak_layer = int(np.argmax(layer_means))
        ax.axvline(peak_layer, color=CAT_COLORS[cat], linestyle=":",
                   linewidth=1.2, alpha=0.8)
        ax.text(peak_layer + 0.5, layer_means.max() * 0.97,
                f"L{peak_layer}", fontsize=7, color=CAT_COLORS[cat])
        ax.set_title(f"{cat.upper()} - layer activation profile")
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Mean routing weight", fontsize=8)
        ax.grid(axis="y")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 2: accuracy & activation profiles")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 - Global expert leaderboard table
# ─────────────────────────────────────────────────────────────────────────────

def page_leaderboard(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Top-30 Spatial Reasoning Experts - Global Leaderboard",
                "Ranked by success_delta = mean_activation_correct − mean_activation_incorrect")

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax.axis("off")

    board = df_board.head(30).copy()
    board["rank"] = range(1, len(board) + 1)
    display = board[["rank", "expert_label",
                      "mean_activation_correct",
                      "mean_activation_incorrect",
                      "success_delta"]].copy()
    display["mean_activation_correct"]   = display["mean_activation_correct"].map("{:.5f}".format)
    display["mean_activation_incorrect"] = display["mean_activation_incorrect"].map("{:.5f}".format)
    display["success_delta"]             = display["success_delta"].map("{:+.5f}".format)

    col_labels = ["Rank", "Expert", "Act (correct)", "Act (incorrect)", "Success Δ"]
    col_widths = [0.06, 0.15, 0.18, 0.20, 0.14]

    # Alternate row shading
    row_colors = []
    for row_idx in range(len(display)):
        shade = "#EEF2FF" if row_idx % 2 == 0 else "white"
        row_colors.append([shade] * 5)

    _add_table(ax, display, col_labels, col_widths, row_colors, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 3: leaderboard table")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 - Category specialist tables (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def page_specialists(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Category Specialist Experts",
                "Top-10 experts per spatial category, ranked by category-specific success Δ")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.88, bottom=0.04, hspace=0.12, wspace=0.1)

    for i, cat in enumerate(CATEGORIES):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"{cat.upper()} specialists", color=CAT_COLORS[cat],
                     fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

        delta_col = f"delta_{cat}"
        specialist_experts = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
               .nlargest(10, delta_col)
               [["expert_label", "mean_activation_correct", delta_col]]
               .copy())
        specialist_experts["mean_activation_correct"] = specialist_experts["mean_activation_correct"].map("{:.5f}".format)
        specialist_experts[delta_col]                 = specialist_experts[delta_col].map("{:+.5f}".format)

        col_labels = ["Expert", "Act (correct)", f"Δ {cat}"]
        col_widths = [0.18, 0.20, 0.18]
        row_colors = []
        for row_idx in range(len(specialist_experts)):
            shade = "#EEF2FF" if row_idx % 2 == 0 else "white"
            row_colors.append([shade] * 3)
        _add_table(ax, specialist_experts, col_labels, col_widths, row_colors, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 4: category specialists")


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 - Ordering failure diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def page_ordering(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Ordering Task - Expert Failure Diagnosis",
                "Experts that activate heavily on ordering tasks but correlate with WRONG answers")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.45, wspace=0.35)

    # ── Good ordering experts (top) ───────────────────────────────────────────
    ax_good = fig.add_subplot(gs[0, :])
    good = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
            .nlargest(10, "delta_ordering")
            [["expert_label", "delta_ordering", "mean_activation_correct"]]
            .copy())
    ax_good.barh(good["expert_label"], good["delta_ordering"],
                 color="#2ecc71", edgecolor="white", height=0.6)
    ax_good.set_xlabel("delta_ordering  (positive = fires more on CORRECT answers)")
    ax_good.set_title("Top Experts Correlated With Correct Ordering Answers")
    ax_good.axvline(0, color=DARK, linewidth=0.8)
    ax_good.grid(axis="x", zorder=0)
    for bar, val in zip(ax_good.patches, good["delta_ordering"]):
        ax_good.text(val + 0.0003, bar.get_y() + bar.get_height()/2,
                     f"{val:+.4f}", va="center", fontsize=7.5)

    # ── Suspect ordering experts (bottom) ─────────────────────────────────────
    ax_bad = fig.add_subplot(gs[1, :])
    suspects = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
                .nsmallest(10, "delta_ordering")
                [["expert_label", "delta_ordering", "mean_activation_correct"]]
                .copy())
    ax_bad.barh(suspects["expert_label"], suspects["delta_ordering"],
                color="#e74c3c", edgecolor="white", height=0.6)
    ax_bad.set_xlabel("delta_ordering  (negative = fires more on WRONG answers)")
    ax_bad.set_title("Suspect Experts - Active on Ordering Failures")
    ax_bad.axvline(0, color=DARK, linewidth=0.8)
    ax_bad.grid(axis="x", zorder=0)
    for bar, val in zip(ax_bad.patches, suspects["delta_ordering"]):
        ax_bad.text(val - 0.0003, bar.get_y() + bar.get_height()/2,
                    f"{val:+.4f}", va="center", ha="right", fontsize=7.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 5: ordering failure diagnosis")


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 - Heatmaps (layer × top-32 experts) per category
# ─────────────────────────────────────────────────────────────────────────────

def page_heatmaps(pdf):
    # Find top-32 most active experts globally to keep heatmaps readable
    global_mean = sum(category_heatmaps[c] for c in CATEGORIES) / len(CATEGORIES)
    top_32_expert_indices = np.argsort(global_mean.mean(axis=0))[-32:][::-1]
    top_32_expert_labels = [f"E{expert_idx}" for expert_idx in top_32_expert_indices]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Visual Token Expert Activation Heatmaps\n"
                 "(Layer × Top-32 most active experts; colour = mean routing weight)",
                 fontsize=12, fontweight="bold", y=0.98)

    vmax = max(category_heatmaps[c][:, top_32_expert_indices].max() for c in CATEGORIES)

    for ax, cat in zip(axes.flat, CATEGORIES):
        category_heatmap_subset = category_heatmaps[cat][:, top_32_expert_indices]
        sns.heatmap(category_heatmap_subset, ax=ax, cmap="YlOrRd",
                    vmin=0, vmax=vmax,
                    xticklabels=top_32_expert_labels,
                    yticklabels=[str(layer_idx) if layer_idx % 8 == 0 else "" for layer_idx in range(NUM_LAYERS)],
                    linewidths=0, cbar=True,
                    cbar_kws={"shrink": 0.6, "label": "routing weight"})
        ax.set_title(f"{cat.upper()}  (acc={category_accuracy[cat]:.1%})",
                     color=CAT_COLORS[cat], fontweight="bold")
        ax.set_xlabel("Expert (top-32 by global mean)", fontsize=7)
        ax.set_ylabel("Layer", fontsize=7)
        ax.tick_params(axis="x", labelsize=5.5, rotation=90)
        ax.tick_params(axis="y", labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 6: heatmaps")


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 - Height vs Rotation difference map
# ─────────────────────────────────────────────────────────────────────────────

def page_diff_heatmap(pdf):
    global_mean = sum(category_heatmaps[c] for c in CATEGORIES) / len(CATEGORIES)
    top_48_expert_indices = np.argsort(global_mean.mean(axis=0))[-48:][::-1]
    top_48_expert_labels = [f"E{expert_idx}" for expert_idx in top_48_expert_indices]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
    page_header(fig, "Height vs Rotation - Expert Specialisation",
                "Blue = Height-preferring experts  |  Red = Rotation-preferring experts  |  White = shared")

    vabs = np.abs(height_rotation_diff[:, top_48_expert_indices]).max()

    sns.heatmap(height_rotation_diff[:, top_48_expert_indices], ax=axes[0], cmap="coolwarm",
                vmin=-vabs, vmax=vabs,
                xticklabels=top_48_expert_labels,
                yticklabels=[str(layer_idx) if layer_idx % 8 == 0 else "" for layer_idx in range(NUM_LAYERS)],
                linewidths=0, cbar=True,
                cbar_kws={"shrink": 0.5, "label": "Height − Rotation"})
    axes[0].set_title("Height − Rotation difference\n(top-48 experts)", fontweight="bold")
    axes[0].set_xlabel("Expert", fontsize=7)
    axes[0].set_ylabel("Layer", fontsize=7)
    axes[0].tick_params(axis="x", labelsize=5, rotation=90)
    axes[0].tick_params(axis="y", labelsize=6)

    # Bar: top height-specialised vs rotation-specialised experts
    diff_by_expert = height_rotation_diff.mean(axis=0)  # [128]
    height_top_5_experts = np.argsort(diff_by_expert)[-5:][::-1]
    rotation_top_5_experts  = np.argsort(diff_by_expert)[:5]

    labels = ([f"E{expert_idx}\n(H)" for expert_idx in height_top_5_experts] +
              [f"E{expert_idx}\n(R)" for expert_idx in rotation_top_5_experts])
    vals   = (list(diff_by_expert[height_top_5_experts]) +
              list(diff_by_expert[rotation_top_5_experts]))
    colors = ["#4C72B0"] * 5 + ["#C44E52"] * 5

    axes[1].barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    axes[1].axvline(0, color=DARK, linewidth=0.8)
    axes[1].set_xlabel("Mean Δ (Height − Rotation)")
    axes[1].set_title("Most specialised experts\nacross all layers", fontweight="bold")
    axes[1].grid(axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 7: Height vs Rotation diff map")


# ─────────────────────────────────────────────────────────────────────────────
# Page 8 - Success-delta scatter: activation vs delta per expert
# ─────────────────────────────────────────────────────────────────────────────

def page_scatter(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Expert Activation vs Success Δ - per Category\n"
                 "(Top-right quadrant = high activation + positive correlation with correct answers)",
                 fontsize=11, fontweight="bold", y=0.98)

    for ax, cat in zip(axes.flat, CATEGORIES):
        delta_col = f"delta_{cat}"
        category_experts = df_rank[df_rank["mean_activation_correct"] >= 0.001].copy()

        scatter_plot = ax.scatter(
            category_experts["mean_activation_correct"],
            category_experts[delta_col],
            c=category_experts[delta_col],
            cmap="RdYlGn",
            s=6, alpha=0.55, linewidths=0,
        )
        ax.axhline(0, color=DARK, linewidth=0.8, linestyle="--")
        ax.set_title(f"{cat.upper()}", color=CAT_COLORS[cat], fontweight="bold")
        ax.set_xlabel("Mean activation (correct answers)", fontsize=8)
        ax.set_ylabel(f"delta_{cat}", fontsize=8)
        ax.grid(True, alpha=0.4)
        plt.colorbar(scatter_plot, ax=ax, shrink=0.7, label="Δ value")

        # Label top-3 in upper-right
        top_3_experts = category_experts.nlargest(3, delta_col)
        for _, expert_row in top_3_experts.iterrows():
            ax.annotate(expert_row["expert_label"],
                        (expert_row["mean_activation_correct"], expert_row[delta_col]),
                        fontsize=6, xytext=(4, 2), textcoords="offset points",
                        color=DARK)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 8: activation vs delta scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Page 9 - Layer-depth distribution of top experts
# ─────────────────────────────────────────────────────────────────────────────

def page_layer_depth(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Where in the Network Do Spatial Experts Live?",
                "Layer distribution of the top-30 global spatial experts vs category specialists")

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           top=0.88, bottom=0.07, hspace=0.45)

    # Top-30 global layer histogram
    ax1 = fig.add_subplot(gs[0])
    top30_layers = df_board.head(30)["layer"].values
    ax1.hist(top30_layers, bins=range(0, NUM_LAYERS + 2), color=ACCENT,
             alpha=0.85, edgecolor="white", rwidth=0.85)
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Number of top-30 experts")
    ax1.set_title("Layer Distribution - Top-30 Global Spatial Experts")
    ax1.set_xlim(-1, NUM_LAYERS)
    ax1.grid(axis="y")

    # Per-category specialist layer histograms (overlaid)
    ax2 = fig.add_subplot(gs[1])
    for cat in CATEGORIES:
        delta_col = f"delta_{cat}"
        top10_layers = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
                        .nlargest(10, delta_col)["layer"].values)
        ax2.hist(top10_layers, bins=range(0, NUM_LAYERS + 2),
                 color=CAT_COLORS[cat], alpha=0.5,
                 edgecolor="white", rwidth=0.85, label=cat)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Number of category-specialist experts (top-10)")
    ax2.set_title("Layer Distribution - Per-Category Top-10 Specialist Experts")
    ax2.set_xlim(-1, NUM_LAYERS)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 9: layer depth distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Assemble PDF
# ─────────────────────────────────────────────────────────────────────────────

print(f"Generating {OUTPUT_PDF} …")
with PdfPages(OUTPUT_PDF) as pdf:
    pdf_metadata = pdf.infodict()
    pdf_metadata["Title"]   = f"MoE Expert Routing Analysis - {MODEL_NAME}"
    pdf_metadata["Subject"] = "LEGOLite 3D Spatial Reasoning Benchmark"
    pdf_metadata["Author"]  = "lego_lite_moe_analysis.py"

    page_cover(pdf)
    page_accuracy(pdf)
    page_leaderboard(pdf)
    page_specialists(pdf)
    page_ordering(pdf)
    page_heatmaps(pdf)
    page_diff_heatmap(pdf)
    page_scatter(pdf)
    page_layer_depth(pdf)

size_mb = OUTPUT_PDF.stat().st_size / 1024 / 1024
print(f"\nDone. {OUTPUT_PDF}  ({size_mb:.1f} MB, 9 pages)")
