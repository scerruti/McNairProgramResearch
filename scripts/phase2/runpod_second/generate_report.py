"""
generate_report_run2.py
=======================
Generates a PDF report for the LEGOLite MoE expert analysis Run 2, with:
  - All plots fully labelled on both axes
  - An additional Run 1 vs Run 2 comparison page
Output: /workspace/lego_lite_analysis_run2/report.pdf
"""

import json
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
RUN1_DIR = Path("/workspace/lego_lite_analysis")
RUN2_DIR = Path("/workspace/lego_lite_analysis_run2")

RESULTS_JSON = RUN2_DIR / "results.json"
RANK_CSV     = RUN2_DIR / "expert_success_rates.csv"
BOARD_CSV    = RUN2_DIR / "spatial_expert_leaderboard.csv"
OUTPUT_PDF   = RUN2_DIR / "report.pdf"

CATEGORIES = ["height", "position", "rotation", "ordering"]
CAT_COLORS = {"height": "#4C72B0", "position": "#55A868",
              "rotation": "#C44E52", "ordering": "#DD8452"}
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"
BENCHMARK  = "LEGOLite (400 questions)"

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

NUM_LAYERS  = 48
NUM_EXPERTS = 128


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading run2 results…")
with open(RESULTS_JSON) as f:
    results = json.load(f)

df_rank  = pd.read_csv(RANK_CSV)
df_board = pd.read_csv(BOARD_CSV)

category_accuracy = {}
for cat in CATEGORIES:
    sub = [r for r in results if r["category"] == cat]
    category_accuracy[cat] = sum(r["correct"] for r in sub) / len(sub) if sub else 0.0
overall_acc = sum(r["correct"] for r in results) / len(results)

category_heatmaps = {}
for cat in CATEGORIES:
    arr = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    n   = 0
    for r in results:
        if r["category"] != cat:
            continue
        n += 1
        for li_str, weights in r["visual_routing"].items():
            arr[int(li_str)] += np.asarray(weights)
    category_heatmaps[cat] = arr / n if n > 0 else arr

height_rotation_diff = category_heatmaps["height"] - category_heatmaps["rotation"]

# Load run1 for comparison
print("Loading run1 results for comparison…")
with open(RUN1_DIR / "results.json") as f:
    results_r1 = json.load(f)

category_accuracy_run1 = {}
for cat in CATEGORIES:
    sub = [r for r in results_r1 if r["category"] == cat]
    category_accuracy_run1[cat] = sum(r["correct"] for r in sub) / len(sub) if sub else 0.0
overall_accuracy_run1 = sum(r["correct"] for r in results_r1) / len(results_r1)

df_board_r1 = pd.read_csv(RUN1_DIR / "spatial_expert_leaderboard.csv")
df_rank_r1  = pd.read_csv(RUN1_DIR / "expert_success_rates.csv")


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
    n_cols = len(col_labels)
    row_colors = row_colors or [["white"] * n_cols] * len(df)
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#DDDDDD")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 - Cover / Summary
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0, 0.78), 1, 0.22, linewidth=0,
                                facecolor=DARK, boxstyle="square,pad=0"))
    fig.text(0.5, 0.955, "MoE Expert Routing Analysis - Run 2", ha="center",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.5, 0.895, MODEL_NAME, ha="center", fontsize=13, color="#AAAACC")
    fig.text(0.5, 0.855, BENCHMARK, ha="center", fontsize=11, color="#AAAACC")

    card_w, card_h = 0.18, 0.13
    xs = [0.07, 0.295, 0.52, 0.745]
    ys = 0.60
    for i, cat in enumerate(CATEGORIES):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x, ys), card_w, card_h, linewidth=1.5,
                                    edgecolor=CAT_COLORS[cat], facecolor=LIGHT,
                                    boxstyle="round,pad=0.01"))
        fig.text(x + card_w/2, ys + card_h - 0.025, cat.upper(),
                 ha="center", fontsize=9, fontweight="bold", color=CAT_COLORS[cat])
        fig.text(x + card_w/2, ys + card_h/2 - 0.008,
                 f"{category_accuracy[cat]:.1%}", ha="center",
                 fontsize=20, fontweight="bold", color=DARK)
        delta = category_accuracy[cat] - category_accuracy_run1[cat]
        delta_str = f"{'↑' if delta >= 0 else '↓'}{abs(delta):.1%} vs Run 1"
        delta_color = "#2ecc71" if delta >= 0 else "#e74c3c"
        fig.text(x + card_w/2, ys + 0.015, delta_str,
                 ha="center", fontsize=7.5, color=delta_color)

    ax.add_patch(FancyBboxPatch((0.35, 0.46), 0.30, 0.10, linewidth=2,
                                edgecolor=ACCENT, facecolor=LIGHT,
                                boxstyle="round,pad=0.01"))
    fig.text(0.5, 0.535, "OVERALL ACCURACY", ha="center",
             fontsize=9, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.488, f"{overall_acc:.1%}", ha="center",
             fontsize=22, fontweight="bold", color=DARK)

    findings = [
        f"• {NUM_LAYERS} MoE layers × {NUM_EXPERTS} experts × Top-8 routing analysed",
        "• 400 LEGOLite questions (100 per spatial category)",
        "• Router activations captured for visual tokens only",
        f"• Run 1 overall: {overall_accuracy_run1:.1%}  |  Run 2 overall: {overall_acc:.1%}",
        f"• Δ overall: {overall_acc - overall_accuracy_run1:+.1%}",
        "• See page 9 for full Run 1 vs Run 2 comparison",
    ]
    fig.text(0.07, 0.415, "Key Findings", fontsize=11, fontweight="bold", color=DARK)
    for j, line in enumerate(findings):
        fig.text(0.07, 0.375 - j * 0.033, line, fontsize=9, color="#333333")

    fig.text(0.5, 0.03, "Generated from /workspace/lego_lite_analysis_run2/",
             ha="center", fontsize=8, color="#999999")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 1: cover")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 - Accuracy breakdown + layer activation profiles
# ─────────────────────────────────────────────────────────────────────────────

def page_accuracy(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Accuracy & Per-Category Expert Activation",
                "How often the model answers correctly, and which layers are most active per category")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.55, wspace=0.35)

    ax_acc = fig.add_subplot(gs[0, :])
    cats   = CATEGORIES + ["overall"]
    vals   = [category_accuracy[c] for c in CATEGORIES] + [overall_acc]
    colors = [CAT_COLORS[c] for c in CATEGORIES] + [DARK]
    bars   = ax_acc.bar(cats, vals, color=colors, width=0.55, zorder=3)
    ax_acc.set_ylim(0, 0.55)
    ax_acc.set_xlabel("Spatial Category", fontsize=9)
    ax_acc.set_ylabel("Accuracy (fraction correct)", fontsize=9)
    ax_acc.set_title("Accuracy by Category - Run 2")
    ax_acc.axhline(0.25, color="#AAAAAA", linestyle="--", linewidth=1,
                   label="random baseline (4-choice = 25%)")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", zorder=0)
    for bar, val in zip(bars, vals):
        ax_acc.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")

    for i, cat in enumerate(CATEGORIES):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row + 1, col])
        layer_means = category_heatmaps[cat].mean(axis=1)
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
        ax.set_xlabel("Layer index (0 = earliest, 47 = latest)", fontsize=8)
        ax.set_ylabel("Mean routing weight\n(avg over 128 experts)", fontsize=8)
        ax.grid(axis="y")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 2: accuracy & activation profiles")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 - Global leaderboard table
# ─────────────────────────────────────────────────────────────────────────────

def page_leaderboard(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Top-30 Spatial Reasoning Experts - Global Leaderboard",
                "Ranked by success_delta = mean_activation_correct − mean_activation_incorrect")

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
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
    row_colors = [["#EEF2FF" if r % 2 == 0 else "white"] * 5 for r in range(len(display))]
    _add_table(ax, display, col_labels, col_widths, row_colors, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 3: leaderboard table")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 - Category specialist tables
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
        sub = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
               .nlargest(10, delta_col)
               [["expert_label", "mean_activation_correct", delta_col]]
               .copy())
        sub["mean_activation_correct"] = sub["mean_activation_correct"].map("{:.5f}".format)
        sub[delta_col]                 = sub[delta_col].map("{:+.5f}".format)

        col_labels = ["Expert", "Act (correct)", f"Δ {cat}"]
        col_widths = [0.18, 0.20, 0.18]
        row_colors = [["#EEF2FF" if r % 2 == 0 else "white"] * 3 for r in range(len(sub))]
        _add_table(ax, sub, col_labels, col_widths, row_colors, fontsize=8)

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

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.5)

    ax_good = fig.add_subplot(gs[0])
    good = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
            .nlargest(10, "delta_ordering")
            [["expert_label", "delta_ordering", "mean_activation_correct"]]
            .copy())
    ax_good.barh(good["expert_label"], good["delta_ordering"],
                 color="#2ecc71", edgecolor="white", height=0.6)
    ax_good.set_xlabel("delta_ordering  (positive = fires more on CORRECT answers)", fontsize=8)
    ax_good.set_ylabel("Expert (Layer_Expert index)", fontsize=8)
    ax_good.set_title("Top Experts Correlated With Correct Ordering Answers")
    ax_good.axvline(0, color=DARK, linewidth=0.8)
    ax_good.grid(axis="x", zorder=0)
    for bar, val in zip(ax_good.patches, good["delta_ordering"]):
        ax_good.text(val + 0.0003, bar.get_y() + bar.get_height()/2,
                     f"{val:+.4f}", va="center", fontsize=7.5)

    ax_bad = fig.add_subplot(gs[1])
    suspects = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
                .nsmallest(10, "delta_ordering")
                [["expert_label", "delta_ordering", "mean_activation_correct"]]
                .copy())
    ax_bad.barh(suspects["expert_label"], suspects["delta_ordering"],
                color="#e74c3c", edgecolor="white", height=0.6)
    ax_bad.set_xlabel("delta_ordering  (negative = fires more on WRONG answers)", fontsize=8)
    ax_bad.set_ylabel("Expert (Layer_Expert index)", fontsize=8)
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
# Page 6 - Heatmaps (layer × top-32 experts)
# ─────────────────────────────────────────────────────────────────────────────

def page_heatmaps(pdf):
    global_mean = sum(category_heatmaps[c] for c in CATEGORIES) / len(CATEGORIES)
    top_32_expert_indices = np.argsort(global_mean.mean(axis=0))[-32:][::-1]
    top_32_expert_labels = [f"E{e}" for e in top_32_expert_indices]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Visual Token Expert Activation Heatmaps - Run 2\n"
                 "(Layer × Top-32 most active experts; colour = mean routing weight)",
                 fontsize=12, fontweight="bold", y=0.98)

    vmax = max(category_heatmaps[c][:, top_32_expert_indices].max() for c in CATEGORIES)

    for ax, cat in zip(axes.flat, CATEGORIES):
        data = category_heatmaps[cat][:, top_32_expert_indices]
        sns.heatmap(data, ax=ax, cmap="YlOrRd",
                    vmin=0, vmax=vmax,
                    xticklabels=top_32_expert_labels,
                    yticklabels=[str(l) if l % 8 == 0 else "" for l in range(NUM_LAYERS)],
                    linewidths=0, cbar=True,
                    cbar_kws={"shrink": 0.6, "label": "Mean routing weight"})
        ax.set_title(f"{cat.upper()}  (acc={category_accuracy[cat]:.1%})",
                     color=CAT_COLORS[cat], fontweight="bold")
        ax.set_xlabel("Expert ID (top-32 by global mean activation)", fontsize=7)
        ax.set_ylabel("Layer index (0 = earliest, 47 = latest)", fontsize=7)
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
    top_48_expert_labels = [f"E{e}" for e in top_48_expert_indices]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
    page_header(fig, "Height vs Rotation - Expert Specialisation",
                "Blue = Height-preferring  |  Red = Rotation-preferring  |  White = shared")

    vabs = np.abs(height_rotation_diff[:, top_48_expert_indices]).max()

    sns.heatmap(height_rotation_diff[:, top_48_expert_indices], ax=axes[0], cmap="coolwarm",
                vmin=-vabs, vmax=vabs,
                xticklabels=top_48_expert_labels,
                yticklabels=[str(l) if l % 8 == 0 else "" for l in range(NUM_LAYERS)],
                linewidths=0, cbar=True,
                cbar_kws={"shrink": 0.5, "label": "Height routing − Rotation routing"})
    axes[0].set_title("Height − Rotation difference\n(top-48 experts)", fontweight="bold")
    axes[0].set_xlabel("Expert ID (top-48 by global mean activation)", fontsize=7)
    axes[0].set_ylabel("Layer index (0 = earliest, 47 = latest)", fontsize=7)
    axes[0].tick_params(axis="x", labelsize=5, rotation=90)
    axes[0].tick_params(axis="y", labelsize=6)

    diff_by_expert = height_rotation_diff.mean(axis=0)
    height_top_5_experts = np.argsort(diff_by_expert)[-5:][::-1]
    rotation_top_5_experts  = np.argsort(diff_by_expert)[:5]

    labels = ([f"E{e}\n(H)" for e in height_top_5_experts] +
              [f"E{e}\n(R)" for e in rotation_top_5_experts])
    vals   = (list(diff_by_expert[height_top_5_experts]) +
              list(diff_by_expert[rotation_top_5_experts]))
    colors = ["#4C72B0"] * 5 + ["#C44E52"] * 5

    axes[1].barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    axes[1].axvline(0, color=DARK, linewidth=0.8)
    axes[1].set_xlabel("Mean Δ (Height routing − Rotation routing)\naveraged across all 48 layers",
                       fontsize=8)
    axes[1].set_ylabel("Expert ID and preference (H=Height, R=Rotation)", fontsize=8)
    axes[1].set_title("Most specialised experts\nacross all layers", fontweight="bold")
    axes[1].grid(axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 7: Height vs Rotation diff map")


# ─────────────────────────────────────────────────────────────────────────────
# Page 8 - Activation vs success-delta scatter per category
# ─────────────────────────────────────────────────────────────────────────────

def page_scatter(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Expert Activation vs Success Δ - per Category (Run 2)\n"
                 "Top-right quadrant = high activation + positive correlation with correct answers",
                 fontsize=11, fontweight="bold", y=0.98)

    for ax, cat in zip(axes.flat, CATEGORIES):
        delta_col = f"delta_{cat}"
        sub = df_rank[df_rank["mean_activation_correct"] >= 0.001].copy()

        sc = ax.scatter(
            sub["mean_activation_correct"],
            sub[delta_col],
            c=sub[delta_col],
            cmap="RdYlGn",
            s=6, alpha=0.55, linewidths=0,
        )
        ax.axhline(0, color=DARK, linewidth=0.8, linestyle="--")
        ax.set_title(f"{cat.upper()}", color=CAT_COLORS[cat], fontweight="bold")
        ax.set_xlabel("Mean routing weight on correct answers\n(visual tokens only)", fontsize=8)
        ax.set_ylabel(f"Success Δ for {cat}\n(correct activation − incorrect activation)",
                      fontsize=8)
        ax.grid(True, alpha=0.4)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="Δ value")

        top3 = sub.nlargest(3, delta_col)
        for _, r in top3.iterrows():
            ax.annotate(r["expert_label"],
                        (r["mean_activation_correct"], r[delta_col]),
                        fontsize=6, xytext=(4, 2), textcoords="offset points",
                        color=DARK)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 8: activation vs delta scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Page 9 - Run 1 vs Run 2 comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_comparison(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Run 1 vs Run 2 - Repeatability Analysis",
                "Same model, same dataset, same prompts - measuring variance across runs")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.55, wspace=0.4)

    # ── Grouped bar: accuracy per category both runs ──────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    x      = np.arange(len(CATEGORIES) + 1)
    cats_w_overall = CATEGORIES + ["overall"]
    r1_vals = [category_accuracy_run1[c] for c in CATEGORIES] + [overall_accuracy_run1]
    r2_vals = [category_accuracy[c]    for c in CATEGORIES] + [overall_acc]
    w = 0.35
    bars1 = ax_bar.bar(x - w/2, r1_vals, w, label="Run 1", color="#7FB3D3",
                       edgecolor="white", zorder=3)
    bars2 = ax_bar.bar(x + w/2, r2_vals, w, label="Run 2", color="#E59866",
                       edgecolor="white", zorder=3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cats_w_overall)
    ax_bar.set_xlabel("Spatial Category", fontsize=9)
    ax_bar.set_ylabel("Accuracy (fraction correct)", fontsize=9)
    ax_bar.set_title("Per-Category Accuracy: Run 1 vs Run 2")
    ax_bar.set_ylim(0, 0.55)
    ax_bar.axhline(0.25, color="#AAAAAA", linestyle="--", linewidth=1,
                   label="random baseline (25%)")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", zorder=0)
    for bar, val in zip(bars1, r1_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                    f"{val:.1%}", ha="center", fontsize=7.5, color="#333333")
    for bar, val in zip(bars2, r2_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                    f"{val:.1%}", ha="center", fontsize=7.5, color="#333333")

    # ── Delta bar: run2 - run1 per category ──────────────────────────────────
    ax_delta = fig.add_subplot(gs[1, :])
    deltas = [category_accuracy[c] - category_accuracy_run1[c] for c in CATEGORIES] + [overall_acc - overall_accuracy_run1]
    colors_delta = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    bars_d = ax_delta.bar(x, deltas, 0.55, color=colors_delta, edgecolor="white", zorder=3)
    ax_delta.axhline(0, color=DARK, linewidth=1)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(cats_w_overall)
    ax_delta.set_xlabel("Spatial Category", fontsize=9)
    ax_delta.set_ylabel("Accuracy change (Run 2 − Run 1)\npositive = improved in Run 2", fontsize=9)
    ax_delta.set_title("Accuracy Delta: Run 2 − Run 1")
    ax_delta.grid(axis="y", zorder=0)
    for bar, val in zip(bars_d, deltas):
        ypos = val + 0.002 if val >= 0 else val - 0.006
        ax_delta.text(bar.get_x() + bar.get_width()/2, ypos,
                      f"{val:+.1%}", ha="center", fontsize=8.5, fontweight="bold")

    # ── Expert leaderboard stability: top-10 overlap ──────────────────────────
    ax_overlap = fig.add_subplot(gs[2, 0])
    top_n_vals = list(range(5, 31, 5))
    overlaps = []
    for n in top_n_vals:
        r1_set = set(df_board_r1.head(n)["expert_label"])
        r2_set = set(df_board.head(n)["expert_label"])
        overlaps.append(len(r1_set & r2_set) / n)
    ax_overlap.plot(top_n_vals, overlaps, marker="o", color=ACCENT, linewidth=2)
    ax_overlap.fill_between(top_n_vals, overlaps, alpha=0.2, color=ACCENT)
    ax_overlap.set_xlabel("Top-N experts considered", fontsize=8)
    ax_overlap.set_ylabel("Fraction of experts in both runs\n(Jaccard-style overlap)", fontsize=8)
    ax_overlap.set_title("Expert Leaderboard Stability\nRun1 ∩ Run2 overlap")
    ax_overlap.set_ylim(0, 1.05)
    ax_overlap.axhline(1.0, color="#AAAAAA", linestyle="--", linewidth=1, label="perfect stability")
    ax_overlap.legend(fontsize=7)
    ax_overlap.grid(True, alpha=0.4)
    for xv, yv in zip(top_n_vals, overlaps):
        ax_overlap.text(xv, yv + 0.03, f"{yv:.0%}", ha="center", fontsize=7.5)

    # ── Success-delta correlation: run1 vs run2 global delta ─────────────────
    ax_corr = fig.add_subplot(gs[2, 1])
    merged = df_rank.merge(df_rank_r1, on="expert_label", suffixes=("_r2", "_r1"))
    sample = merged.sample(min(500, len(merged)), random_state=42)
    sc = ax_corr.scatter(
        sample["success_delta_r1"],
        sample["success_delta_r2"],
        c=sample["success_delta_r2"] - sample["success_delta_r1"],
        cmap="RdYlGn", s=5, alpha=0.5, linewidths=0,
    )
    lim = max(abs(sample["success_delta_r1"].max()),
              abs(sample["success_delta_r2"].max())) * 1.1
    ax_corr.set_xlim(-lim, lim)
    ax_corr.set_ylim(-lim, lim)
    ax_corr.axhline(0, color=DARK, linewidth=0.6)
    ax_corr.axvline(0, color=DARK, linewidth=0.6)
    ax_corr.plot([-lim, lim], [-lim, lim], color="#AAAAAA",
                 linewidth=1, linestyle="--", label="perfect agreement")
    ax_corr.set_xlabel("Expert success_delta - Run 1\n(correct − incorrect activation)",
                       fontsize=8)
    ax_corr.set_ylabel("Expert success_delta - Run 2\n(correct − incorrect activation)",
                       fontsize=8)
    ax_corr.set_title("Expert Δ Correlation\nRun 1 vs Run 2 (500 random experts)")
    ax_corr.legend(fontsize=7)
    plt.colorbar(sc, ax=ax_corr, shrink=0.7, label="Δ(R2)−Δ(R1)")

    # Pearson r annotation
    corr = np.corrcoef(merged["success_delta_r1"], merged["success_delta_r2"])[0, 1]
    ax_corr.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax_corr.transAxes,
                 fontsize=8.5, color=DARK, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 9: Run 1 vs Run 2 comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Page 10 - Layer depth distribution
# ─────────────────────────────────────────────────────────────────────────────

def page_layer_depth(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    page_header(fig, "Where in the Network Do Spatial Experts Live?",
                "Layer distribution of the top-30 global spatial experts vs category specialists")

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           top=0.88, bottom=0.07, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    top30_layers = df_board.head(30)["layer"].values
    ax1.hist(top30_layers, bins=range(0, NUM_LAYERS + 2), color=ACCENT,
             alpha=0.85, edgecolor="white", rwidth=0.85)
    ax1.set_xlabel("Layer index (0 = first transformer layer, 47 = last)", fontsize=9)
    ax1.set_ylabel("Number of top-30 experts at this layer", fontsize=9)
    ax1.set_title("Layer Distribution - Top-30 Global Spatial Experts (Run 2)")
    ax1.set_xlim(-1, NUM_LAYERS)
    ax1.grid(axis="y")

    ax2 = fig.add_subplot(gs[1])
    for cat in CATEGORIES:
        delta_col = f"delta_{cat}"
        top10_layers = (df_rank[df_rank["mean_activation_correct"] >= 0.003]
                        .nlargest(10, delta_col)["layer"].values)
        ax2.hist(top10_layers, bins=range(0, NUM_LAYERS + 2),
                 color=CAT_COLORS[cat], alpha=0.5,
                 edgecolor="white", rwidth=0.85, label=cat)
    ax2.set_xlabel("Layer index (0 = first transformer layer, 47 = last)", fontsize=9)
    ax2.set_ylabel("Number of top-10 category-specialist\nexperts at this layer", fontsize=9)
    ax2.set_title("Layer Distribution - Per-Category Top-10 Specialist Experts (Run 2)")
    ax2.set_xlim(-1, NUM_LAYERS)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 10: layer depth distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Assemble PDF
# ─────────────────────────────────────────────────────────────────────────────

print(f"Generating {OUTPUT_PDF} …")
with PdfPages(OUTPUT_PDF) as pdf:
    meta = pdf.infodict()
    meta["Title"]   = f"MoE Expert Routing Analysis Run 2 - {MODEL_NAME}"
    meta["Subject"] = "LEGOLite 3D Spatial Reasoning Benchmark"
    meta["Author"]  = "generate_report_run2.py"

    page_cover(pdf)
    page_accuracy(pdf)
    page_leaderboard(pdf)
    page_specialists(pdf)
    page_ordering(pdf)
    page_heatmaps(pdf)
    page_diff_heatmap(pdf)
    page_scatter(pdf)
    page_comparison(pdf)
    page_layer_depth(pdf)

size_mb = OUTPUT_PDF.stat().st_size / 1024 / 1024
print(f"\nDone. {OUTPUT_PDF}  ({size_mb:.1f} MB, 10 pages)")
