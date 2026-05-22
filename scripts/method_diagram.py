"""
Professional MoE Expert Routing Analysis Methodology Diagram
============================================================
High-quality flowchart with modern styling, shadows, and visual hierarchy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import patheffects as path_effects
import matplotlib.patheffects as peffects

# Create figure with high DPI
fig = plt.figure(figsize=(20, 13), dpi=150)
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# Professional color palette
colors = {
    'header': '#1F77B4',
    'phase1': '#2CA02C',
    'phase1_light': '#D4E8D4',
    'phase2': '#FF7F0E',
    'phase2_light': '#FFE6CC',
    'box': '#FFFFFF',
    'accent': '#D62728',
    'text_dark': '#2C3E50',
    'text_light': '#7F8C8D'
}

def draw_shadow_box(ax, x, y, width, height, color='#FFFFFF', 
                    shadow=True, shadow_offset=0.08):
    """Draw a box with optional shadow effect"""
    if shadow:
        shadow_box = FancyBboxPatch(
            (x - width/2 + shadow_offset, y - height/2 - shadow_offset), 
            width, height,
            boxstyle="round,pad=0.12", 
            edgecolor='none', 
            facecolor='#00000015', 
            linewidth=0,
            zorder=1
        )
        ax.add_patch(shadow_box)
    
    main_box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.12", 
        edgecolor='#CCCCCC', 
        facecolor=color, 
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(main_box)

def draw_header_box(ax, x, y, width, height, text, color):
    """Draw header box with gradient effect"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        edgecolor='#333333', 
        facecolor=color, 
        linewidth=2.5,
        zorder=3
    )
    ax.add_patch(box)
    
    txt = ax.text(x, y, text, ha='center', va='center', fontsize=12, 
                  fontweight='bold', color='white', zorder=4)
    txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='#333333'),
                          path_effects.Normal()])

def draw_content_box(ax, x, y, width, height, text, color, fontsize=10):
    """Draw content box with text"""
    draw_shadow_box(ax, x, y, width, height, color=color, shadow=True)
    
    txt = ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                  fontweight='normal', color=colors['text_dark'], wrap=True, 
                  multialignment='center', zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, width=2.5, color='#333333', style='->'):
    """Draw professional arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=25, linewidth=width,
        color=color, zorder=2
    )
    ax.add_patch(arrow)

# ============ MAIN HEADER ============
draw_header_box(ax, 10, 12.2, 19, 0.9, 
                'GOAL: Extract Internal Routing Metrics by Bypassing Standard Serverless APIs', 
                colors['header'])

ax.text(10, 11.6, 'Two-Phase Testing Approach for Model Routing Metrics', 
        ha='center', va='center', fontsize=11, style='italic', color=colors['text_light'])

# ============ PHASE 1: MODALITY DISCOVERY (LEFT) ============
phase1_x = 5
y_start = 10.2

# Phase 1 header
draw_header_box(ax, phase1_x, y_start, 8.5, 0.7, 
                'PHASE 1: MODALITY DISCOVERY (GOOGLE COLAB)', colors['phase1'])

# Phase 1 boxes
y_boxes = 8.8
box_width, box_height = 2.3, 0.8

draw_content_box(ax, phase1_x - 2.5, y_boxes, box_width, box_height,
                'Run 4-bit\nQuantized Model\non T4 GPU', colors['phase1_light'], fontsize=9)

draw_content_box(ax, phase1_x, y_boxes, box_width, box_height,
                'Log Routing\nWeights During\nSample Run', colors['phase1_light'], fontsize=9)

draw_content_box(ax, phase1_x + 2.5, y_boxes, box_width, box_height,
                'Analyze Pathway\nUsage: Text vs\nImages', colors['phase1_light'], fontsize=9)

# Arrows between phase 1 boxes
draw_arrow(ax, phase1_x - 1.1, y_boxes, phase1_x - 1.15, y_boxes, width=2.5)
draw_arrow(ax, phase1_x + 1.1, y_boxes, phase1_x + 1.15, y_boxes, width=2.5)

# Phase 1 result
y_result1 = 7.2
draw_shadow_box(ax, phase1_x, y_result1, 8, 0.95, color=colors['phase1'])
ax.text(phase1_x, y_result1, 
        'RESULT: Visual Experts Operate Independently\n(CONFIRMED: Distinct pathways for text and image processing)',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=5)

draw_arrow(ax, phase1_x, y_boxes - 0.4, phase1_x, y_result1 + 0.5, width=2.5)

# ============ PHASE 2: SPATIAL EXPERTS (RIGHT) ============
phase2_x = 15
y_start = 10.2

# Phase 2 header
draw_header_box(ax, phase2_x, y_start, 8.5, 0.7, 
                'PHASE 2: FINDING THE SPATIAL EXPERTS (RUNPOD)', colors['phase2'])

# Setup box
y_setup = 9.3
draw_content_box(ax, phase2_x, y_setup, 8, 0.65,
                'Spin up GPU Server Instance for Visual Node Mapping',
                colors['phase2_light'], fontsize=10)

draw_arrow(ax, phase2_x, y_setup - 0.35, phase2_x, 8.6, width=2.5)

# Analysis loop box with border
loop_y_top = 8.2
loop_height = 4.2
loop_box = FancyBboxPatch(
    (phase2_x - 4.3, loop_y_top - loop_height), 8.6, loop_height,
    boxstyle="round,pad=0.15", 
    edgecolor=colors['phase2'], 
    facecolor=colors['phase2_light'], 
    linewidth=2.5,
    linestyle='--',
    zorder=1
)
ax.add_patch(loop_box)

ax.text(phase2_x + 3.8, loop_y_top - 0.3, 'ANALYSIS LOOP', 
        ha='left', va='center', fontsize=9, fontweight='bold', 
        color=colors['phase2'], style='italic')

# Loop components
loop_items_y = [7.5, 5.8, 4.1]
loop_labels = ['A FULL TESTING', 'B MAPPING & RANKING', 'C ISOLATING NODES']
loop_texts = [
    'Push 400 LEGOLite\nquestions through model\nCapture router data',
    'Group router data\nby question & category\nCompare vs accuracy scores',
    'Analyze correlations\nFilter & isolate nodes\nGood vs Bad nodes'
]

for i, (y, label, text) in enumerate(zip(loop_items_y, loop_labels, loop_texts)):
    x_pos = phase2_x - 2.2 if i % 2 == 0 else phase2_x + 2.2
    draw_content_box(ax, x_pos, y, 3.5, 1.0, 
                    f'{label}\n\n{text}', colors['box'], fontsize=8.5)

# Loop arrows
draw_arrow(ax, phase2_x - 1, 7.0, phase2_x - 0.5, 6.3, width=1.8, color=colors['phase2'])
draw_arrow(ax, phase2_x + 0.8, 5.3, phase2_x + 1.5, 4.6, width=1.8, color=colors['phase2'])
draw_arrow(ax, phase2_x + 3.5, 3.6, phase2_x - 3.5, 3.6, width=1.8, color=colors['phase2'])

# Feedback loop back up
loop_back = FancyArrowPatch(
    (phase2_x - 4, 3.9), (phase2_x - 4, 7),
    arrowstyle='->', mutation_scale=20, linewidth=1.8,
    color=colors['phase2'], connectionstyle="arc3,rad=-0.5", zorder=2)
ax.add_patch(loop_back)
ax.text(phase2_x - 4.7, 5.5, 'Iterate', fontsize=8, style='italic', 
        color=colors['phase2'], fontweight='bold')

# Phase 2 result
y_result2 = 1.8
draw_shadow_box(ax, phase2_x, y_result2, 8, 0.85, color=colors['phase2'])
ax.text(phase2_x, y_result2, 'IDENTIFIED: 3D Spatial Reasoning Expert Network',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=5)

draw_arrow(ax, phase2_x, 2.9, phase2_x, 2.25, width=2.5)

# ============ CROSS-PHASE CONNECTION ============
draw_arrow(ax, phase1_x + 4, 7.2, phase2_x - 4, 7.2, width=3, color='#666666')
ax.text(10, 7.5, 'Feedback & Validation', fontsize=9, style='italic', 
        fontweight='bold', color='#666666', ha='center')

# ============ FOOTER ============
footer_y = 0.5
footer_box = FancyBboxPatch(
    (0.5, footer_y - 0.35), 19, 0.7,
    boxstyle="round,pad=0.08", 
    edgecolor=colors['phase1'], 
    facecolor='#F0F8F0', 
    linewidth=2,
    zorder=1
)
ax.add_patch(footer_box)

ax.text(10, footer_y + 0.05, 
        'DELIVERABLES: JSON Export (Per-Question Expert Frequencies) | Category Summaries (Top 15 Experts) | Accuracy Correlations',
        ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text_dark'], zorder=5)

plt.tight_layout()
plt.savefig('/Users/waleed/CSE 199/method_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='#F8F9FA', edgecolor='none')
print("✓ Professional diagram saved!")
plt.show()
