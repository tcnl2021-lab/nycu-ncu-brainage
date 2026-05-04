"""Figure 5 scaffold — voxel vs ROI task-fMRI BETAs (paired-bootstrap forest plot).

Shows per-task augmentation gain when adding voxelwise task-fMRI ridge OOFs on
top of the addendum_2 baseline (00c). Sign convention follows audit_stats.py:

    dMAE = MAE_baseline − MAE_baseline+voxel    (positive ⇒ voxel improves)

Reference lines: 0 (no effect) and +0.02 yr (preregister §6 magnitude floor).

Until the external voxelwise CLI completes, the bars are rendered as light-grey
placeholders with a "PLACEHOLDER — values pending" annotation. Replace
PLACEHOLDER with the per-task results from `audit_stats.py` once available.

Output: audit/reports/figures/figure5_voxel_vs_roi_forest.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "black": "#000000",
    "grey":  "#7A7A7A",
    "lgrey": "#CCCCCC",
    "blue":  "#0072B2",   # seed 20260425
    "verm":  "#D55E00",   # seed 20260426
    "green": "#009E73",   # confirmed
    "yellow":"#F0E442",   # fragile
    "red":   "#CC0000",   # overturned
}

# Per-task placeholder data. Replace with addendum_3 stats when ready.
# Schema: (task_name, ROI_n_cols, dMAE_seed1, ci1_lo, ci1_hi,
#                                  dMAE_seed2, ci2_lo, ci2_hi)
TASKS = [
    ("MOTOR_GFORCE",          64,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    ("MEMORY_MST",           105,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    ("LANGUAGE_SPEECHCOMP",   21,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    ("LANGUAGE_WORDNAME",     18,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
]

ACCEPT_DELTA = 0.02   # preregister §6 magnitude floor (yr)
PLACEHOLDER_HALF = 0.045  # symmetric CI half-width drawn for placeholder bars


def draw(ax, x_label_pos):
    n = len(TASKS)
    y_positions = np.arange(n)[::-1]   # top-down

    # Reference lines
    ax.axvline(0.0, color=C["black"], lw=0.7, zorder=1)
    ax.axvline(+ACCEPT_DELTA, color=C["grey"], lw=0.7, ls="--", zorder=1)
    # §6 floor label lives just below the bottom data row (under x-axis area)
    ax.text(+ACCEPT_DELTA, -0.55, "§6 floor (+0.02 yr)",
            ha="center", va="top", fontsize=6.4, color=C["grey"],
            style="italic")

    for i, row in enumerate(TASKS):
        task, k_roi, d1, l1, h1, d2, l2, h2 = row
        y = y_positions[i]
        is_placeholder = np.isnan(d1)

        if is_placeholder:
            ax.plot([-PLACEHOLDER_HALF, +PLACEHOLDER_HALF],
                    [y, y], color=C["lgrey"], lw=4, solid_capstyle="round",
                    zorder=2)
            ax.text(0.0, y + 0.20, "pending", ha="center", va="bottom",
                    fontsize=6.4, color=C["grey"], style="italic")
        else:
            y1 = y + 0.14
            ax.plot([l1, h1], [y1, y1], color=C["blue"], lw=1.2, zorder=2)
            ax.plot(d1, y1, marker="o", color=C["blue"], markersize=4.5,
                    zorder=3, markeredgecolor=C["black"], markeredgewidth=0.4)
            y2 = y - 0.14
            ax.plot([l2, h2], [y2, y2], color=C["verm"], lw=1.2, zorder=2)
            ax.plot(d2, y2, marker="s", color=C["verm"], markersize=4.5,
                    zorder=3, markeredgecolor=C["black"], markeredgewidth=0.4)

        # Row label
        ax.text(x_label_pos, y + 0.06, task,
                ha="right", va="center", fontsize=7.4, weight="bold")
        ax.text(x_label_pos, y - 0.24, f"ROI: {k_roi} cols",
                ha="right", va="center", fontsize=6.4, color=C["grey"])

    ax.set_xlim(-0.105, +0.105)
    ax.set_ylim(-0.7, n - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel(
        "$\\Delta$MAE  (yr)   =   MAE$_{baseline}$ − MAE$_{baseline+voxel}$",
        fontsize=7.8)
    ax.set_xticks([-0.10, -0.05, 0.0, +0.02, +0.05, +0.10])
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", left=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C["black"])


def main():
    fig = plt.figure(figsize=(5.6, 3.6))   # ~142 mm × 91 mm
    ax = fig.add_axes([0.30, 0.22, 0.66, 0.55])
    draw(ax, x_label_pos=-0.115)

    # Title + placeholder note
    fig.text(0.04, 0.95,
             "Figure 5.  Voxelwise vs ROI task-fMRI BETAs:",
             ha="left", va="center", fontsize=8.3, weight="bold")
    fig.text(0.04, 0.905,
             "paired-bootstrap $\\Delta$MAE per task",
             ha="left", va="center", fontsize=8.3, weight="bold")
    fig.text(0.04, 0.855,
             "PLACEHOLDER — values pending external voxelwise CLI",
             ha="left", va="center", fontsize=6.7, style="italic",
             color=C["grey"], weight="bold")

    # Direction-of-effect annotations as footer text (clear of plot area)
    fig.text(0.30, 0.82, "← ROI alone better",
             ha="left", va="center", fontsize=7.0, color=C["grey"])
    fig.text(0.96, 0.82, "voxel improves →",
             ha="right", va="center", fontsize=7.0, color=C["grey"])

    # Legend in upper-right, clear of the title and axes
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C["blue"], markeredgecolor=C["black"],
                   markersize=5, label="seed 20260425"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=C["verm"], markeredgecolor=C["black"],
                   markersize=5, label="seed 20260426"),
        plt.Line2D([0], [0], color=C["lgrey"], lw=4,
                   label="placeholder"),
    ]
    fig.legend(handles=legend_handles,
               loc="upper right", bbox_to_anchor=(0.96, 0.96),
               frameon=False, fontsize=6.7, handlelength=1.4,
               labelspacing=0.30, ncol=1)

    pdf_path = OUT_DIR / "figure5_voxel_vs_roi_forest.pdf"
    png_path = OUT_DIR / "figure5_voxel_vs_roi_forest.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
