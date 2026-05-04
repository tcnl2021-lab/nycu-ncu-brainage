"""Render the brain-age audit workflow diagram (HBM-style figure).

Layout — full-page double-column (178 mm wide):
    Panel A. Cohort, modality blocks, fold preregistration.
    Panel B. Per-block fold-aligned RidgeCV -> meta-stacker -> OOF.
    Panel C. Addendum 1: information-gain frontier (cumulative blocks).
    Panel D. Addendum 2: acquisition-design LOO ablation.
    Panel E. Addendum 3 (planned): feature-granularity, voxel vs ROI task-fMRI.

Style:
    - Arial / Helvetica, 8 pt body, 9 pt panel headers, 10 pt panel labels.
    - Okabe-Ito colour-blind-safe palette.
    - Vector PDF for submission, 300-dpi PNG for preview.
    - Output: audit/reports/figures/workflow_diagram.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

# Okabe-Ito palette.
C = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#7A7A7A",
    "lgrey":  "#ECECEC",
    "mgrey":  "#CCCCCC",
    "white":  "#FFFFFF",
}

MOD = {
    "BEH":  C["orange"],
    "EEG":  C["sky"],
    "MRI":  C["green"],
    "DTI":  C["purple"],
    "DL":   C["yellow"],
    "proc": C["blue"],
    "out":  C["verm"],
}


# ---------- helpers -----------------------------------------------------------

def box(ax, x, y, w, h, text="", *, fc=C["lgrey"], ec=C["black"], lw=0.6,
        fs=8, weight="normal", italic=False, color=None,
        ha="center", va="center", rounding=0.012, pad=0.012):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    if text:
        if color is None:
            color = C["black"]
        if va == "center":
            ty = y + h / 2
        elif va == "top":
            ty = y + h - 0.018
        else:  # bottom
            ty = y + 0.018
        if ha == "center":
            tx = x + w / 2
        elif ha == "left":
            tx = x + 0.014
        else:
            tx = x + w - 0.014
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs,
                weight=weight, style="italic" if italic else "normal",
                color=color, zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=C["black"], lw=0.9, mut=8,
          style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def panel_label(ax, txt):
    ax.text(0.005, 0.985, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, weight="bold")


def panel_header(ax, txt):
    ax.text(0.5, 0.965, txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, weight="bold")


def clean(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def stacked_box(ax, x, y, w, h, title, sub, *, fc, fs_t=7.7, fs_s=6.9):
    """Box with two-line label: bold title above, small subtitle below."""
    box(ax, x, y, w, h, fc=fc)
    ax.text(x + w / 2, y + h * 0.62, title,
            ha="center", va="center", fontsize=fs_t, weight="bold")
    ax.text(x + w / 2, y + h * 0.27, sub,
            ha="center", va="center", fontsize=fs_s, color=C["black"])


# ---------- panels ------------------------------------------------------------

def draw_panel_a(ax):
    panel_label(ax, "A")
    panel_header(ax, "Cohort and modality blocks")

    box(ax, 0.04, 0.78, 0.92, 0.10,
        "Cohort: $n$ = 427 adults · age 19–80 yr · 5-fold preregistered (hash db1510dd…)",
        fc=C["lgrey"], fs=8.3, weight="bold")

    # Top row: source modalities (4 boxes)
    src = [
        ("BEH",       "199 + sex",  MOD["BEH"]),
        ("EEG",       "509 cols",   MOD["EEG"]),
        ("MRI",       "1266 cols",  MOD["MRI"]),
        ("DTI_ROI",   "90 cols",    MOD["DTI"]),
    ]
    n = len(src); pad = 0.04; gap = 0.012
    w = (1 - 2 * pad - (n - 1) * gap) / n; h = 0.16
    yt = 0.55
    for i, (name, sub, col) in enumerate(src):
        x = pad + i * (w + gap)
        stacked_box(ax, x, yt, w, h, name, sub, fc=col)

    # Bottom row: derived DL features (6 boxes)
    der = [
        ("DL_SFCN",        "65 cols",   MOD["DL"]),
        ("DL_Pyment",      "6 cols",    MOD["DL"]),
        ("DL_RSFMRI_FC",   "1 scalar",  MOD["DL"]),
        ("EEG_PSD_RAW*",   "150 cols",  MOD["EEG"]),
        ("DL_DTI_OOF*",    "1 scalar",  MOD["DTI"]),
        ("DL_DTI_VOXEL*",  "3 scalars", MOD["DTI"]),
    ]
    n2 = len(der)
    w2 = (1 - 2 * pad - (n2 - 1) * gap) / n2
    yb = 0.32
    for i, (name, sub, col) in enumerate(der):
        x = pad + i * (w2 + gap)
        stacked_box(ax, x, yb, w2, h, name, sub, fc=col, fs_t=7.4)

    ax.text(0.04, 0.18,
            "* Candidate blocks evaluated only in addendum 1 "
            "(information-gain frontier).",
            fontsize=7.4, style="italic", color=C["grey"])
    ax.text(0.04, 0.08,
            "Audit reads a frozen feature cache; results.tsv is path-guarded "
            "and the (subject, fold) hash is verified at every load.",
            fontsize=7.4, style="italic", color=C["grey"])
    clean(ax)


def draw_panel_b(ax):
    panel_label(ax, "B")
    panel_header(ax, "Stacked per-block ridge with fold-aligned out-of-fold predictions")

    # Five horizontal stages — narrower boxes, tighter spacing
    yb = 0.50; hb = 0.32
    stages = [
        (0.030, 0.150, "Per-block\npipeline",
         "Impute · Scale ·\nRidgeCV", C["lgrey"]),
        (0.215, 0.150, "Per-block OOF",
         "(427 × 1)\nfold-aligned", C["white"]),
        (0.400, 0.150, "Stack Z",
         "(427 × $n_b$)\none col / block", C["lgrey"]),
        (0.585, 0.150, "Meta-stacker",
         "RidgeCV on Z\nouter fold $k$", MOD["proc"]),
        (0.770, 0.180, "Final OOF $\\hat y$",
         "(427)\nMAE · R² · ρ", MOD["out"]),
    ]
    for (x, w, title, sub, fc) in stages:
        text_color = C["white"] if fc in (MOD["proc"], MOD["out"]) else C["black"]
        box(ax, x, yb, w, hb, fc=fc)
        ax.text(x + w / 2, yb + hb * 0.72, title,
                ha="center", va="center", fontsize=7.6, weight="bold",
                color=text_color)
        ax.text(x + w / 2, yb + hb * 0.32, sub,
                ha="center", va="center", fontsize=6.8,
                color=text_color)

    # arrows between stages
    edges = [
        (0.180, 0.215),
        (0.365, 0.400),
        (0.550, 0.585),
        (0.735, 0.770),
    ]
    for x1, x2 in edges:
        arrow(ax, x1, yb + hb / 2, x2, yb + hb / 2, mut=7)

    # Residual loop box centred below stages
    box(ax, 0.250, 0.10, 0.500, 0.22, fc=C["lgrey"])
    ax.text(0.50, 0.265, "Residual stacker (inner KFold = 5, seed-keyed)",
            ha="center", va="center", fontsize=7.5, weight="bold")
    ax.text(0.50, 0.205,
            "RidgeCV on meta nested-CV residuals; sum added to outer pred.",
            ha="center", va="center", fontsize=6.9)
    ax.text(0.50, 0.150,
            "Clip = none (preregister §8 Option A).",
            ha="center", va="center", fontsize=6.9, style="italic",
            color=C["grey"])

    # Loop arrows: meta -> residual -> final
    arrow(ax, 0.660, yb, 0.55, 0.32, color=C["grey"], mut=6)
    arrow(ax, 0.75, 0.32, 0.86, yb, color=C["grey"], mut=6)

    clean(ax)


def draw_panel_c(ax):
    panel_label(ax, "C")
    panel_header(ax, "Addendum 1 — information-gain frontier")

    rows = [
        ("01b", "baseline_v2",
         "BEH+EEG+MRI+DTI_ROI+SFCN+Pyment+RSFMRI_FC"),
        ("02b", "+ DL_DTI_OOF_AGE",
         "scalar: external 90-ROI DTI ridge"),
        ("03b", "+ DL_DTI_VOXEL_RIDGE",
         "3 voxel-ridge OOFs (FA, MD, FA+MD)"),
        ("04b", "+ EEG_PSD_RAW",
         "150 log band-powers (30 ch × 5 bands)"),
        ("05b", "drop_fa_cortical",
         "sensitivity: drop 172 *_FA cortical cols"),
    ]
    xc = 0.05; wc = 0.90; hc = 0.095
    y0 = 0.86; dy = 0.108
    for i, (cid, title, sub) in enumerate(rows):
        y = y0 - i * dy
        fc = C["lgrey"] if i == 0 else C["white"]
        box(ax, xc, y, wc, hc, fc=fc)
        ax.text(xc + 0.015, y + hc * 0.65, f"{cid}  {title}",
                ha="left", va="center", fontsize=7.5, weight="bold")
        ax.text(xc + 0.015, y + hc * 0.28, sub,
                ha="left", va="center", fontsize=6.8)
        if i > 0:
            xa = xc + wc / 2
            arrow(ax, xa, y0 - (i - 1) * dy, xa, y + hc, mut=6)

    # Bottom banner: stats spec + result summary
    box(ax, 0.05, 0.04, 0.90, 0.21, fc=MOD["out"])
    ax.text(0.50, 0.215,
            "Paired bootstrap   $N$ = 10 000",
            ha="center", va="center", fontsize=7.4,
            color=C["white"], weight="bold")
    ax.text(0.50, 0.165,
            "seeds 20260425 / 20260426  →  §6 verdict",
            ha="center", va="center", fontsize=7.2, color=C["white"])
    ax.text(0.50, 0.105,
            "Result: 4 adjacent pairs and aggregate",
            ha="center", va="center", fontsize=7.0,
            color=C["white"], weight="bold")
    ax.text(0.50, 0.060,
            "01b → 04b OVERTURNED at both seeds",
            ha="center", va="center", fontsize=7.0, color=C["white"])
    clean(ax)


def draw_panel_d(ax):
    panel_label(ax, "D")
    panel_header(ax, "Addendum 2 — acquisition-design leave-one-out")

    # Headers
    box(ax, 0.05, 0.86, 0.42, 0.075, "MRI (1266 cols)",
        fc=MOD["MRI"], fs=7.8, weight="bold")
    box(ax, 0.53, 0.86, 0.42, 0.075, "EEG (509 cols)",
        fc=MOD["EEG"], fs=7.8, weight="bold")

    # Trimmed labels (drop the _NULL acquisition marker; LANGUAGE_* shortened).
    mri_doms = [
        ("STRUCTURE",            684),
        ("RESTING",              374),
        ("MEMORY_MST",           105),
        ("MOTOR_GFORCE",          64),
        ("LANG_SPEECHCOMP",       21),
        ("LANG_WORDNAME",         18),
    ]
    eeg_doms = [
        ("MOTOR_GOFITTS",        216),
        ("MEMORY_OSPAN",         120),
        ("RESTING",               81),
        ("MEMORY_EXCLUSION",      60),
        ("MOTOR_BILPRESS",        32),
    ]

    yb = 0.79; rh = 0.052; gap = 0.008
    for i, (name, k) in enumerate(mri_doms):
        y = yb - i * (rh + gap)
        box(ax, 0.05, y, 0.42, rh, fc=C["white"])
        ax.text(0.065, y + rh / 2, name,
                ha="left", va="center", fontsize=6.8)
        ax.text(0.445, y + rh / 2, str(k),
                ha="right", va="center", fontsize=7.0, weight="bold")
    for i, (name, k) in enumerate(eeg_doms):
        y = yb - i * (rh + gap)
        box(ax, 0.53, y, 0.42, rh, fc=C["white"])
        ax.text(0.545, y + rh / 2, name,
                ha="left", va="center", fontsize=6.8)
        ax.text(0.925, y + rh / 2, str(k),
                ha="right", va="center", fontsize=7.0, weight="bold")

    # Sign convention banner
    box(ax, 0.05, 0.21, 0.90, 0.11, fc=C["lgrey"])
    ax.text(0.50, 0.290,
            "LOO  ·  drop one domain  →  re-run pipeline",
            ha="center", va="center", fontsize=7.2, weight="bold")
    ax.text(0.50, 0.235,
            "$\\Delta$MAE < 0  ⇒  domain carries unique age signal",
            ha="center", va="center", fontsize=6.9, style="italic")

    # Output banner
    box(ax, 0.05, 0.04, 0.90, 0.14, fc=MOD["out"])
    ax.text(0.50, 0.150,
            "13 LOO comparisons × 2 seeds",
            ha="center", va="center", fontsize=7.3,
            color=C["white"], weight="bold")
    ax.text(0.50, 0.105,
            "→ outputs/addendum_2/stats_{seed}.csv",
            ha="center", va="center", fontsize=6.9,
            color=C["white"])
    ax.text(0.50, 0.060,
            "per-domain $\\Delta$MAE, 95 % CI, per-fold $t$",
            ha="center", va="center", fontsize=6.9, color=C["white"])
    clean(ax)


def draw_panel_e(ax):
    """Addendum 3 (planned) — feature-granularity comparison."""
    panel_label(ax, "E")
    panel_header(ax, "Addendum 3 (planned) — feature-granularity: voxelwise vs ROI task-fMRI")

    # Pending banner (top-right corner)
    box(ax, 0.74, 0.84, 0.22, 0.10,
        "PLANNED · pending\nexternal voxel CLI",
        fc=C["yellow"], fs=7.0, weight="bold", italic=True)

    # Two-column comparison
    box(ax, 0.04, 0.62, 0.42, 0.18, fc=MOD["MRI"])
    ax.text(0.25, 0.755, "ROI-level task-fMRI",
            ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(0.25, 0.700, "(current baseline 00c)",
            ha="center", va="center", fontsize=7.0)
    ax.text(0.25, 0.645,
            "block ridge over BETAs;\n"
            "MRI-block input",
            ha="center", va="center", fontsize=6.8)

    box(ax, 0.54, 0.62, 0.42, 0.18, fc=MOD["DTI"])
    ax.text(0.75, 0.755, "Voxel-level task-fMRI  (new)",
            ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(0.75, 0.700, "per-task voxel ridge → 1 OOF scalar",
            ha="center", va="center", fontsize=7.0)
    ax.text(0.75, 0.645,
            "parallel to DL_DTI_VOXEL_RIDGE;\n"
            "added as DL_*_VOXEL block(s)",
            ha="center", va="center", fontsize=6.8)

    arrow(ax, 0.46, 0.71, 0.54, 0.71, mut=8)

    # Per-task table
    tasks = [
        ("MOTOR_GFORCE",          64, "voxel TBSS-style ridge"),
        ("MEMORY_MST",           105, "voxel TBSS-style ridge"),
        ("LANGUAGE_SPEECHCOMP",   21, "voxel TBSS-style ridge"),
        ("LANGUAGE_WORDNAME",     18, "voxel TBSS-style ridge"),
    ]
    box(ax, 0.04, 0.30, 0.92, 0.27, fc=C["white"])
    ax.text(0.50, 0.535,
            "Per task: voxelwise ridge OOF scalar → block-augmentation candidate",
            ha="center", va="center", fontsize=7.5, weight="bold")
    rh = 0.045
    for i, (name, k_roi, kind) in enumerate(tasks):
        y = 0.475 - i * (rh + 0.005)
        box(ax, 0.06, y, 0.88, rh, fc=C["lgrey"], lw=0.4)
        ax.text(0.075, y + rh / 2, name,
                ha="left", va="center", fontsize=7.2, weight="bold")
        ax.text(0.42, y + rh / 2, f"ROI: {k_roi} cols",
                ha="left", va="center", fontsize=7.0)
        ax.text(0.62, y + rh / 2, "→",
                ha="center", va="center", fontsize=8.0, color=C["grey"])
        ax.text(0.78, y + rh / 2, kind,
                ha="center", va="center", fontsize=7.0, style="italic")

    # Test spec banner
    box(ax, 0.04, 0.04, 0.92, 0.21, fc=MOD["out"])
    ax.text(0.50, 0.205,
            "Test (paired bootstrap, $N$ = 10 000, seeds 20260425 / 20260426)",
            ha="center", va="center", fontsize=7.4, color=C["white"],
            weight="bold")
    ax.text(0.50, 0.150,
            "$\\Delta$MAE = MAE$_{ROI}$ − MAE$_{ROI+voxel}$  ·  "
            "+0.02 yr §6 floor  ·  CI > 0 at both seeds",
            ha="center", va="center", fontsize=7.0, color=C["white"])
    ax.text(0.50, 0.095,
            "Confirms only if any per-task voxelwise augmentation clears the floor.",
            ha="center", va="center", fontsize=7.0, color=C["white"])
    ax.text(0.50, 0.050,
            "Otherwise: methodological footnote — multimodal plateau persists at $n$ = 427.",
            ha="center", va="center", fontsize=6.9, color=C["white"],
            style="italic")
    clean(ax)


# ---------- assembly ----------------------------------------------------------

def main():
    fig = plt.figure(figsize=(7.0, 11.5))   # 178 mm × 292 mm
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[0.80, 0.85, 1.40, 1.20],
        width_ratios=[1, 1],
        left=0.025, right=0.975, top=0.99, bottom=0.012,
        wspace=0.07, hspace=0.11,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])
    ax_e = fig.add_subplot(gs[3, :])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)
    draw_panel_e(ax_e)

    pdf_path = OUT_DIR / "workflow_diagram.pdf"
    png_path = OUT_DIR / "workflow_diagram.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
