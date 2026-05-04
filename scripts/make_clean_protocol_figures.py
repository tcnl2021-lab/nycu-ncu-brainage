"""Regenerate manuscript scatter and PAD figures from the audit clean-protocol OOF.

Source: outputs/addendum_1/oof/oof_pred_05b_<seed>.csv (config 05b =
drop_fa_cortical_sensitivity, the final addendum-1 baseline; vetted
val_MAE = 2.965 yr per audit_report_2026-04-25.md).

Outputs:
  /media/DATA3/quanta-brainage-manuscript/figures/scatter.pdf
  /media/DATA3/quanta-brainage-manuscript/figures/pad_vs_age.pdf
  /media/DATA3/quanta-brainage-manuscript/figures/pipeline.pdf

The pipeline.pdf depicts the CLEAN-PROTOCOL stack (preregister.md §8),
not the engineering ensemble. Specifically: per-block SimpleImputer(median)
→ StandardScaler → RidgeCV(alphas=logspace(-3,3,21)) inside outer-fold
pipelines; outer-fold-aligned OOF columns assembled into stack matrix Z;
RidgeCV meta + RidgeCV residual (inner-CV nested); no clip.

Run from inside the audit/ sub-repo:
  python scripts/make_clean_protocol_figures.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
OOF_CSV = AUDIT_ROOT / "outputs" / "addendum_1" / "oof" / "oof_pred_05b_20260425.csv"
SENS_CSV = AUDIT_ROOT / "outputs" / "addendum_1" / "oof" / "oof_pred_05b_20260426.csv"
FIG_DIR = Path("/media/DATA3/quanta-brainage-manuscript/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_oof(csv: Path):
    df = pd.read_csv(csv).sort_values("subject_id").reset_index(drop=True)
    return df["pred"].to_numpy(np.float32), df["true_age"].to_numpy(np.float32), df["fold"].to_numpy(int)


def metrics(y, p):
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    r = float(np.corrcoef(y, p)[0, 1])
    r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((y - y.mean()) ** 2))
    return mae, rmse, r, r2


def zhang_correct(oof: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, window: float = 8.0):
    """Zhang age-bias correction: per-subject leave-fold-out local-mean adjustment."""
    out = np.zeros_like(oof, dtype=np.float32)
    for i in range(len(oof)):
        fi = fold_ids[i]
        age = y[i]
        other = (fold_ids != fi) & (np.abs(y - age) <= window / 2.0)
        if other.sum() == 0:
            out[i] = oof[i] - age
        else:
            local_pred = float(oof[other].mean())
            out[i] = oof[i] - local_pred
    return out


def fig_scatter(y, p, mae, r2):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    age_min, age_max = float(y.min()), float(y.max())
    ax.plot([age_min, age_max], [age_min, age_max], "--", color="0.6", lw=1.0, zorder=1)
    ax.scatter(y, p, s=14, alpha=0.55, edgecolor="none", color="#1f77b4", zorder=2)
    slope, intercept, r_val, _, _ = stats.linregress(y, p)
    xs = np.array([age_min, age_max])
    ax.plot(xs, slope * xs + intercept, "-", color="#d62728", lw=1.2, zorder=3, label="OLS fit")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title(f"Audit clean-protocol model (config 05b)\nval MAE = {mae:.3f} yr · R² = {r2:.3f} · r = {r_val:.3f}")
    ax.set_xlim(age_min - 2, age_max + 2)
    ax.set_ylim(age_min - 2, age_max + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    out = FIG_DIR / "scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pad_vs_age(y, p, fold_ids):
    pad_raw = p - y
    pad_corr = zhang_correct(p, y, fold_ids)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, pad, ttl in zip(axes, [pad_raw, pad_corr], ["Raw PAD", "Zhang-corrected PAD"]):
        ax.scatter(y, pad, s=12, alpha=0.55, edgecolor="none", color="#2ca02c")
        slope, intercept, r_val, p_val, _ = stats.linregress(y, pad)
        xs = np.array([float(y.min()), float(y.max())])
        ax.plot(xs, slope * xs + intercept, "-", color="#d62728", lw=1.2)
        ax.axhline(0, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel("Chronological age (years)")
        ax.set_title(f"{ttl}\nslope = {slope:+.3f}, r = {r_val:+.3f}, p = {p_val:.3g}")
    axes[0].set_ylabel("PAD (predicted − chronological, years)")
    fig.suptitle("PAD vs age — audit clean-protocol model (config 05b)", fontsize=11, y=1.02)
    out = FIG_DIR / "pad_vs_age.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pipeline():
    """Architecture diagram for the audit clean-protocol model (preregister.md §8)."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, color, text, fontsize=8, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                                              facecolor=color, edgecolor="0.3", linewidth=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                fontweight=("bold" if bold else "normal"))

    def arrow(x0, y0, x1, y1, color="0.4"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    blocks = [
        ("BEH",          "#fef3c7"),
        ("EEG",          "#fde68a"),
        ("MRI (sMRI)",   "#ddd6fe"),
        ("DTI ROI",      "#c7d2fe"),
        ("DL_SFCN",      "#bfdbfe"),
        ("DL_PYMENT",    "#a5f3fc"),
        ("DL_RSFMRI_FC", "#bbf7d0"),
    ]
    block_w = 1.2
    block_h = 0.55
    block_y = 5.7
    spacing = 1.32
    n = len(blocks)
    total_w = (n - 1) * spacing + block_w
    block_x0 = (10 - total_w) / 2

    for i, (name, color) in enumerate(blocks):
        x = block_x0 + i * spacing
        box(x, block_y, block_w, block_h, color, name, fontsize=8, bold=True)

    pipe_y = 4.4
    pipe_h = 0.7
    for i, (_, color) in enumerate(blocks):
        x = block_x0 + i * spacing
        box(x, pipe_y, block_w, pipe_h, "#f5f5f5",
            "SimpleImputer\n(median)\n→ StandardScaler\n→ RidgeCV", fontsize=5.6)
        arrow(x + block_w / 2, block_y, x + block_w / 2, pipe_y + pipe_h)

    note_y = pipe_y - 0.45
    ax.text(5, note_y, "Per-block fit on outer-training fold ONLY (no leakage). "
                       "Outer-fold-aligned OOF column → stack matrix Z.",
            ha="center", va="center", fontsize=7.5, style="italic", color="0.3")

    z_y = 3.0
    z_h = 0.6
    z_x = 3.0
    z_w = 4.0
    box(z_x, z_y, z_w, z_h, "#fff7ed",
        "Stack matrix Z  (n_subjects × n_blocks of OOF predictions)", fontsize=8.5, bold=True)
    for i in range(n):
        x = block_x0 + i * spacing + block_w / 2
        arrow(x, pipe_y, z_x + z_w / 2, z_y + z_h)

    meta_y = 1.85
    meta_h = 0.55
    box(2.4, meta_y, 2.4, meta_h, "#fce7f3",
        "Meta: RidgeCV(Z)\n(outer-fold-aligned)", fontsize=8, bold=True)
    box(5.2, meta_y, 2.4, meta_h, "#fbcfe8",
        "Residual: RidgeCV(inner-CV residuals)", fontsize=8, bold=True)
    arrow(5.0, z_y, 3.6, meta_y + meta_h)
    arrow(5.0, z_y, 6.4, meta_y + meta_h)

    pred_y = 0.55
    pred_h = 0.55
    box(3.4, pred_y, 3.2, pred_h, "#dcfce7",
        "Predicted age = meta + residual\n(NO clip, NO bespoke FE)",
        fontsize=8.5, bold=True)
    arrow(3.6, meta_y, 4.4, pred_y + pred_h)
    arrow(6.4, meta_y, 5.6, pred_y + pred_h)

    ax.text(5, 6.55, "Clean-protocol stack (preregister-v1 §8)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(5, 0.10, "vetted val_MAE = 2.965 yr · all 10 frontier-step claims overturned · "
                     "aggregate gain (+0.15 yr, 95% CI [+0.027, +0.272]) confirmed",
            ha="center", va="bottom", fontsize=7.8, style="italic", color="0.25")

    out = FIG_DIR / "pipeline.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    if not OOF_CSV.exists():
        sys.exit(f"missing OOF: {OOF_CSV}")
    p, y, folds = load_oof(OOF_CSV)
    mae, rmse, r, r2 = metrics(y, p)
    print(f"primary seed 20260425: n={len(y)}  MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}  R²={r2:.4f}")

    if SENS_CSV.exists():
        ps, ys, _ = load_oof(SENS_CSV)
        mae_s, _, r_s, r2_s = metrics(ys, ps)
        print(f"sensitivity seed 20260426: MAE={mae_s:.4f}  r={r_s:.4f}  R²={r2_s:.4f}")
        print(f"seed delta: ΔMAE = {mae_s - mae:+.4f}")

    fig_scatter(y, p, mae, r2)
    fig_pad_vs_age(y, p, folds)
    fig_pipeline()


if __name__ == "__main__":
    main()
