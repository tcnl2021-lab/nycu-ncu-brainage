"""Paired bootstrap + per-fold paired t-test for adjacent frontier configs.

Implements preregister §5–§7. For each adjacent pair (parent A, child B)
in `configs/frontier_last10.yaml`, computes:

  - Paired bootstrap (N=10 000, resampler seed 20260425) on per-subject
    absolute errors. Reports mean(MAE(A) − MAE(B)), 95 % CI, and
    p(ΔMAE ≤ 0).
  - Per-fold paired t-test (n=5).
  - Verdict ∈ {confirmed, fragile, overturned} per §6 acceptance rule:
        confirmed:  CI excludes 0 AND point estimate ≥ +0.02 yr
        fragile:    CI excludes 0 BUT point estimate < +0.02 yr
        overturned: CI includes 0 (or favours the parent)

Writes outputs/stats_{seed}.csv per audit seed. Path-guarded.
"""
from __future__ import annotations

import _audit_common as ac

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
    if df["subject_id"].is_unique is False:
        raise RuntimeError(f"non-unique subject_id in {path}")
    return df


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int,
                     seed: int) -> tuple[float, float, float, float]:
    """Resample subjects with replacement; report d = MAE(A) − MAE(B)."""
    rng = np.random.default_rng(seed)
    n = len(err_a)
    if len(err_b) != n:
        raise ValueError("paired bootstrap: error arrays must have same length")
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(err_a[idx]) - np.mean(err_b[idx])
    mean = float(deltas.mean())
    lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    p_le_0 = float(np.mean(deltas <= 0.0))
    return mean, lo, hi, p_le_0


def per_fold_t(err_a: np.ndarray, err_b: np.ndarray, fold_a: np.ndarray,
               fold_b: np.ndarray) -> tuple[float, float, list[float]]:
    """Per-fold paired t on MAE differences (n=5)."""
    if not np.array_equal(fold_a, fold_b):
        raise ValueError("fold ids must match between A and B")
    fold_dmae = []
    for k in range(ac.N_SPLITS):
        m = fold_a == k
        fold_dmae.append(float(np.mean(np.abs(err_a[m]).mean() - np.abs(err_b[m]).mean())))
    arr = np.array([
        float(np.abs(err_a[fold_a == k]).mean() - np.abs(err_b[fold_a == k]).mean())
        for k in range(ac.N_SPLITS)
    ])
    if np.allclose(arr, 0.0):
        return 0.0, 1.0, arr.tolist()
    t, p = stats.ttest_1samp(arr, 0.0)
    return float(t), float(p), arr.tolist()


def verdict(point: float, lo: float, hi: float) -> str:
    excludes_zero = (lo > 0.0) or (hi < 0.0)
    if not excludes_zero:
        return "overturned"
    if hi < 0:
        return "overturned"   # CI favours the parent (improvement is negative)
    if point >= ac.ACCEPT_DELTA:
        return "confirmed"
    return "fragile"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=str(ac.AUDIT_ROOT / "configs" / "frontier_last10.yaml"))
    ap.add_argument("--oof-dir", default=str(ac.AUDIT_ROOT / "outputs" / "oof"))
    ap.add_argument("--out-dir", default=str(ac.AUDIT_ROOT / "outputs"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[ac.PRIMARY_SEED, ac.SENSITIVITY_SEED])
    ap.add_argument("--bootstrap-seed", type=int, default=ac.PRIMARY_SEED,
                    help="Resampler seed (preregister §5: fixed at 20260425).")
    ap.add_argument("--n-boot", type=int, default=ac.N_BOOTSTRAP)
    args = ap.parse_args()

    cfgs = yaml.safe_load(Path(args.configs).read_text())["configs"]
    cfg_by_id = {c["id"]: c for c in cfgs}

    pairs = []
    for c in cfgs:
        if c.get("parent"):
            pairs.append((c["parent"], c["id"]))
    print(f"comparing {len(pairs)} adjacent pairs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = Path(args.oof_dir)

    for seed in args.seeds:
        rows = []
        for parent_id, child_id in pairs:
            parent = cfg_by_id[parent_id]
            child = cfg_by_id[child_id]
            df_a = load_oof(oof_dir / f"oof_pred_{parent_id}_{seed}.csv")
            df_b = load_oof(oof_dir / f"oof_pred_{child_id}_{seed}.csv")
            if not (df_a["subject_id"] == df_b["subject_id"]).all():
                raise RuntimeError(f"subject_id mismatch {parent_id} vs {child_id}")
            if not (df_a["true_age"].round(4) == df_b["true_age"].round(4)).all():
                raise RuntimeError(f"true_age mismatch {parent_id} vs {child_id}")
            err_a = (df_a["pred"] - df_a["true_age"]).abs().to_numpy()
            err_b = (df_b["pred"] - df_b["true_age"]).abs().to_numpy()
            fold_a = df_a["fold"].to_numpy()
            fold_b = df_b["fold"].to_numpy()

            mae_a = float(err_a.mean())
            mae_b = float(err_b.mean())
            d_point_observed = mae_a - mae_b  # >0 means child is better

            d_boot, ci_lo, ci_hi, p_le_0 = paired_bootstrap(
                err_a, err_b, n_boot=args.n_boot, seed=args.bootstrap_seed
            )
            t_stat, t_p, fold_dmae = per_fold_t(err_a, err_b, fold_a, fold_b)
            v = verdict(d_point_observed, ci_lo, ci_hi)

            rows.append({
                "comparison": f"{parent_id}->{child_id}",
                "parent_name": parent["name"],
                "child_name": child["name"],
                "no_new_information": bool(child.get("no_new_information", False)),
                "mae_parent_clean": mae_a,
                "mae_child_clean": mae_b,
                "dMAE_observed": d_point_observed,
                "dMAE_boot_mean": d_boot,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_le_zero": p_le_0,
                "t_stat": t_stat,
                "t_p": t_p,
                "fold_dmae_0": fold_dmae[0],
                "fold_dmae_1": fold_dmae[1],
                "fold_dmae_2": fold_dmae[2],
                "fold_dmae_3": fold_dmae[3],
                "fold_dmae_4": fold_dmae[4],
                "verdict": v,
                "pre_audit_mae_parent": parent.get("pre_audit_val_mae"),
                "pre_audit_mae_child": child.get("pre_audit_val_mae"),
            })
            print(f"  {parent_id}->{child_id} "
                  f"({parent['name']} -> {child['name']}): "
                  f"dMAE={d_point_observed:+.6f} "
                  f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
                  f"t={t_stat:+.3f} p={t_p:.4f} -> {v}")

        out_path = out_dir / f"stats_{seed}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
