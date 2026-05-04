"""Aggregate paired-bootstrap test for preregister_addendum_1.

Implements §5 of preregister_addendum_1.md: the cumulative gain
01b -> 04b under the same paired-bootstrap CI used for adjacent pairs.
Reuses paired_bootstrap, per_fold_t, verdict, and load_oof from
audit_stats.py so the statistical machinery is byte-identical.
"""
from __future__ import annotations

import _audit_common as ac
import argparse
from pathlib import Path

import pandas as pd

from audit_stats import load_oof, paired_bootstrap, per_fold_t, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ac.AUDIT_ROOT / "outputs" / "addendum_1" / "oof"))
    ap.add_argument("--out-dir", default=str(ac.AUDIT_ROOT / "outputs" / "addendum_1"))
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[ac.PRIMARY_SEED, ac.SENSITIVITY_SEED])
    ap.add_argument("--bootstrap-seed", type=int, default=ac.PRIMARY_SEED)
    ap.add_argument("--n-boot", type=int, default=ac.N_BOOTSTRAP)
    ap.add_argument("--from-id", default="01b")
    ap.add_argument("--to-id", default="04b")
    args = ap.parse_args()

    oof_dir = Path(args.oof_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        df_a = load_oof(oof_dir / f"oof_pred_{args.from_id}_{seed}.csv")
        df_b = load_oof(oof_dir / f"oof_pred_{args.to_id}_{seed}.csv")
        if not (df_a["subject_id"] == df_b["subject_id"]).all():
            raise RuntimeError("subject_id mismatch")
        err_a = (df_a["pred"] - df_a["true_age"]).abs().to_numpy()
        err_b = (df_b["pred"] - df_b["true_age"]).abs().to_numpy()
        fold_a = df_a["fold"].to_numpy()
        fold_b = df_b["fold"].to_numpy()

        mae_a = float(err_a.mean())
        mae_b = float(err_b.mean())
        d_obs = mae_a - mae_b

        d_boot, ci_lo, ci_hi, p_le_0 = paired_bootstrap(
            err_a, err_b, n_boot=args.n_boot, seed=args.bootstrap_seed
        )
        t_stat, t_p, fold_dmae = per_fold_t(err_a, err_b, fold_a, fold_b)
        v = verdict(d_obs, ci_lo, ci_hi)

        row = {
            "comparison": f"{args.from_id}->{args.to_id} (aggregate)",
            "from_id": args.from_id,
            "to_id": args.to_id,
            "mae_from_clean": mae_a,
            "mae_to_clean": mae_b,
            "dMAE_observed": d_obs,
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
        }
        out_path = out_dir / f"stats_aggregate_{seed}.csv"
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"  {args.from_id}->{args.to_id} seed={seed}: "
              f"dMAE={d_obs:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] -> {v}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
