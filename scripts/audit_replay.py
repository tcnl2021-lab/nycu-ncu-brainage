"""Replay every config in configs/frontier_last10.yaml under the clean
protocol, for both audit seeds, and write OOF CSVs to outputs/oof/.

Path guard: refuses to open `codes/brain_age_2026/tuning/results.tsv`.
The frontier window was extracted into the YAML once during the freeze
step; this script never touches results.tsv.
"""
from __future__ import annotations

import _audit_common as ac

import argparse
import json
from pathlib import Path
import time

import yaml

from train_clean import BlockSource, run_config, write_oof_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=str(ac.AUDIT_ROOT / "configs" / "frontier_last10.yaml"))
    ap.add_argument("--out-dir", default=str(ac.AUDIT_ROOT / "outputs" / "oof"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[ac.PRIMARY_SEED, ac.SENSITIVITY_SEED])
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip configs whose output CSV already exists.")
    args = ap.parse_args()

    cfgs = yaml.safe_load(Path(args.configs).read_text())["configs"]
    print(f"loaded {len(cfgs)} configs from {args.configs}")
    print(f"seeds: {args.seeds}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = BlockSource()  # also verifies fold hash on first .raw access

    summary_rows = []
    for cfg in cfgs:
        for seed in args.seeds:
            out_path = out_dir / f"oof_pred_{cfg['id']}_{seed}.csv"
            if args.skip_existing and out_path.exists():
                print(f"[skip] {out_path.name} exists")
                continue
            t0 = time.time()
            print(f"\n=== config {cfg['id']} ({cfg['name']})  seed={seed} ===")
            print(f"  blocks: {cfg['blocks']}")
            res = run_config(cfg["blocks"], seed=seed, source=src)
            write_oof_csv(out_path, res, cfg["id"], cfg["name"])
            dt = time.time() - t0
            row = {
                "config_id": cfg["id"],
                "name": cfg["name"],
                "seed": seed,
                "mae": res["metrics"]["mae"],
                "rmse": res["metrics"]["rmse"],
                "r2": res["metrics"]["r2"],
                "pearson": res["metrics"]["pearson"],
                "fold_mae_mean": res["metrics"]["fold_mae_mean"],
                "fold_mae_std": res["metrics"]["fold_mae_std"],
                "fold_mae": res["metrics"]["fold_mae"],
                "seconds": dt,
                "out": str(out_path),
            }
            summary_rows.append(row)
            print(f"  wrote {out_path.name}  ({dt:.1f}s)")

    summary_path = out_dir / "_replay_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    else:
        existing = []
    # Replace any rows for the (id, seed) pairs we just produced.
    seen = {(r["config_id"], r["seed"]) for r in summary_rows}
    merged = [r for r in existing if (r["config_id"], r["seed"]) not in seen] + summary_rows
    summary_path.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {summary_path}  ({len(merged)} rows total, {len(summary_rows)} new)")


if __name__ == "__main__":
    main()
