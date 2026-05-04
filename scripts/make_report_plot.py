"""Generate the audit report's MAE-per-frontier-step plot.

Two lines:
  - clean-protocol pooled OOF MAE (this audit, primary seed)
  - pre-audit val_mae from results.tsv (extracted into the YAML at freeze)

Output: reports/figs/frontier_mae.png
"""
from __future__ import annotations

import _audit_common as ac

import argparse
from pathlib import Path

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=str(ac.AUDIT_ROOT / "configs" / "frontier_last10.yaml"))
    ap.add_argument("--summary", default=str(ac.AUDIT_ROOT / "outputs" / "oof" / "_replay_summary.json"))
    ap.add_argument("--out", default=str(ac.AUDIT_ROOT / "reports" / "figs" / "frontier_mae.png"))
    ap.add_argument("--seed", type=int, default=ac.PRIMARY_SEED)
    args = ap.parse_args()

    cfgs = yaml.safe_load(Path(args.configs).read_text())["configs"]
    rows = json.loads(Path(args.summary).read_text())
    by_id_seed = {(r["config_id"], r["seed"]): r for r in rows}

    ids = [c["id"] for c in cfgs]
    labels = [f"{c['id']}\n{c['name']}" for c in cfgs]
    pre_audit_mae = [c["pre_audit_val_mae"] for c in cfgs]
    clean_mae = [by_id_seed[(c["id"], args.seed)]["mae"] for c in cfgs]
    clean_mae_alt = [by_id_seed[(c["id"], ac.SENSITIVITY_SEED)]["mae"]
                     for c in cfgs if (c["id"], ac.SENSITIVITY_SEED) in by_id_seed]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(ids)))

    ax.plot(x, pre_audit_mae, marker="o", color="#888", linestyle="--",
            label="pre-audit val_mae (results.tsv)")
    ax.plot(x, clean_mae, marker="o", color="#1f77b4",
            label=f"clean protocol (seed {args.seed})")
    if len(clean_mae_alt) == len(ids):
        ax.plot(x, clean_mae_alt, marker="x", color="#1f77b4", alpha=0.5,
                linestyle=":", label=f"clean protocol (seed {ac.SENSITIVITY_SEED})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Pooled OOF MAE (years)")
    ax.set_title("Brain-age frontier: pre-audit vs clean-protocol MAE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
