"""Clean-protocol per-config OOF predictor.

Implements preregister §8 verbatim:

    Per-block pipeline = SimpleImputer(strategy='median')
                       -> StandardScaler()
                       -> RidgeCV(alphas=np.logspace(-3, 3, 21))
    fit only on the outer-training fold.

    Stacker (meta) = RidgeCV(alphas=np.logspace(-3, 3, 21)) fit on
                     Z[fold != k] vs y[fold != k]; predicts Z[fold == k].

    Residual = RidgeCV(alphas=np.logspace(-3, 3, 21)) fit on the meta's
               nested-CV training residuals; added to the meta's outer
               prediction. Inner CV = KFold(n_splits=5, shuffle=True,
               random_state=<seed>).

    Clip = none (Option A).

The audit's only per-config knob is which feature blocks enter the stack.
Block loaders below mirror the source pipeline's merge-by-subject
behaviour and leave NaNs in place — the per-fold imputer fills them.

Usage:

    from train_clean import run_config
    out = run_config(blocks=['BEH', 'EEG', ...], seed=20260425)
    out['oof']  # (427,) per-subject OOF predictions
    out['per_block_oof']  # dict[block_name, (427,) per-subject base OOF]

CLI form:
    python train_clean.py --config-id 02 --seed 20260425 \\
        --configs ../configs/frontier_last10.yaml \\
        --out ../outputs/oof/oof_pred_02_20260425.csv
"""
from __future__ import annotations

import _audit_common as ac  # path-guard install + constants

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ALPHAS = np.logspace(-3, 3, 21)


# Per-domain prefixes used by MRI_drop_<domain> / EEG_drop_<domain> blocks
# (added for preregister_addendum_2 leave-one-out ablation). Purely
# additive: existing block dispatches are untouched, so existing block
# outputs remain byte-identical.
_MRI_DROP_DOMAINS = {
    "structure":           ("STRUCTURE_NULL_",),
    "resting":             ("RESTING_NULL_",),
    "motor_gforce":        ("MOTOR_GFORCE_",),
    "memory_mst":          ("MEMORY_MST_",),
    "language_speechcomp": ("LANGUAGE_SPEECHCOMP_",),
    "language_wordname":   ("LANGUAGE_WORDNAME_",),
    "task_fmri_all":       ("MOTOR_GFORCE_", "MEMORY_MST_",
                            "LANGUAGE_SPEECHCOMP_", "LANGUAGE_WORDNAME_"),
}
_EEG_DROP_DOMAINS = {
    "resting":          ("RESTING_NULL_",),
    "motor_gofitts":    ("MOTOR_GOFITTS_",),
    "memory_ospan":     ("MEMORY_OSPAN_",),
    "memory_exclusion": ("MEMORY_EXCLUSION_",),
    "motor_bilpress":   ("MOTOR_BILPRESS_",),
    "task_eeg_all":     ("MOTOR_GOFITTS_", "MEMORY_OSPAN_",
                         "MEMORY_EXCLUSION_", "MOTOR_BILPRESS_"),
}


# ---------------------------------------------------------------- block loaders

class BlockSource:
    """Lazy loader for the un-imputed cache + external feature CSVs."""

    def __init__(self):
        self._raw = None
        self._aux: dict[str, np.ndarray] = {}

    @property
    def raw(self) -> dict:
        if self._raw is None:
            npz = np.load(ac.RAW_CACHE_NPZ, allow_pickle=True)
            self._raw = {k: npz[k] for k in npz.files}
            ac.verify_fold_hash(self._raw["subject_ids"], self._raw["fold_ids"])
        return self._raw

    @property
    def subject_ids(self) -> np.ndarray:
        return self.raw["subject_ids"]

    @property
    def y(self) -> np.ndarray:
        return self.raw["y"].astype(np.float64)

    @property
    def fold_ids(self) -> np.ndarray:
        return self.raw["fold_ids"].astype(np.int64)

    # ------------------------------------------------- merge helper

    def _merge_by_subject(self, df: pd.DataFrame, value_cols: list[str]) -> np.ndarray:
        """Left-merge `df` on subject_id; return (n, len(value_cols)) with NaN
        for subjects not in `df`. `df` must have a 'subject' column."""
        sub = pd.DataFrame({"subject": self.subject_ids})
        merged = sub.merge(df[["subject"] + value_cols], on="subject", how="left")
        return merged[value_cols].to_numpy(dtype=np.float32)

    # ------------------------------------------------- block API

    def load_block(self, name: str) -> np.ndarray:
        if name == "BEH":
            X = self.raw["X_beh"]  # (427, 199)
            sex = self.raw["sex"].astype(np.float32).reshape(-1, 1)  # (427, 1)
            return np.concatenate([X, sex], axis=1)

        if name == "EEG":
            return self.raw["X_eeg"]

        if name == "EEG_PSD_RAW":
            df = pd.read_csv(ac.EEG_PSD_CSV)
            cols = [c for c in df.columns if c != "subject"]
            return self._merge_by_subject(df, cols)

        if name == "MRI":
            return self.raw["X_mri"]

        if name == "MRI_drop_fa_cortical":
            mask = self.raw["fa_cortical_mask"].astype(bool)
            return self.raw["X_mri"][:, ~mask]

        if name == "DL_SFCN":
            df = pd.read_csv(ac.SFCN_CSV)
            cols = [c for c in df.columns if c != "subject"]
            return self._merge_by_subject(df, cols)

        if name == "DL_PYMENT":
            df = pd.read_csv(ac.PYMENT_CSV)
            cols = [c for c in df.columns if c != "subject"]
            return self._merge_by_subject(df, cols)

        if name == "DL_DTI_OOF_AGE":
            df = pd.read_csv(ac.DTI_ROI_OOF_CSV).rename(columns={"subject_id": "subject"})
            # Use only predicted_age. The 'bag' column would leak (= pred − age).
            return self._merge_by_subject(df, ["predicted_age"])

        if name == "DL_DTI_VOXEL_RIDGE":
            df = pd.read_csv(ac.DTI_VOXEL_RIDGE_CSV)
            cols = ["dti_fa_voxel_ridge", "dti_md_voxel_ridge", "dti_famd_voxel_ridge"]
            return self._merge_by_subject(df, cols)

        if name == "DL_RSFMRI_FC":
            df = pd.read_csv(ac.RSFMRI_FC_CSV)
            return self._merge_by_subject(df, ["rsfmri_fc_ridge"])

        if name == "DTI_ROI":
            df = pd.read_csv(ac.DTI_ROI_CSV).rename(columns={"subject_id": "subject"})
            cols = [c for c in df.columns if c not in ("subject", "age", "sex")]
            return self._merge_by_subject(df, cols)

        # addendum_2 LOO blocks: drop one acquisition-design domain from MRI/EEG.
        if name.startswith("MRI_drop_") and name[len("MRI_drop_"):] in _MRI_DROP_DOMAINS:
            prefixes = _MRI_DROP_DOMAINS[name[len("MRI_drop_"):]]
            feat = self.raw["feature_names_mri"]
            mask = np.array([str(n).startswith(prefixes) for n in feat])
            return self.raw["X_mri"][:, ~mask]

        if name.startswith("EEG_drop_") and name[len("EEG_drop_"):] in _EEG_DROP_DOMAINS:
            prefixes = _EEG_DROP_DOMAINS[name[len("EEG_drop_"):]]
            feat = self.raw["feature_names_eeg"]
            mask = np.array([str(n).startswith(prefixes) for n in feat])
            return self.raw["X_eeg"][:, ~mask]

        raise KeyError(f"unknown block name: {name}")


# ---------------------------------------------------------------- core pipeline

def _make_block_pipeline() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])


def _block_oof(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
               n_splits: int = ac.N_SPLITS) -> np.ndarray:
    """Outer-fold-aligned OOF predictions from a single block."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for k in range(n_splits):
        va = fold_ids == k
        tr = ~va
        pipe = _make_block_pipeline()
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict(X[va])
    if np.isnan(oof).any():
        raise RuntimeError("block OOF contains NaN — fold ids missing some k")
    return oof


def _meta_oof(Z: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
              seed: int, n_splits: int = ac.N_SPLITS) -> np.ndarray:
    """meta + residual stacker, outer-fold-aligned."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for k in range(n_splits):
        va = fold_ids == k
        tr = ~va
        Z_tr, Z_va = Z[tr], Z[va]
        y_tr = y[tr]

        meta = RidgeCV(alphas=ALPHAS).fit(Z_tr, y_tr)

        # Inner KFold CV on the outer-training fold to compute residual targets.
        inner = KFold(n_splits=5, shuffle=True, random_state=seed)
        inner_oof = np.full(len(y_tr), np.nan, dtype=np.float64)
        for itr, iva in inner.split(Z_tr):
            inner_oof[iva] = RidgeCV(alphas=ALPHAS).fit(Z_tr[itr], y_tr[itr]).predict(Z_tr[iva])
        residual_train = y_tr - inner_oof
        res_model = RidgeCV(alphas=ALPHAS).fit(Z_tr, residual_train)

        oof[va] = meta.predict(Z_va) + res_model.predict(Z_va)
    if np.isnan(oof).any():
        raise RuntimeError("meta OOF contains NaN")
    return oof


# ---------------------------------------------------------------- public API

def run_config(blocks: list[str], seed: int,
               source: BlockSource | None = None,
               verbose: bool = True) -> dict:
    if source is None:
        source = BlockSource()

    y = source.y
    fold_ids = source.fold_ids

    per_block_oof: dict[str, np.ndarray] = {}
    Z_cols = []
    for name in blocks:
        X = source.load_block(name)
        if verbose:
            n_nan = int(np.isnan(X).sum())
            print(f"  block {name:<24}  shape={X.shape}  nan_cells={n_nan}")
        oof = _block_oof(X, y, fold_ids)
        per_block_oof[name] = oof
        Z_cols.append(oof.reshape(-1, 1))

    Z = np.concatenate(Z_cols, axis=1)
    if verbose:
        print(f"  stack matrix Z shape: {Z.shape}")
    final_oof = _meta_oof(Z, y, fold_ids, seed=seed)

    err = final_oof - y
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pearson": float(np.corrcoef(y, final_oof)[0, 1]),
    }
    fold_mae = []
    for k in range(ac.N_SPLITS):
        va = fold_ids == k
        fold_mae.append(float(np.mean(np.abs(final_oof[va] - y[va]))))
    metrics["fold_mae"] = fold_mae
    metrics["fold_mae_mean"] = float(np.mean(fold_mae))
    metrics["fold_mae_std"] = float(np.std(fold_mae))

    if verbose:
        print(f"  pooled MAE={metrics['mae']:.6f}  "
              f"per-fold mean={metrics['fold_mae_mean']:.6f}±{metrics['fold_mae_std']:.6f}  "
              f"r={metrics['pearson']:.4f}  R²={metrics['r2']:.4f}")

    return {
        "blocks": blocks,
        "seed": int(seed),
        "subject_ids": source.subject_ids,
        "y": y,
        "fold_ids": fold_ids,
        "oof": final_oof,
        "per_block_oof": per_block_oof,
        "metrics": metrics,
    }


def write_oof_csv(out_path: Path, result: dict, config_id: str, name: str) -> None:
    df = pd.DataFrame({
        "subject_id": result["subject_ids"],
        "fold": result["fold_ids"],
        "true_age": result["y"],
        "pred": result["oof"],
    })
    df["config_id"] = config_id
    df["config_name"] = name
    df["seed"] = result["seed"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------- CLI

def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "PyYAML not available; install pyyaml or call run_config() directly."
        ) from e
    return yaml.safe_load(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", default=str(ac.AUDIT_ROOT / "configs" / "frontier_last10.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfgs = _load_yaml(Path(args.configs))["configs"]
    cfg = next(c for c in cfgs if c["id"] == args.config_id)

    print(f"running config {cfg['id']} ({cfg['name']}) seed={args.seed}")
    print(f"  blocks: {cfg['blocks']}")
    src = BlockSource()
    res = run_config(cfg["blocks"], seed=args.seed, source=src)
    write_oof_csv(Path(args.out), res, cfg["id"], cfg["name"])
    print(f"wrote {args.out}")
    print(json.dumps({k: v for k, v in res["metrics"].items() if k != "fold_mae"}, indent=2))


if __name__ == "__main__":
    main()
