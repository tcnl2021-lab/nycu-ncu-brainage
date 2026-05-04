"""Build the un-imputed cache used by the audit.

Mirrors `codes/brain_age_2026/01_load_filter.py` + `02_preprocess.py`
**except** that:

  - the 172 `STRUCTURE_NULL_MRI_*_FA` cortical-projection columns are
    **kept** (the audit needs them for configs 01–08; config 09 drops
    them via a within-block mask), and
  - the median-impute step is **omitted** (NaNs are preserved). This is
    the leakage-fix demanded by codex critique #1.

Outputs `Data/brain_age_cache_raw.npz` with arrays:

    y              : (427,) float32 chronological age
    fold_ids       : (427,) int64 — copied verbatim from
                     codes/brain_age_2026/tuning/cache/brain_age_cache.npz
    subject_ids    : (427,) <U8 — recovered from cleaned CSV
    sex            : (427,) float32 — for BEH-block augmentation
    X_beh          : (427, 199) float32 with NaNs
    X_eeg          : (427, 509) float32 with NaNs
    X_mri          : (427, 1266) float32 with NaNs
    missing_mask_beh / _eeg / _mri : bool arrays of the same shape
    fa_cortical_mask : (1266,) bool — True for FA-cortical MRI cols
    feature_names_beh / _eeg / _mri : object arrays of column names

Refuses to run unless the fold-hash matches `preregister_folds.sha256`.
"""
from __future__ import annotations

import _audit_common as ac  # installs path guard, exposes constants

import re
import numpy as np
import pandas as pd


def modality_of(col: str) -> str | None:
    if re.search(r"_BEH_", col):
        return "BEH"
    if re.search(r"_EEG_", col):
        return "EEG"
    if re.search(r"_MRI_", col):
        return "MRI"
    return None


def is_pad_validation_feature(col: str) -> bool:
    """Standing rule: any column with `_ST_` or `_Q_` is a PAD validation
    measure and must NEVER be used as a model input."""
    return ("_ST_" in col) or ("_Q_" in col)


def main() -> None:
    print(f"reading {ac.CLEANED_CSV} ...")
    df = pd.read_csv(ac.CLEANED_CSV, low_memory=False)
    df = df.dropna(subset=["BASIC_INFO_AGE"]).reset_index(drop=True)
    print(f"  rows after dropna(BASIC_INFO_AGE): {len(df)}")

    subject_ids = df["BASIC_INFO_ID"].astype(str).to_numpy()
    y = df["BASIC_INFO_AGE"].to_numpy(dtype=np.float32)
    sex = pd.to_numeric(df["BASIC_INFO_SEX"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    feat_cols = [c for c in df.columns
                 if modality_of(c) is not None
                 and not is_pad_validation_feature(c)]
    print(f"  tokenized BEH/EEG/MRI cols (after PAD-validation exclusion): {len(feat_cols)}")

    # Coerce numeric, then drop features with > 30 % missing on the FULL data
    # (matches the column set kept by 02_preprocess.py prior to its imputation).
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    miss = df[feat_cols].isna().mean()
    keep = miss[miss <= 0.30].index.tolist()
    print(f"  kept after >30%-missing filter: {len(keep)} (dropped {len(feat_cols)-len(keep)})")

    by_mod: dict[str, list[str]] = {"BEH": [], "EEG": [], "MRI": []}
    for c in keep:
        by_mod[modality_of(c)].append(c)
    counts = {k: len(v) for k, v in by_mod.items()}
    print(f"  per-modality counts: {counts}")
    expected = {"BEH": 199, "EEG": 509, "MRI": 1266}
    if counts != expected:
        raise RuntimeError(
            f"per-modality counts {counts} do not match expected {expected}; "
            "source CSV may have changed since the audit was specified."
        )

    fa_cortical_mask = np.array(
        [c.startswith("STRUCTURE_NULL_MRI_") and c.endswith("_FA") for c in by_mod["MRI"]],
        dtype=bool,
    )
    print(f"  FA-cortical cols (STRUCTURE_NULL_MRI_*_FA) within MRI block: {int(fa_cortical_mask.sum())}")

    X_beh = df[by_mod["BEH"]].to_numpy(dtype=np.float32)
    X_eeg = df[by_mod["EEG"]].to_numpy(dtype=np.float32)
    X_mri = df[by_mod["MRI"]].to_numpy(dtype=np.float32)

    print("  missing-cell counts (un-imputed): "
          f"BEH={int(np.isnan(X_beh).sum())}, "
          f"EEG={int(np.isnan(X_eeg).sum())}, "
          f"MRI={int(np.isnan(X_mri).sum())}")

    # Fold IDs: copy from the canonical cache. We do NOT regenerate.
    fold_ids = ac.load_fold_ids()
    if len(fold_ids) != len(y):
        raise RuntimeError(f"fold_ids length {len(fold_ids)} != n_subjects {len(y)}")

    # Sanity: ages must match the cache's y
    npz = np.load(ac.CACHE_NPZ)
    if not np.allclose(y, npz["y"], atol=1e-4):
        raise RuntimeError("y from cleaned CSV does not match the cache's y")
    print("  y matches existing cache: OK")

    # Hash check — fail loudly if the source data shifted
    h = ac.verify_fold_hash(subject_ids, fold_ids)
    print(f"  fold hash verified: {h}")

    out_path = ac.RAW_CACHE_NPZ
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        y=y,
        fold_ids=fold_ids,
        subject_ids=subject_ids.astype("U16"),
        sex=sex,
        X_beh=X_beh,
        X_eeg=X_eeg,
        X_mri=X_mri,
        missing_mask_beh=np.isnan(X_beh),
        missing_mask_eeg=np.isnan(X_eeg),
        missing_mask_mri=np.isnan(X_mri),
        fa_cortical_mask=fa_cortical_mask,
        feature_names_beh=np.array(by_mod["BEH"], dtype=object),
        feature_names_eeg=np.array(by_mod["EEG"], dtype=object),
        feature_names_mri=np.array(by_mod["MRI"], dtype=object),
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
