"""Shared utilities for the brain-age audit.

Two responsibilities:

1. **Path guard.** Refuse to open the forbidden file
   (`codes/brain_age_2026/tuning/results.tsv` of the main repo). Monkey-
   patches `builtins.open` and `os.open` so that any read attempt — direct,
   via pandas, via numpy — raises `AuditPathGuardError`.

2. **Fold-integrity check.** Recompute the SHA-256 of
   `(subject_ids, fold_ids)` in cache row order and compare against
   `preregister_folds.sha256`.

Import this module FIRST in every audit script.
"""
from __future__ import annotations

import builtins
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----- canonical paths ------------------------------------------------------

AUDIT_ROOT = Path("/media/DATA3/Quanta/audit").resolve()
MAIN_ROOT = Path("/media/DATA3/Quanta").resolve()

CACHE_NPZ = MAIN_ROOT / "codes" / "brain_age_2026" / "tuning" / "cache" / "brain_age_cache.npz"
CLEANED_CSV = MAIN_ROOT / "Data" / "DATA_ses-01_cleaned.csv"
RAW_CACHE_NPZ = AUDIT_ROOT / "Data" / "brain_age_cache_raw.npz"
FOLD_HASH_FILE = AUDIT_ROOT / "preregister_folds.sha256"

DL_FEATURES_DIR = MAIN_ROOT / "Data" / "DL_features"
SFCN_CSV = DL_FEATURES_DIR / "sfcn_features.csv"
PYMENT_CSV = DL_FEATURES_DIR / "pyment_features.csv"
EEG_PSD_CSV = DL_FEATURES_DIR / "eeg_psd_features.csv"
DTI_VOXEL_RIDGE_CSV = DL_FEATURES_DIR / "dti_voxel_ridge_oof.csv"
RSFMRI_FC_CSV = DL_FEATURES_DIR / "rsfmri_fc_oof.csv"
DTI_ROI_CSV = Path("/media/DATA3/GripForce/analysis/brainage/features_roi.csv")
DTI_ROI_OOF_CSV = Path("/media/DATA3/GripForce/analysis/brainage/cv_predictions.csv")

# Anything whose resolved path equals this is FORBIDDEN.
FORBIDDEN_PATHS = {
    (MAIN_ROOT / "codes" / "brain_age_2026" / "tuning" / "results.tsv").resolve(),
}


class AuditPathGuardError(RuntimeError):
    """Raised when an audit script tries to open a forbidden path."""


# ----- path-guard install --------------------------------------------------

_real_open = builtins.open
_real_os_open = os.open
_installed = False


def _resolve_pathlike(p):
    try:
        return Path(os.fspath(p)).resolve()
    except (TypeError, ValueError, OSError):
        return None


def _check_path(p):
    resolved = _resolve_pathlike(p)
    if resolved is not None and resolved in FORBIDDEN_PATHS:
        raise AuditPathGuardError(
            f"audit path guard: refusing to open forbidden file {resolved} "
            "(see preregister.md §9)"
        )


def _guarded_open(file, *args, **kwargs):
    _check_path(file)
    return _real_open(file, *args, **kwargs)


def _guarded_os_open(path, *args, **kwargs):
    _check_path(path)
    return _real_os_open(path, *args, **kwargs)


def install_path_guard():
    """Install the path guard. Idempotent. Call this before any I/O."""
    global _installed
    if _installed:
        return
    builtins.open = _guarded_open  # type: ignore[assignment]
    os.open = _guarded_os_open     # type: ignore[assignment]
    _installed = True


# Self-install on import — the guard is meant to be unconditional.
install_path_guard()


# ----- fold integrity ------------------------------------------------------

def recover_subject_ids() -> np.ndarray:
    """Recover the 427 subject IDs in cache row order.

    Cache row order = `df.dropna(subset=['BASIC_INFO_AGE']).reset_index(drop=True)`
    where `df = pd.read_csv(DATA_ses-01_cleaned.csv)`.
    """
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df = df.dropna(subset=["BASIC_INFO_AGE"]).reset_index(drop=True)
    return df["BASIC_INFO_ID"].astype(str).to_numpy()


def load_fold_ids() -> np.ndarray:
    """Load the canonical fold_ids from the main repo's cache (read-only)."""
    npz = np.load(CACHE_NPZ)
    return npz["fold_ids"].astype(np.int64)


def compute_fold_hash(subject_ids: np.ndarray, fold_ids: np.ndarray) -> str:
    """Canonical recipe documented in preregister_folds.sha256."""
    if len(subject_ids) != len(fold_ids):
        raise ValueError("subject_ids and fold_ids must have the same length")
    buf = "\n".join(f"{s}\t{int(f)}" for s, f in zip(subject_ids, fold_ids)) + "\n"
    return hashlib.sha256(buf.encode("utf-8")).hexdigest()


def read_expected_fold_hash() -> str:
    """The single non-comment line of preregister_folds.sha256."""
    text = FOLD_HASH_FILE.read_text().splitlines()
    for line in text:
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise RuntimeError(f"no hash line found in {FOLD_HASH_FILE}")


def verify_fold_hash(subject_ids: np.ndarray, fold_ids: np.ndarray) -> str:
    """Recompute the hash, compare to the frozen one, abort on mismatch."""
    got = compute_fold_hash(subject_ids, fold_ids)
    expected = read_expected_fold_hash()
    if got != expected:
        raise RuntimeError(
            "audit fold-integrity check FAILED:\n"
            f"  expected: {expected}\n"
            f"  got     : {got}\n"
            "If the source data or cache changed, the audit must be re-frozen."
        )
    return got


# ----- audit constants -----------------------------------------------------

PRIMARY_SEED = 20260425
SENSITIVITY_SEED = 20260426
N_SPLITS = 5
N_BOOTSTRAP = 10_000
ACCEPT_DELTA = 0.02  # years (preregister §6 magnitude floor)
