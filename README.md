# NYCU-NCU Brain-Age Prediction: Analysis Scripts and Outputs

[![DOI](https://zenodo.org/badge/[ZENODO_ID].svg)](https://zenodo.org/badge/latestdoi/[ZENODO_ID])

Analysis pipeline and pre-computed outputs for the NYCU-NCU brain-age prediction study.

## Overview

This repository contains the **clean-protocol analysis stack** and all pre-computed model outputs reported in the manuscript:

> Chang, E., et al. (2025). Brain-age prediction from multimodal neuroimaging in a healthy adult cohort. *[Journal, Volume, Pages].* https://doi.org/[MANUSCRIPT_DOI]

**Cohort:** N = 427 healthy adults; age 19–80 years; single session at Taiwan Brain and Mind Imaging Center (NCCU)

**Modalities:** Structural/functional MRI, DTI, resting and task EEG, cognitive/motor testing, questionnaires

**Model:** 7-block ridge-regression stack: BEH+SEX → EEG+PSD → sMRI → DTI ROI → DL embeddings → rs-fMRI FC ridge → DTI voxel ridge → meta-learner

---

## Quick Start

### 1. Get the Feature Data

Download from **OSF** (https://doi.org/[OSF_DOI]):

**Restricted access** (approved via data access form):
- `DATA_ses-01_cleaned.csv` — All 427 × 2,121 features (demographics, behavioral, EEG, sMRI, task-fMRI)

**Public**:
- `sfcn_features.csv`, `pyment_features.csv`, `eeg_psd_features.csv`, `rsfmri_fc_oof.csv`, `dti_voxel_ridge_oof.csv`, `dti_roi_features.csv`, `dti_roi_oof.csv`
- `preregister_folds.sha256` — Fold integrity hash

### 2. Set Up Environment

```bash
git clone https://github.com/tcnl2021-lab/nycu-ncu-brainage.git
cd nycu-ncu-brainage

conda env create -f environment.yml
conda activate nycu-ncu-brainage
```

### 3. Reproduce All Results

**Run the clean-protocol stack on all configs:**
```bash
python scripts/audit_replay.py --config configs/frontier_addendum_1.yaml --seed 20260426
```

**Compute bootstrap statistics:**
```bash
python scripts/audit_stats.py --config configs/frontier_addendum_1.yaml --seed 20260426
```

**Generate manuscript figures:**
```bash
python scripts/make_clean_protocol_figures.py
```

Pre-computed outputs are already in `outputs/` for immediate inspection.

---

## Repository Structure

```
nycu-ncu-brainage/
├── README.md                                    # This file
├── LICENSE                                      # MIT
├── environment.yml                              # Conda environment (pinned versions)
├── preregister_folds.sha256                     # Fold integrity hash
├── scripts/
│   ├── _audit_common.py                        # Shared constants, path guard, fold check
│   ├── cache_raw.py                            # Build NPZ cache from cleaned CSV
│   ├── train_clean.py                          # Clean-protocol model (one config)
│   ├── audit_replay.py                         # Run all configs from YAML
│   ├── audit_stats.py                          # Paired bootstrap + fold t-tests
│   ├── audit_aggregate.py                      # Cumulative window test
│   └── make_clean_protocol_figures.py           # Generate manuscript figures
├── configs/
│   ├── frontier_addendum_1.yaml                # RQ1: primary + post-hoc additions
│   └── frontier_addendum_2.yaml                # RQ2: leave-one-domain-out ablation
└── outputs/
    ├── oof/                                    # Out-of-fold predictions (frontier configs 01–11)
    ├── addendum_1/
    │   ├── oof/                                # OOF for configs 01b–05b (RQ1)
    │   └── stats_*.csv                         # Bootstrap stats for addendum_1
    └── addendum_2/
        ├── oof/                                # OOF for configs 00c–13c (RQ2 ablation)
        └── stats_*.csv                         # Bootstrap stats for addendum_2
```

---

## Key Scripts

### `cache_raw.py`
Builds the un-imputed feature cache (`audit/Data/brain_age_cache_raw.npz`) from the cleaned CSV and original fold structure.

```bash
python scripts/cache_raw.py
```

**Input:** `DATA_ses-01_cleaned.csv` (OSF restricted)
**Output:** `brain_age_cache_raw.npz` (427 subjects × features, no imputation)

### `train_clean.py`
Runs the clean-protocol model for one config:
- Per-block: `SimpleImputer → StandardScaler → RidgeCV`
- Out-of-fold (5 folds): produces per-subject age predictions
- No clipping, no composite features, no bespoke engineering

```bash
python scripts/train_clean.py --config_id 05b --seed 20260426
```

**Output:** `oof_pred_05b_20260426.csv` (427 rows, subject + OOF age + fold)

### `audit_replay.py`
Loops over all configs in a YAML list and runs `train_clean.py` for each.

```bash
python scripts/audit_replay.py --config configs/frontier_addendum_1.yaml --seed 20260426
```

**Output:** `outputs/addendum_1/oof/oof_pred_{01b..05b}_{seed}.csv`

### `audit_stats.py`
For each adjacent config pair (01b→02b, 02b→03b, etc.):
- Paired bootstrap (10,000 resamples) of MAE difference
- Per-fold t-test
- Cumulative effect (01b→05b)

```bash
python scripts/audit_stats.py --config configs/frontier_addendum_1.yaml --seed 20260426
```

**Output:** `outputs/addendum_1/stats_{seed}.csv`

### `make_clean_protocol_figures.py`
Generates all manuscript figures:
- `frontier_mae.png` — MAE progression across configs
- `pad_vs_age.png` — Predicted age deviation vs. actual age (scatter)
- `fold_reliability.png` — MAE per fold (error bars)

```bash
python scripts/make_clean_protocol_figures.py
```

---

## Data Codebook

### Feature Domains

| Domain | Columns | Notes |
|---|---|---|
| Demographics | 29 | Age, sex, education, occupation, medical history, lifestyle |
| Questionnaires | 30 | PSQI, SF-36, MOCA, Beck anxiety/depression, IPAQ, MSPSS, EHI |
| Behavioral | 207 | Cognitive/motor task metrics (accuracy, RT, variance) |
| EEG | 509 | Resting microstate stats (81) + task-EEG (428) |
| MRI | 1,269 | FreeSurfer sMRI (684), task-fMRI betas (208), rs-fMRI metrics (374) |
| **Total** | **2,121** | |

### Feature Cleaning

See `data_dictionary.csv` (OSF deposit) for column-level documentation.

Cleaning log: 3,354 cells modified (0.32%):
- 1 age/sex swap → corrected
- 0–3 → 7 questionnaire values → NaN
- 5 IPAQ extreme → winsorized at p99
- 629 RTvar = 0 → NaN (impossible variances)
- 1,367 values > 5 SD → winsorized (BEH/EEG/MRI/ST-Q)

---

## Model Architecture

### Block Structure

Each feature block is independently modeled via ridge regression:

1. **BEH+SEX block** (200 features)
   - RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
   - OOF prediction: `predicted_age_beh`

2. **EEG+PSD block** (659 features: EEG + post-hoc PSD)
   - RidgeCV(alphas=[...])
   - OOF prediction: `predicted_age_eeg`

3. **sMRI block** (1,094 features: sMRI only, FA-on-cortex dropped)
   - RidgeCV(alphas=[...])
   - OOF prediction: `predicted_age_mri`

4. **DTI ROI block** (90 features: 45 FA + 45 MD)
   - RidgeCV(alphas=[...])
   - OOF prediction: `predicted_age_dti_roi`

5. **DL embeddings block** (71 features: 65 SFCN + 6 Pyment)
   - RidgeCV(alphas=[...])
   - OOF prediction: `predicted_age_dl`

6. **rs-fMRI FC ridge block** (1 OOF scalar from Schaefer-400)
   - Frozen pre-computed via `train_fc_ridge.py`
   - OOF prediction: `predicted_age_rsfmri_fc`

7. **DTI voxel ridge block** (3 OOF scalars: FA, MD, FA+MD)
   - Frozen pre-computed via `train_voxel_ridge.py`
   - OOF predictions: `predicted_age_dti_{fa,md,famd}_voxel`

### Meta-Learner (Stack)

**Input matrix Z:** 427 subjects × 7 columns (one OOF per block, outer-fold-aligned)

**Final stack:** RidgeCV on Z → final age prediction

**Evaluation:** MAE (mean absolute error) in years

---

## Pre-Registration

Fold structure pre-registered before analysis:

```bash
# Verify fold integrity
sha256sum preregister_folds.sha256
# Expected: db1510dd5442d4492955579dee406c6dd7be8feabea80d929672b2dc0f5b5a6d
```

---

## Configuration Files

### `frontier_addendum_1.yaml`

Configs for **RQ1: Feature importance ranking**

```yaml
configs:
  - id: "01b"
    name: "Cohort-design baseline (config 05b from original audit)"
    blocks: ["BEH_SEX", "EEG_PSD", "MRI", "DTI_ROI", "DL", "RSFMRI_FC", "DTI_VOXEL"]
  
  - id: "02b"
    name: "Remove DTI_VOXEL"
    blocks: ["BEH_SEX", "EEG_PSD", "MRI", "DTI_ROI", "DL", "RSFMRI_FC"]
  
  # ... 03b, 04b, 05b similarly
```

### `frontier_addendum_2.yaml`

Configs for **RQ2: Leave-one-domain-out ablation**

```yaml
configs:
  - id: "00c"
    name: "Full stack"
    blocks: [all]
  
  - id: "01c"
    name: "No BEH"
    blocks: ["EEG_PSD", "MRI", "DTI_ROI", "DL", "RSFMRI_FC", "DTI_VOXEL"]
  
  # ... 02c (no EEG), 03c (no MRI), etc.
```

---

## Output Files

### OOF Prediction CSVs

**Format:** One CSV per config per seed

```
BASIC_INFO_ID,fold,{config_id}_predicted_age
sub-0001,0,56.32
sub-0002,1,48.91
...
```

**Columns:**
- `BASIC_INFO_ID`: Subject ID (de-identified)
- `fold`: 0–4 (outer CV fold assignment)
- `{config_id}_predicted_age`: Predicted brain age (years)

### Stats CSVs

**Format:** Paired bootstrap statistics

```
config_pair,test,metric,estimate,ci_lower,ci_upper,p_value,n_bootstrap
01b-02b,paired_bootstrap,mae_diff,-0.053,-0.104,0.002,0.031,10000
01b-02b,fold_ttest,mae_diff,-0.048,-0.089,0.004,0.037,5
```

**Interpretation:**
- `mae_diff` < 0 → config 2 is better (lower MAE)
- `p_value` < 0.05 → statistically significant difference

---

## Reproducing the Manuscript

### 1. Download Data from OSF

Create a `data/` directory and place:
```
data/
├── DATA_ses-01_cleaned.csv          (restricted)
├── sfcn_features.csv
├── pyment_features.csv
├── eeg_psd_features.csv
├── rsfmri_fc_oof.csv
├── dti_voxel_ridge_oof.csv
├── dti_roi_features.csv
└── dti_roi_oof.csv
```

Update paths in `scripts/_audit_common.py` if needed.

### 2. Run the Pipeline

```bash
# Build cache
python scripts/cache_raw.py

# Run all configs
python scripts/audit_replay.py --config configs/frontier_addendum_1.yaml --seed 20260426

# Compute stats
python scripts/audit_stats.py --config configs/frontier_addendum_1.yaml --seed 20260426

# Generate figures
python scripts/make_clean_protocol_figures.py
```

### 3. Verify Outputs

```bash
# Check OOF dimensions
wc -l outputs/addendum_1/oof/oof_pred_05b_20260426.csv
# Expected: 428 lines (427 subjects + 1 header)

# Check fold integrity
sha256sum preregister_folds.sha256
# Expected: db1510dd5442d4492955579dee406c6dd7be8feabea80d929672b2dc0f5b5a6d
```

---

## Dependencies

See `environment.yml` for exact pinned versions. Key packages:
- Python 3.11
- scikit-learn 1.4.x (SimpleImputer, RidgeCV, cross_val_predict)
- pandas 2.x
- numpy 1.x
- matplotlib (for figures)
- scipy (for statistics)

---

## Citation

If you use these scripts or outputs, please cite:

```bibtex
@article{chang2025brainage,
  title   = {Brain-age prediction from multimodal neuroimaging in a healthy adult cohort},
  author  = {Chang, E. and [Co-authors]},
  journal = {[Journal]},
  year    = {2025},
  volume  = {[Volume]},
  pages   = {[Pages]},
  doi     = {[DOI]}
}

@software{chang2025brainage_code,
  title   = {NYCU-NCU brain-age prediction: analysis scripts and outputs},
  author  = {Chang, E.},
  year    = {2025},
  url     = {https://github.com/tcnl2021-lab/nycu-ncu-brainage},
  doi     = {10.5281/zenodo.[ZENODO_ID]}
}
```

---

## Related Resources

- **Feature data:** https://doi.org/[OSF_DOI] (OSF project)
- **SFCN weights:** https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
- **Pyment weights:** https://github.com/estenhl/pyment-public
- **Pre-registration:** See `preregister_folds.sha256` and fold hash verification

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Pre-trained model weights (SFCN, Pyment) retain their original licenses.

---

## Contact & Support

**Corresponding Author:** Erik Chang (National Cheng Chi University)

For questions about the analysis, code, or pre-computed outputs, open an issue on GitHub.

For data access requests, visit the OSF project: https://doi.org/[OSF_DOI]

---

**Last updated:** 2026-05-02
