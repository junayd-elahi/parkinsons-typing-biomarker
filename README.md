# Typing-Based Biomarker Classification for Parkinson's Disease

A reproducible benchmark testing whether keystroke timing dynamics can separate
Parkinson's Disease (PD) patients from controls under leakage-resistant
evaluation, using the Tappy keystroke dataset.

**Headline finding:** under subject-wise cross-validation that prevents data
leakage, neither Random Forest nor XGBoost reliably separates PD from control
above a majority-class baseline. This contributes a reproducible benchmark
showing that previously reported results on this dataset do not generalise once
the same participant is prevented from appearing in both training and test
folds.

## Why this matters

Many keystroke-based PD studies report high accuracy by splitting data at the
keystroke or window level, which lets samples from the same person land in both
train and test. The model then partly learns to recognise individuals rather
than the disease, inflating performance. This project evaluates the same
problem with **subject-wise GroupKFold**, so every participant appears in only
one fold, and benchmarks against a majority-class baseline. The honest negative
result is the contribution: it quantifies how much of the prior signal was
leakage.

## Methodology

- **Subject-wise evaluation:** nested `GroupKFold` (5 outer folds, 3 inner),
  grouped by `user_id`, so hyperparameter tuning and testing never share a
  participant.
- **Models:** Random Forest and XGBoost, each tuned with `GridSearchCV` on the
  inner folds, optimising F1.
- **Baseline:** majority-class classifier for an honest reference point.
- **51 features** per user spanning hold time, latency and flight time:
  central-tendency, variability (std, skew, kurtosis, IQR, CV, MAD, burstiness),
  hand asymmetry, and directional (LL/LR/RL/RR) statistics.
- **Robustness checks:** a feature-group ablation, a feature-stability analysis
  (which features are consistently selected across folds), and a SMOTE
  experiment to test whether class imbalance explains the result.

## Repository structure

```
.
├── src/
│   ├── 01_clean.py        # parse raw logs + metadata, clean, merge to one CSV
│   ├── 02_features.py     # per-user feature engineering (51 features)
│   ├── 03_train.py        # primary nested GroupKFold benchmark + baseline
│   ├── 04_ablation.py     # feature-group ablation (central / variability / combined)
│   ├── 05_stability.py    # per-fold stable-feature selection + retraining
│   ├── 06_smote.py        # SMOTE oversampling check
│   └── 07_plots.py        # figures from the generated result workbooks
├── data/                  # dataset goes here (not committed - see data/README.md)
├── requirements.txt
├── LICENSE
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
```

Download the dataset and place it under `data/` as described in
[`data/README.md`](data/README.md), then run from the repository root in order:

```bash
python src/01_clean.py        # -> Tappy_Data_Cleaned.csv
python src/02_features.py     # -> User_Features.csv
python src/03_train.py        # -> Model_training.xlsx
python src/04_ablation.py     # -> ablation_summary.xlsx
python src/05_stability.py    # -> Feature_stability.xlsx
python src/06_smote.py        # -> smote_summary.xlsx
python src/07_plots.py        # -> figures (.png)
```

Each script reads the artefacts produced by the earlier ones, so order matters.

## Pipeline

1. **Cleaning** (`01_clean.py`): parses per-user keystroke logs and metadata,
   coerces timing columns to numeric, drops missing rows, filters physiologically
   implausible values, and merges keystrokes with PD labels into one CSV.
2. **Feature engineering** (`02_features.py`): keeps users with at least 2000
   keystrokes and computes the 51-feature per-user matrix.
3. **Primary benchmark** (`03_train.py`): nested GroupKFold for RF and XGBoost
   against a majority baseline, with confusion matrices and averaged feature
   importances.
4. **Ablation** (`04_ablation.py`): compares central-tendency vs variability vs
   combined feature sets.
5. **Stability** (`05_stability.py`): selects features that are repeatedly
   important across inner folds and retrains on those subsets.
6. **SMOTE** (`06_smote.py`): tests whether oversampling the minority class
   changes the conclusion.
7. **Figures** (`07_plots.py`): reads the result workbooks and produces the
   plots.


