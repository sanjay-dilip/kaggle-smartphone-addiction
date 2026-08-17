# Kaggle Playground Series S6E8 - Predicting Smartphone Addiction

Binary classification of smartphone addiction risk, built for Kaggle
Playground Series Season 6, Episode 8.

## Objective

Predict `addicted_label` (probability of smartphone addiction) for each
row in the competition test set.

## Evaluation metric

ROC AUC.

## Current status

**Build 2 - Validation Harness and Logistic Regression Baseline
complete.** Repository foundation (Build 0), a full dataset audit (Build
1 — see `notebooks/01_eda.ipynb`), and a validated Logistic Regression
baseline (Build 2 — see `notebooks/02_baseline.ipynb`) are in place. E001
(Logistic Regression, raw/imputed features) scores a mean 5-fold ROC AUC
of 0.9115 (std 0.0008); see `experiments/experiments.csv`. No feature
engineering, strong-model benchmarking, hyperparameter tuning, or Kaggle
submissions have been done yet.

## Planned build sequence

0. Competition foundation
1. Data audit and synthetic-data EDA
2. Validation harness and logistic-regression baseline
3. Strong model benchmarks
4. Hypothesis-driven feature engineering
5. Synthetic-generator investigation
6. Controlled hyperparameter tuning
7. Ensembling and blending
8. CV vs leaderboard reconciliation
9. Final submission strategy
10. Consolidation, reproduction, documentation, publication (Build C)

## Repository structure

```text
src/            reusable, accepted logic (imported by notebooks and tests)
notebooks/      experimentation laboratory
tests/          smoke checks and unit tests
data/           competition CSVs (gitignored; see data/README.md)
experiments/    experiment tracker (experiments.csv) and per-run artifacts
deliverables/   submission-ready CSVs (see deliverables/README.md)
docs/           decision log, build history, competition notes
outputs/        generated figures and artifacts
```

## Local setup

```bash
python -m venv <env-name>
<env-name>\Scripts\activate        # Windows
pip install -r requirements-dev.txt
pytest tests/
```

## Data acquisition

Raw competition data is not committed to this repository. See
`data/README.md` for how to obtain and place it after a fresh clone.

## Reproducibility

- All project paths are centralized in `src/config.py` via `pathlib`.
- A shared random seed (`RANDOM_SEED = 42`) is defined for later builds.
- Dependencies are pinned by major version in `requirements.txt` and
  `requirements-dev.txt`.
