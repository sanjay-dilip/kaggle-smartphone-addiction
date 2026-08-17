# Kaggle Playground Series S6E8 - Predicting Smartphone Addiction

Binary classification of smartphone addiction risk, built for Kaggle
Playground Series Season 6, Episode 8.

## Objective

Predict `addicted_label` (probability of smartphone addiction) for each
row in the competition test set.

## Evaluation metric

ROC AUC.

## Current status

**Build 1 - Data Audit and Synthetic-Data EDA complete.** Repository
foundation (Build 0) and a full dataset audit (Build 1 — schema,
missingness, duplicates, train/test shift, univariate target
relationships, and a leakage assessment) are in place; see
`notebooks/01_eda.ipynb`. No feature engineering, modeling, or submissions
have been done yet.

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
submissions/    Kaggle submission files (see submissions/README.md)
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
