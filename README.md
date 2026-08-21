# Kaggle Playground Series S6E8 - Predicting Smartphone Addiction

A gradient-boosting pipeline that predicts smartphone addiction risk from
self-reported usage data, built for Kaggle Playground Series Season 6,
Episode 8.

---

## Overview

The task is binary classification: given a person's daily screen time,
app usage, sleep, and a few categorical lifestyle fields, predict the
probability that they're flagged as addicted (`addicted_label`), scored
by ROC AUC. The dataset is synthetic (Kaggle-generated, not scraped real
users), which changes what "understanding the data" means: instead of
asking what's true about smartphone use in general, the project spends
real effort asking what's true about *this generator specifically* -
where the signal actually sits, whether it's exploitable, and whether
exploiting it would even be legitimate.

Current state: five builds in, best validated model at CV mean ROC AUC
0.96445 (public leaderboard 0.96608), with a full audit of the generator's
structure completed and no further exploitable signal found beyond what's
already in the model.

## Why I built this

I wanted a project that forced discipline around validation, not just
model fitting - a competition with a public leaderboard is a good way to
find out fast whether your cross-validation actually agrees with reality.
It's also a deliberate exercise in staged, auditable development: every
build has a stated objective, a frozen control to compare against, and a
paper trail of what was tried and rejected, not just what worked.

## Key features

- A validated 5-fold cross-validation harness that CatBoost, LightGBM,
  and XGBoost all run through identically, so model comparisons aren't
  contaminated by different preprocessing
- An engineered feature (`screen_residual`) derived from a compositional
  relationship found in the data during EDA, validated against two
  separate model families before being accepted
- A forensic investigation notebook that audits the synthetic generator
  itself - quantization, arithmetic constraints, missingness patterns,
  ID drift, near-duplicate structure - and documents what was found,
  rejected, and why
- An experiment tracker (`experiments/experiments.csv`) recording every
  model run with its exact configuration, fold scores, and conclusion, so
  no result exists only as a remembered number

## Tech stack

- Python
- pandas, NumPy
- scikit-learn (preprocessing, cross-validation)
- XGBoost, CatBoost, LightGBM
- SciPy (statistical tests)
- Jupyter (experimentation), pytest (testing)
- Git / GitHub, issue-branch-PR workflow

## Project workflow

1. Data audit - schema, missingness, duplicates, distribution shift,
   leakage check
2. Validation harness - stratified 5-fold CV, shared across every model
3. Model benchmarking - CatBoost, LightGBM, XGBoost on identical raw
   features
4. Feature engineering - hypotheses from the audit, tested against frozen
   controls, accepted or rejected on CV evidence
5. Synthetic-generator investigation - forensic audit of what's real
   generator structure versus ordinary signal versus noise
6. Hyperparameter tuning, ensembling, and final submission strategy -
   not started yet (see Future improvements)

## Data source

Competition data from Kaggle Playground Series S6E8 - 691,369 training
rows, 296,302 test rows, all fields synthetically generated. It's not
included in this repository: `data/*.csv` is gitignored per the
competition's terms, and `data/README.md` explains how to pull it down
yourself after accepting the competition rules on Kaggle.

Every predictor column has some missingness (4-19% depending on the
column), and the target is moderately imbalanced (71% positive). Both are
handled explicitly in the preprocessing rather than assumed away.

---

## Results / outcomes

| Experiment | Model | Feature set | CV mean ROC AUC | Public LB |
|---|---|---|---|---|
| E001 | Logistic Regression | raw predictors | 0.91149 (std 0.00081) | 0.91358 |
| E002 | CatBoost | raw predictors | 0.96040 (std 0.00051) | 0.96151 |
| E004 | XGBoost | raw predictors | 0.96382 (std 0.00056) | 0.96539 |
| **E006** | **XGBoost** | **raw + `screen_residual`** | **0.96445 (std 0.00056)** | **0.96608** |

E006 is the current best model and the frozen control for everything
after it. `screen_residual` - the gap between `daily_screen_time_hours`
and the sum of its three named components - was found during EDA,
tested in isolation against both XGBoost and CatBoost, and only accepted
once it showed a consistent gain on both (+0.00062 and +0.00064
respectively, 5/5 folds improved each time). A second candidate feature
built from the same relationship (`component_sum`) was tested and
rejected as redundant once `screen_residual` was already in the model -
that negative result is recorded too, not just the win.

Every number above traces to a committed artifact
(`experiments/experiments.csv`, `outputs/model_benchmarks.csv`,
`outputs/feature_experiments.csv`), not a remembered figure.

**Synthetic-generator investigation (Build 5):** ran a full forensic audit
of the dataset's generation process - numeric precision grids, the
screen-time compositional constraint behind `screen_residual`, exact-value
target rates, repeated and near-duplicate profiles, missingness patterns,
ID drift, and cross-feature constraints. Found one previously-uncharted
generator artifact (`sleep_hours + daily_screen_time_hours` clips near
20h) and confirmed it's fully redundant with a feature already in the
model. Every other candidate - missingness encoding, near-duplicate
lookup, frequency encoding - was tested and rejected on evidence. No new
feature came out of it, and that's recorded as a real finding
(`outputs/synthetic_generator_findings.csv`), not a gap in the work.
