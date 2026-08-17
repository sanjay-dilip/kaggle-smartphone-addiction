# Decision Log

Records why, not what happened chronologically. See `BUILD_HISTORY.md` for
the chronological record.

## This is a Kaggle competition project, not a deployment project

No Streamlit, FastAPI, Docker, cloud deployment, or orchestration
infrastructure. The deliverable is a reproducible modeling pipeline and
submission history, not a running application.

## Notebooks are the experimentation laboratory

`notebooks/` is where hypotheses are explored. Code only graduates to
`src/` once it is reused across notebooks or accepted as stable logic.

## `src/` holds reusable, accepted logic only

Single-use code stays in the notebook that uses it. Premature abstraction
into `src/` before a second real usage exists is avoided.

## Raw competition data is not committed

`data/*.csv` is gitignored. Kaggle competition data is licensed for
competition use, not redistribution, and the files are large. `data/README.md`
documents how to obtain them.

## Experiments receive stable IDs

`experiments/experiments.csv` is the single source of truth for what was
tried, with what hypothesis, and what the result was. IDs are assigned only
when an experiment is actually run — no placeholder rows.

## Both CV and leaderboard results are tracked

`experiments.csv` has columns for both cross-validation scores and the
public leaderboard score. Neither is deleted or overwritten by the other.

## Public leaderboard performance does not automatically override CV evidence

Public leaderboard scores are computed on a subset of the test set and are
noisier and more overfittable than a well-constructed local CV. When the two
disagree, the CV methodology is scrutinized before trusting the leaderboard
number, not the other way around.

## `id` is excluded from the feature set

Build 1 audited `id` for ordering, batching, generator drift, and a target
relationship (`notebooks/01_eda.ipynb`, Section 13). It is a sequential,
contiguous integer with no detectable structure: target rate is flat
(70.6%-71.3%) across 20 id bins, and feature means/missingness rates show
no drift across 10 id bins. It carries no signal and is dropped from the
feature matrix starting Build 2.

## Plain `StratifiedKFold` is the starting validation scheme

Build 1 checked for the two main reasons to deviate from plain random CV:
duplicate/group leakage and train/test distribution shift. Neither was
found (`notebooks/01_eda.ipynb`, Sections 5 and 9) — no predictor-vector
duplicates within train or test, and no numeric feature showed a
significant KS-test difference between train and test. Build 2 starts with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` rather than a
grouped or shift-aware scheme, revisited only if CV/leaderboard scores
diverge later.

## Missing values are not treated as informative by default

Build 1 tested both per-column missingness and total-missing-count against
the target (`notebooks/01_eda.ipynb`, Section 4.2). The largest per-column
effect was 0.42 percentage points (`sleep_hours`, p ≈ 0.06); target rate by
`row_missing_count` was flat across the bulk of the distribution. Missing
indicators are not assumed to help and are deferred to a low-priority Build
2 experiment (E002) rather than built into the default preprocessing
pipeline.

## Categorical missing values get an explicit "Missing" category, not imputation

`gender`, `stress_level`, and `academic_work_impact` each have 4-8%
missingness with no evidence it is random-within-category (Build 1,
Section 4). Rather than impute a mode and hide that missingness,
Build 2 preprocessing encodes missing as its own explicit category.

## E001 Logistic Regression baseline is the comparison target for future experiments

E001 (median imputation + StandardScaler for numeric, explicit Missing
category + OneHotEncoder for categorical, `StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)`) scores a mean 5-fold ROC AUC of 0.9115
(std 0.0008) — see `experiments/experiments.csv` and
`notebooks/02_baseline.ipynb`. The low fold-to-fold variance is treated as
confirmation that the Build 1 leakage/shift findings hold up under an
actual model fit, not just descriptive statistics. Later experiments
(E002-E005 and Build 3 strong models) are compared against this number
using the same harness.

## `deliverables/` holds submission-ready files

Build 0 originally created a `submissions/` directory for Kaggle
submission tracking, but it was never populated with an actual
submission. When a submission-ready CSV was first requested
(`E001_submission.csv`), it was generated as a standalone handoff artifact
under a new `deliverables/` directory instead — traceable to an
experiment, reproducible via a committed script. `submissions/` was
subsequently deleted (empty except for its README) to avoid two
directories serving the same purpose. See `deliverables/README.md`.
