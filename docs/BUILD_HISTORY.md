# Build History

Chronological record of each build. See `DECISIONS.md` for the reasoning
behind specific choices.

## Build 0 - Competition Foundation

**Objective:** Establish a clean, reproducible repository foundation before
any modeling work begins.

**Work completed:**

- Verified the repository was not yet a Git repository and initialized one
  (independent of the parent `D:/Projects` config-sync repository).
- Created the Kaggle-specific directory structure: `src/`, `notebooks/`,
  `tests/`, `data/`, `experiments/`, `submissions/`, `docs/`, `outputs/`.
- Wrote `.gitignore` protecting raw competition CSVs, the in-project venv
  (`phone-addiction/`), and local-only files (`CONTEXT.md`), while still
  allowing `data/README.md` to be tracked.
- Wrote `.gitattributes` for consistent LF line endings across platforms.
- Added `src/config.py` with centralized, pathlib-based project paths and
  a shared random seed.
- Installed and pinned minimal runtime dependencies (pandas, numpy,
  scikit-learn) and dev dependencies (pytest, ipykernel, jupyter).
- Created `experiments/experiments.csv` with a header-only schema.
- Wrote `data/README.md` and `submissions/README.md`.
- Wrote `docs/DECISIONS.md`, `docs/COMPETITION_NOTES.md`, this file, and
  the initial `README.md`.
- Wrote `tests/test_data_smoke.py` verifying the three competition files
  exist, are readable, have the expected target/predictor columns, and have
  a sample submission row count matching the test set.
- Wrote `CONTEXT.md` (gitignored, not committed) as a resumability document.

**Important decisions:** see `docs/DECISIONS.md`.

**Validation performed:** smoke tests run via pytest against the actual
local competition files (see `CONTEXT.md` for the exact command and
output at build completion).

**Final status:** complete.

## Build 1 - Data Audit and Synthetic-Data EDA

**Objective:** Understand the training and test data deeply enough that
Build 2 can start with a trustworthy validation strategy and a justified
baseline. Data audit only — no models trained, no CV ROC AUC computed, no
submissions generated.

**Work completed:**

- Wrote `notebooks/01_eda.ipynb`, a 16-section audit covering: schema,
  target balance, missingness (train/test and vs. target), exact and
  predictor-only duplicates, train/test predictor overlap, numeric and
  categorical feature audits, univariate target relationships, train/test
  distribution shift (KS tests), pairwise correlations, a targeted
  screen-time component investigation, other deterministic-relationship
  checks, an `id` structure audit, and an explicit leakage assessment.
- Saved supporting audit tables to `outputs/` (`schema_summary.csv`,
  `missingness_summary.csv`, `missingness_vs_target.csv`,
  `numeric_summary_train.csv`, `numeric_summary_test.csv`,
  `train_test_ks.csv`, `categorical_target_rates.csv`,
  `univariate_target_numeric.csv`, `correlation_matrix.csv`) and five
  figures to `outputs/figures/`.
- Added `scipy` and `matplotlib` to `requirements.txt` (deferred from
  Build 0).

**Major findings:**

- Target is moderately imbalanced: 70.9% positive / 29.1% negative.
- Every predictor has missingness (4.2%-19.4% in train); train/test
  missingness gaps are small (largest 3.4 points, `social_media_hours`).
- Missingness is not meaningfully related to the target, per-column or via
  total missing count.
- No predictor-vector duplicates within train or within test; only 2
  negligible train/test predictor-vector overlaps, both explained by
  near-total-missingness rows, not genuine duplication.
- No meaningful train/test distribution shift on any numeric feature (KS
  tests all non-significant) or categorical feature.
- `daily_screen_time_hours`, `weekend_screen_time`, and
  `social_media_hours` carry by far the strongest univariate target
  association; `age`, `notifications_per_day`, and the categoricals carry
  very little.
- `daily_screen_time_hours >= social_media_hours + gaming_hours +
  work_study_hours` holds in 100% of the 421k fully-observed rows, with a
  right-skewed non-negative residual that correlates with the target — a
  candidate engineered-feature hypothesis, not yet implemented.
- `id` is sequential and contiguous with no drift or target relationship.
- No leakage found.

**Decisions made:** see `docs/DECISIONS.md` (`id` excluded from features;
`StratifiedKFold` retained as the starting validation scheme; missingness
not treated as informative by default; categorical missing values get an
explicit "Missing" category).

**Validation performed:** notebook restarted and run top-to-bottom from a
clean kernel via `jupyter nbconvert --to notebook --execute --inplace`
with zero cell errors (verified by scanning all 36 code-cell outputs for
`output_type == "error"`); existing `pytest tests/ -v` suite re-run,
6/6 passing.

**Final status:** complete.

## Build 2 - Validation Harness and Logistic-Regression Baseline

Not started.

## Build 3 - Strong Model Benchmarks

Not started.

## Build 4 - Hypothesis-Driven Feature Engineering

Not started.

## Build 5 - Synthetic-Generator Investigation

Not started.

## Build 6 - Controlled Hyperparameter Tuning

Not started.

## Build 7 - Ensembling and Blending

Not started.

## Build 8 - CV vs Leaderboard Reconciliation

Not started.

## Build 9 - Final Submission Strategy

Not started.

## Build C - Consolidation, Reproduction, Documentation, Publication

Not started.
