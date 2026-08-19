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

**Objective:** Implement a reusable, trustworthy CV harness and
preprocessing pipeline per the frozen Build 1 decisions, and establish a
real, cross-validated Logistic Regression baseline (E001).

**Work completed:**

- Added `src/validation.py` (`get_cv_splitter()`, wrapping
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) and
  `src/preprocessing.py` (`build_preprocessor()`, a `ColumnTransformer`:
  median imputation + `StandardScaler` for numeric predictors, explicit
  "Missing" category + `OneHotEncoder` for categoricals; `id` excluded by
  construction).
- Added `NUMERIC_COLS` / `CATEGORICAL_COLS` / `ID_COLUMN` constants to
  `src/config.py`.
- Added `tests/test_preprocessing.py` (no missing values survive
  transform, `id`/target never referenced, row count preserved) and
  `tests/test_validation.py` (splitter settings match the frozen
  decision, deterministic across runs, stratification holds).
- Wrote `notebooks/02_baseline.ipynb`, which runs experiment **E001**
  (Logistic Regression on the full imputed/encoded feature set) through
  the harness above, fitting the preprocessor fresh inside each fold (no
  cross-fold leakage), and records the result as a real row in
  `experiments/experiments.csv` (idempotent — re-running the notebook does
  not duplicate the row).

**Major findings:**

- E001 mean 5-fold ROC AUC: **0.9115** (std 0.0008) — a stable, credible
  linear baseline, consistent with Build 1's finding of concentrated
  signal in the screen-time feature family and no leakage/shift risk.
- E001 public leaderboard score: **0.91358** (submitted
  `deliverables/E001_submission.csv`) — 0.0021 above the CV mean, well
  within the fold-to-fold std of 0.0008-scale noise expected from Build
  1's finding of no meaningful train/test shift. No CV/LB divergence to
  investigate at this stage.

**Decisions made:** see `docs/DECISIONS.md` (E001 adopted as the
comparison target for later experiments).

**Validation/checks:** notebook run top-to-bottom from a clean kernel via
`jupyter nbconvert --to notebook --execute --inplace`, verified zero cell
errors; re-run a second time to confirm the `experiments.csv` append is
idempotent (still 1 data row after two runs); `pytest tests/ -v` run
directly, 12/12 passing (6 Build 0 smoke tests + 6 new Build 2 tests).

**Final status:** complete.

## Build 3 - Strong Model Benchmarks

**Objective:** Answer one question cleanly — which strong gradient-
boosting model family (CatBoost, LightGBM, XGBoost) performs best on the
current raw feature set, under the frozen Build 2 validation framework —
and select the primary/secondary controls for Build 4 feature
engineering. No feature engineering, hyperparameter search, ensembling,
or final submission strategy in this build.

**Work completed:**

- Added `src/benchmarking.py` (`run_cv_benchmark`, `CVBenchmarkResult`):
  extends the frozen `StratifiedKFold` harness with OOF prediction
  storage and cross-fold test-prediction averaging, while keeping model
  construction/fitting fully model-specific.
- Added `src/boosting_models.py`: CatBoost/LightGBM/XGBoost fold
  callbacks sharing one training budget (`iterations=800`,
  `learning_rate=0.1`, `early_stopping_rounds=50`) and native
  categorical/missing-value handling.
- Added `build_boosting_frame()` to `src/preprocessing.py`: numeric NaNs
  preserved (native handling), categorical NaNs filled to "Missing" and
  cast to `category` dtype (native handling) — the same treatment for
  all three boosters.
- Added `src/submission_validation.py` (`validate_submission`) and
  refactored `generate_submission.py` to use it (behavior-preserving;
  E001's result is unchanged).
- Added `catboost`, `lightgbm`, `xgboost` to `requirements.txt`. No GPU
  available in this environment (verified: no `nvidia-smi`) — CPU only.
- Wrote `notebooks/03_model_benchmarks.ipynb`, which ran E002 (CatBoost),
  E003 (LightGBM), E004 (XGBoost) through the shared harness, built a
  durable comparison artifact (`outputs/model_benchmarks.csv`), computed
  OOF prediction correlations (`outputs/oof_prediction_correlation.csv`),
  and generated + validated submission files for the two benchmarks a
  programmatic diversity rule selected.
- Added `tests/test_benchmarking.py`, `tests/test_boosting_models.py`,
  `tests/test_submission_validation.py`, and 3 new tests in
  `tests/test_preprocessing.py` for `build_boosting_frame`.

**Major findings:**

| Experiment | Model | CV mean | CV std | Delta vs E001 | Public LB | Elapsed |
|---|---|---|---|---|---|---|
| E001 | LogisticRegression | 0.91149 | 0.00081 | — | 0.91358 | 25s |
| E002 | CatBoostClassifier | 0.96040 | 0.00051 | +0.04891 | 0.96151 | 2663s (~44 min) |
| E003 | LGBMClassifier | 0.96106 | 0.00113 | +0.04957 | not submitted | 142s (~2.4 min) |
| E004 | XGBClassifier | 0.96382 | 0.00056 | +0.05233 | 0.96539 | 572s (~9.5 min) |

- All three boosters beat E001 by a wide margin — the smallest delta
  (E002, +0.04891) is roughly 50x the combined fold-to-fold noise of the
  two models being compared. This is not a marginal result.
- XGBoost (E004) had the best CV mean and becomes the primary Build 4
  control.
- CatBoost's fold results (`best_iteration=799` in every fold) show it
  never triggered early stopping within the shared 800-iteration budget —
  a runtime-practicality cap (an earlier 2000-iteration timing check
  found it still improving), not evidence of CatBoost's true ceiling.
  XGBoost mostly hit the same cap (best iterations 794-799). LightGBM
  converged well inside the budget (best iterations 190-634) and trained
  ~19x faster than XGBoost and ~19x faster than CatBoost.
- CatBoost had the lowest fold-to-fold variance (std 0.00051); LightGBM
  had the highest (std 0.00113), consistent with its best_iteration
  varying more across folds (spread 444 vs XGBoost's 5 and CatBoost's 0).
- OOF prediction correlations between the three boosters are all high
  (0.988-0.992), as expected given they are fit to the same strong signal
  Build 1 identified — CatBoost is the least correlated with XGBoost
  (0.9879), making it the more useful secondary control despite its
  longer training time.
- E002 and E004 were selected for submission by a programmatic rule (best
  CV mean, plus the most-diverse model among those that clearly beat
  E001 by 3x E001's fold std); `deliverables/E002_submission.csv` and
  `deliverables/E004_submission.csv` were generated (fold-averaged test
  probabilities, no retraining on full data) and validated against
  `data/sample_submission.csv`. E003 was not submitted.
- Public LB scores (submitted after this build's PR merged): E002 0.96151
  (+0.00111 vs its CV mean), E004 0.96539 (+0.00157 vs its CV mean). Both
  boosters landed slightly above their CV mean, in the same direction and
  similar small magnitude as E001's CV/LB gap (+0.0021) — consistent with
  Build 1's finding of no meaningful train/test shift, no CV/LB divergence
  to investigate. E004 remains the best public LB score, consistent with
  its status as the best CV score.

**Decisions made:** see `docs/DECISIONS.md` — XGBoost (E004) as primary
Build 4 control, CatBoost (E002) as secondary control, LightGBM (E003)
deprioritized as a control (not rejected as a model family), native
categorical/missing handling used for all three boosters, and CatBoost's
benchmark noted as resource-capped rather than converged.

**Validation/checks:** notebook run top-to-bottom from a clean kernel via
`jupyter nbconvert --to notebook --execute --inplace` (total runtime
~55 min), verified zero cell errors across all 14 code cells; both
generated submission files independently re-validated (schema, id order,
range) outside the notebook; `pytest tests/ -v` run directly, 35/35
passing (13 prior + 22 new Build 3 tests, including 6 boosting-model
fold-callback tests parametrized across all three libraries).

**Final status:** complete. Public LB scores for E002/E004 recorded above
and in `experiments/experiments.csv`.

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
