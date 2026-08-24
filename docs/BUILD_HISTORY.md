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

**Objective:** Test the screen-time-component engineered features
identified in Build 1 (`component_sum`, `screen_residual`) plus a
missingness-flag candidate against the frozen Build 3 controls (XGBoost
E004, CatBoost E002), and settle on a default Build 4+ feature set. No
hyperparameter tuning, iteration-budget expansion, ensembling,
adversarial validation, or generator-exploitation tricks in this build.

**Work completed:**

- Added `src/features.py`: `add_component_sum`, `add_screen_residual`,
  `add_missing_flag` — pure, deterministic feature functions with no
  model imports.
- Added `tests/test_features.py` (13 tests: row count, missing
  propagation, non-mutation, no target/id reference, stable output
  names, train/test consistency, no infinities).
- Wrote `notebooks/04_feature_engineering.ipynb`, which ran E005
  (XGBoost + `component_sum`, isolated), E006 (XGBoost +
  `screen_residual`, isolated), E007 (XGBoost + both, combined), E008
  (CatBoost + `screen_residual`, transfer test on the frozen E002
  config), and E009 (XGBoost + `app_opens_per_day_is_missing`); recorded
  all five to `experiments/experiments.csv`; built a durable comparison
  artifact (`outputs/feature_experiments.csv`); computed an OOF
  correlation diagnostic between E006 and E008 (0.9877); ran a
  lightweight feature-importance diagnostic; and generated + validated
  `deliverables/E006_xgb_screen_residual_submission.csv`.

**Major findings:**

| Experiment | Model | Feature set | CV mean | CV std | Delta vs control | Folds improved | Elapsed |
|---|---|---|---|---|---|---|---|
| E005 | XGBClassifier | raw + component_sum | 0.96408 | 0.00056 | +0.00026 vs E004 | 5/5 | 789s |
| E006 | XGBClassifier | raw + screen_residual | 0.96445 | 0.00056 | +0.00062 vs E004 | 5/5 | 849s |
| E007 | XGBClassifier | raw + component_sum + screen_residual | 0.96443 | 0.00055 | +0.00061 vs E004 | 5/5 | 910s |
| E008 | CatBoostClassifier | raw + screen_residual | 0.96104 | 0.00055 | +0.00064 vs E002 | 5/5 | 2436s (~41 min) |
| E009 | XGBClassifier | raw + app_opens_per_day_is_missing | 0.96384 | 0.00052 | +0.00002 vs E004 | 3/5 | 628s |

- `screen_residual` is the strongest and most consistent Build 4
  candidate: clear gain on XGBoost (E006), and the gain transfers to
  CatBoost at a near-identical magnitude (E008, +0.00064 vs E006's
  +0.00062) — strong evidence the signal is model-independent, not an
  XGBoost-specific artifact.
- `component_sum` (E005) is a real but smaller gain in isolation, but
  redundant once `screen_residual` is present: E007 (both features) is
  statistically indistinguishable from E006 (`screen_residual` alone).
- `app_opens_per_day_is_missing` (E009) shows no consistent improvement
  (3/5 folds, mixed sign) — within noise, consistent with Build 1's
  finding that missingness carries little target signal in this dataset.
- Real notebook-execution results matched the session's pre-commit
  scratch validation numbers to 5 decimal places across all five
  experiments — no divergence to investigate.
- E006 public leaderboard score: **0.96608** (submitted
  `deliverables/E006_xgb_screen_residual_submission.csv`) — 0.00163 above
  the CV mean, same direction and similar small magnitude as Build 3's
  E002 (+0.00111) and E004 (+0.00157) CV/LB gaps. No CV/LB divergence to
  investigate. E006 is the best public LB score recorded across all
  builds, surpassing E004's 0.96539.

**Decisions made:** see `docs/DECISIONS.md` — `screen_residual` accepted
into the default Build 4+ feature set; `component_sum` rejected as
redundant (not as individually useless); `app_opens_per_day_is_missing`
rejected as noise.

**Validation/checks:** notebook run top-to-bottom from a clean kernel via
`jupyter nbconvert --to notebook --execute --inplace` (total runtime
~104 min, dominated by the ~41 min CatBoost fit), verified zero cell
errors across all 16 code cells; `deliverables/E006_xgb_screen_residual_submission.csv`
generated and validated in-notebook (296,302 rows, schema matches
`data/sample_submission.csv`); `pytest tests/ -v` run directly, 48/48
passing (35 prior + 13 new Build 4 tests).

**Final status:** complete. Public LB score for E006 recorded above and
in `experiments/experiments.csv`.

## Build 5 - Synthetic-Generator Investigation

**Objective:** forensic investigation of whether the Playground S6E8
synthetic dataset contains repeatable generator structure that is real,
measurable, understandable, relevant to modeling/validation, and
rules-compliant to exploit — distinguishing ordinary predictive
relationships from generator artifacts, and only proposing formal model
experiments if Phase A found a strong, explainable, rules-compliant
candidate. No hyperparameter tuning, ensembling, or final submission
strategy in this build.

**Work completed:**

- Wrote `notebooks/05_synthetic_generator.ipynb`, a 15-section forensic
  audit covering: a revisit of prior (Build 1/4) synthetic-structure
  evidence, numeric quantization/precision, `screen_residual` arithmetic
  structure, exact-value target-rate analysis (with a 200-row minimum
  support threshold), value-frequency analysis, cross-feature repeated
  patterns, near-duplicate analysis, missingness-pattern structure,
  `id`/batch structure, cross-feature constraints, a source-data
  fingerprint assessment, a rules/risk classification table, ranked
  candidate hypotheses, and Phase B justification (none run).
- Added `outputs/numeric_quantization_audit.csv` (grid/precision per
  numeric feature), `outputs/generator_constraints.csv` (four
  semantically-justified constraint checks with train/test violation
  rates), and `outputs/synthetic_generator_findings.csv` (10 findings,
  each with evidence, train/test behavior, target relationship, modeling
  relevance, risk level, and recommended action).
- No changes to `src/` — no new reusable generator-derived feature
  cleared the evidence bar, so no new feature function was added and no
  new tests were required.

**Major findings:**

- **F01 (screen-time composition, real/understood/already exploited):**
  `daily_screen_time_hours >= social_media_hours + gaming_hours +
  work_study_hours` holds exactly in both splits (0/421,427 train,
  0/182,287 test violations). `screen_residual` (the slack) is
  continuous on a clean 0.01 grid, 1,022 distinct values, right-skewed
  (mean 1.34h), never meaningfully negative, KS-consistent between train
  and test (p≈0.39), and strongly related to the target (mean 0.64h for
  `addicted_label==0` vs 1.63h for `addicted_label==1`, Spearman
  corr(exact value, target rate)=0.75 among well-supported values). This
  characterizes, rather than extends, the already-accepted Build 4
  feature.
- **F02 (sleep/screen-time clipping, real generator fingerprint, not
  actionable):** `sleep_hours + daily_screen_time_hours` never exceeds
  24h but spikes sharply at exactly 20.00h (2.53% of complete train
  rows). `sleep_hours` and `daily_screen_time_hours` are nearly
  uncorrelated (r=0.03), consistent with independent draws plus a
  post-hoc joint clip. Target rate at the cap is statistically
  indistinguishable from rows just below it — the effect is fully
  explained by `daily_screen_time_hours` alone. Rejected as a feature.
- **F03 (rounded `daily_screen_time_hours` staircase, ordinary predictive
  relationship):** target rate rises smoothly and near-monotonically from
  ~21% (1.0h bin) to 100% (13-14h bins) across 28 well-supported 0.5h
  bins — the dataset's dominant signal (known since Build 1), already
  captured optimally by tree-based splits.
- **F05-F08 (no repeated-profile/near-duplicate/missingness-pattern
  structure):** the 4-column screen-time combination is 99.99% unique at
  native (0.01) precision (max repeat group size 3); 0.1h-rounded groups
  of 5+ rows have target rates spanning the full 0.0-1.0 range;
  missingness realizes 2,925 of a ~4,096-pattern combinatorial space with
  flat target rates throughout (0.696-0.719); value-frequency is stable
  between train/test (corr 0.995-0.999) but target-uncorrelated. Every
  check argues against a small, reusable source-data template.
- **F09 (`id`/batch, reconfirms Build 1):** `screen_residual` and
  `row_missing_count` means are both flat across 20 `id` bins — no
  batching or generator drift evidence.
- **F10 (source-data fingerprint assessment): weak evidence.** Real
  generator structure exists (F01, F02) but no evidence supports
  reconstructing or exploiting a smaller latent source dataset.
- **Candidate hypotheses (G001-G005):** all rejected or non-actionable on
  Phase A evidence alone — no candidate reached the bar for a formal
  Phase B model experiment. This is a valid, and in this case the actual,
  Build 5 outcome per the build's own scope definition.

**Decisions made:** see `docs/DECISIONS.md` ("No Build 5 generator-inspired
feature is accepted..." section) — sleep/screen-time at-cap flag,
missingness-pattern encoding, near-duplicate/profile-target-lookup, and
frequency encoding on `age`/`notifications_per_day`/`app_opens_per_day`
all rejected; `screen_residual` remains the sole accepted generator-derived
feature.

**Rules/transductive assessment:** no Bucket 3 (lookup-like) or Bucket 2
(transductive) technique was implemented. The one train+test-frequency
idea considered was rejected on evidence before any rules question became
live. No external private data, leaked labels, hidden test labels, other
competitors' outputs, or unauthorized source reconstruction were used.

**Formal Build 5 experiments:** none run. E006 (XGBoost +
`screen_residual`, CV mean 0.96445, public LB 0.96608) remains the best
validated model and best public LB score, unchanged by this build.

**Validation/checks:** notebook run top-to-bottom from a clean kernel via
`jupyter nbconvert --to notebook --execute --inplace`, verified zero cell
errors across all 23 code cells; `pytest tests/ -v` run directly, 48/48
passing (no new tests required — no new `src/` code was introduced).

**Final status:** complete. No public LB submission from this build (no
new candidate feature was accepted, so no new deliverable was generated).

## Build 6 - Controlled XGBoost Hyperparameter Tuning

**Objective:** how much additional ROC AUC can be obtained from the
current best XGBoost model (E006) through controlled hyperparameter
tuning, keeping the accepted feature set (raw predictors +
`screen_residual`) and validation framework fixed. Staged search:
iteration budget/learning rate -> tree complexity -> sampling ->
regularization -> limited joint refinement (only if warranted).
Screening on a single frozen fold used to eliminate poor regions
cheaply; every formal claim comes only from full 5-fold CV. No feature
changes, no ensembling, no LightGBM tuning, no final submission
strategy in this build.

**Work completed:**

- Added `src/tuning.py`: `E006_XGB_PARAMS` (the exact reconstructed E006
  configuration), `make_xgboost_fold`/`run_xgboost_trial` (full 5-fold
  CV via the existing `run_cv_benchmark` harness, parameterized by
  model config), `screen_xgboost_single_fold` (cheap single-fold
  screening, never a substitute for full CV), `paired_fold_deltas`, and
  `tuning_result_row` (formats a formal-experiment row for the tuning
  tracker). Added `tests/test_tuning.py` (7 tests).
- Added `outputs/xgboost_tuning_results.csv`: every screening and
  full-CV trial, with a `stage` column distinguishing screening from
  formal full-CV rows and a `decision` column explaining why each
  candidate was or wasn't promoted.
- Added `outputs/best_xgboost_params.json`: the final tuned
  configuration (E010).
- Added `outputs/e010_e006_oof_correlation.csv`: OOF prediction
  correlation between E010 and E006 (diagnostic only).
- Wrote `notebooks/06_xgboost_tuning.ipynb` (13 sections): setup and
  frozen controls, Phase 0 (E006 reconstruction, run live), Phase 1
  screening and E010's full CV (run live), Phases 2-4 screening (read
  from the tuning tracker rather than recomputed live, with an explicit
  note explaining why — see "Validation/checks" below), the stopping
  rule and final-candidate decision, the OOF correlation diagnostic, the
  best-params artifact, submission generation, consolidated result
  tables, and conclusions.
- Generated and validated
  `deliverables/submission_E010_xgb_tuned.csv` (fold-averaged E010 test
  predictions, 296,302 rows).

**Major findings:**

| Phase | What was tested | Result |
|---|---|---|
| 0 | E006 reconstruction | Confirmed: `best_iterations` `[799, 794, 799, 794, 794]` — 4/5 folds at or one below the 800-iteration cap. Resource-capped, not converged. |
| 1 | `learning_rate` in {0.03, 0.05, 0.07} with proportionally raised `n_estimators` | `lr=0.05/n_estimators=2500` promoted to full CV as **E010**: CV mean **0.96499** (std 0.00051) vs E006's 0.96445 — **+0.00055, 5/5 folds improved**, range [+0.00042, +0.00062]. No fold hit the 2500 ceiling (best_iterations 2294-2485, mean 2390.6) — genuine early-stopping convergence. |
| 2 | `max_depth` in {5, 7}, `min_child_weight=3` | No gain. `max_depth=5` -0.00015, `max_depth=7` -0.00027, `min_child_weight=3` +0.00007 (noise). E010's `max_depth=6, min_child_weight=1` retained. |
| 3 | `subsample`/`colsample_bytree` in {0.8/0.8, 1.0/1.0, 0.7/0.9} | No gain. Best -0.00001 (flat), worst -0.00028 (no subsampling hurts). E010's `subsample=0.9, colsample_bytree=0.9` retained. |
| 4 | `reg_alpha=0.1`, `reg_lambda` in {2.0, 5.0} | No gain. All three within +0.00001 to -0.00005 (noise). E010's `reg_alpha=0, reg_lambda=1` retained. |

- Phases 2, 3, and 4 all found nothing exceeding the ~0.0005
  single-fold noise band observed throughout this build. Per the Build
  6 stopping rule ("several consecutive sensible configurations fail to
  improve the current best"), **joint refinement was skipped and E010
  adopted directly as Build 6's final tuned XGBoost configuration** —
  no full-CV run was spent re-confirming a null result.
- E010's out-of-fold predictions correlate with E006's at Pearson
  0.9967 (Spearman 0.9974) — expected, since E010 refines E006's exact
  architecture and feature set rather than introducing a diverse
  alternative. Diagnostic only; does not motivate any Build 7
  ensembling decision on its own. CatBoost/E008's OOF predictions were
  not recomputed (a full CatBoost CV run takes ~40-44 min; explicitly
  optional/diagnostic-only per the build's scope, not required for
  acceptance).

**Decisions made:** see `docs/DECISIONS.md` — E010 becomes the new
primary XGBoost control, superseding E006, with the stopping rationale
above.

**Validation/checks:** notebook run top-to-bottom from a clean kernel
via `jupyter nbconvert --to notebook --execute --inplace`, verified
zero cell errors across all 17 code cells. Phase 0 (E006
reconstruction) and Phase 1's promoted candidate (E010's full CV) were
executed live in the notebook as reproducibility anchors and both
exactly matched their previously recorded values (E006: CV mean
0.96445/std 0.00056 both matching exactly; E010: CV mean 0.96499/std
0.00051, 5/5 folds improved, matching exactly). Phase 2-4's 9 screening
trials were not re-executed live in this notebook run — they were run
interactively earlier in the same investigation using the identical
code path (`screen_xgboost_single_fold`, same seed, same deterministic
splitter) and are read from `outputs/xgboost_tuning_results.csv` in the
notebook rather than recomputed, to keep total notebook runtime
practical (a full live re-run of all ~15 single-fold and full-CV
XGBoost fits was measured to take 2+ hours on the CPU-only environment
used for this build; the reproducibility anchors above establish that
the underlying code path is correct and deterministic). `pytest tests/
-v` run directly, 55/55 passing (48 prior + 7 new Build 6
`test_tuning.py` tests).

**Final status:** complete. `deliverables/submission_E010_xgb_tuned.csv`
submitted to Kaggle (user action); public LB score **0.96653**, recorded
in `experiments/experiments.csv`'s E010 row — an improvement over E006's
0.96608 and the new best public LB score. Merged to `main` via PR #22.

## Build 7 - Ensembling and Blending

**Objective:** determine whether combining genuinely different strong
models beats E010 (frozen single-model control, CV mean 0.96499, public
LB 0.96653), via a disciplined diversity-check -> simple-blend ->
weighted-blend -> (rank averaging / stacking only if justified) pipeline.
Ensembling is not pursued for its own sake — every step requires evidence
before proceeding to the next. No new features, no retuning of any base
model's hyperparameters; only combination of already-frozen models.

**Candidate inventory** (from `experiments/experiments.csv` and
`outputs/model_benchmarks.csv`, before any Build 7 compute):

| Experiment | Model | Feature set | CV mean | CV std | Public LB | Role |
|---|---|---|---|---|---|---|
| E010 | XGBoost (tuned) | raw + `screen_residual` | 0.96499 | 0.00051 | 0.96653 | **Primary / benchmark** |
| E008 | CatBoost | raw + `screen_residual` | 0.96104 | 0.00055 | — | Diversity candidate (different model family) |
| E006 | XGBoost (pre-tuned) | raw + `screen_residual` | 0.96445 | 0.00056 | 0.96608 | Suspected redundant — already measured at Pearson 0.9967 vs E010's OOF; tested anyway to formally answer whether the pre-tuned XGBoost adds anything despite being same-family |
| E003 | LightGBM | raw only | 0.96106 | 0.00113 | — | Diversity candidate (different family *and* a weaker/different feature set — never tested with `screen_residual`) |
| E002 | CatBoost | raw only | 0.96040 | 0.00051 | 0.96151 | Dominated — same model as E008, strictly worse feature set; excluded |
| E001 | LogisticRegression | raw | 0.91149 | 0.00081 | 0.91358 | Dominated — ~5.3pt CV AUC gap to E010, too weak to plausibly help; excluded from blending |
| E005, E007, E009 | XGBoost variants | rejected features | <=0.9639 | — | — | Dominated — rejected features (Build 4), redundant with E006/E010; excluded |

Active pool: **E010, E008, E006, E003** (4 models). Every other
experiment is excluded from blending consideration and stays in this
table only for completeness — none earns inclusion given the "strong
individually + meaningfully different" bar this build requires.

**Key finding shaping the rest of this build:** no experiment in this
repository has ever persisted a raw per-row OOF or test prediction array
to disk. `outputs/oof_prediction_correlation.csv` and
`outputs/e010_e006_oof_correlation.csv` contain only the scalar
correlation values computed transiently inside their respective
notebooks — the underlying arrays were discarded once each notebook
finished running. Specifically: E010's OOF array was never saved (only
its fold-averaged test predictions survive, as
`deliverables/submission_E010_xgb_tuned.csv`); E006 was reconstructed in
`notebooks/06_xgboost_tuning.ipynb` without `X_test`, so no E006 test
predictions exist; E008 was run in `notebooks/04_feature_engineering.ipynb`
with `X_test=None`, so no E008 test predictions exist; E003 has never had
test predictions generated at all. All four candidates' OOF and test
predictions must therefore be regenerated by reconstructing each
experiment's exact frozen configuration (verified against its recorded
`cv_mean`) before any blending work can begin.

**Prediction artifact regeneration:** all four candidates' OOF and test
predictions were regenerated via `src/regenerate_ensemble_predictions.py`,
run once in the background (E003 4 min, E006 15 min, E010 37 min, E008
35 min). Every reconstructed `cv_mean` matched its recorded
`experiments.csv` value exactly (E003 0.96106, E006 0.96445, E010
0.96499, E008 0.96104) before its artifacts were trusted. Predictions
are rounded to 8 decimals on save (`src.ensembling`) — re-verified this
changes `cv_mean` by <1e-5, far below any meaningful signal, while
cutting artifact size by ~30%. Persisted to
`outputs/oof_predictions/{E003,E006,E010,E008}.csv`,
`outputs/test_predictions/{...}.csv`, and one canonical
`outputs/cv_fold_assignments.csv` (all four candidates share the same
`StratifiedKFold(seed=42)` fold membership, since it depends only on row
count and `y`, not on the feature set or model).

**OOF diversity analysis** (`outputs/ensemble_prediction_correlations.csv`,
via `src.ensembling.pairwise_diversity`):

| Pair | Pearson | Spearman | Mean abs diff | Top-decile disagreement | Bottom-decile disagreement |
|---|---|---|---|---|---|
| E010 vs E008 | 0.9849 | 0.9847 | 0.0357 | 0.281 | 0.116 |
| E010 vs E006 | 0.9967 | 0.9974 | 0.0151 | 0.104 | 0.056 |
| E010 vs E003 | 0.9849 | 0.9833 | 0.0319 | 0.328 | 0.109 |
| E008 vs E003 | 0.9898 | 0.9783 | 0.0254 | 0.425 | 0.100 |
| E008 vs E006 | 0.9877 | 0.9865 | 0.0315 | 0.266 | 0.107 |
| E006 vs E003 | 0.9872 | 0.9842 | 0.0282 | 0.326 | 0.101 |

- **Which model is least correlated with E010?** No single clean winner
  on aggregate correlation alone — E008 and E003 are both ~0.985 Pearson
  vs E010, a near-tie. Top-decile disagreement separates them: E003
  (LightGBM) disagrees with E010 more often in the highest-probability
  decile (32.8% vs E008's 28.1%), while E008 (CatBoost) shows larger raw
  magnitude differences overall (mean abs diff 0.0357 vs 0.0319) and
  slightly more bottom-decile disagreement. **Both models are genuinely
  diverse relative to E010, in different regions of the probability
  distribution** — this is why Pearson alone is not sufficient evidence,
  per the Build 7 brief.
- **Is that model still strong enough to help?** Yes for both: E008 (CV
  0.96104) and E003 (CV 0.96106) are essentially tied individually,
  about 0.004 AUC below E010 (0.96499) — a real but modest gap, not the
  "weak model with low correlation" pattern the brief warns against.
- **Are E006 and E010 too similar to bother blending?** By every metric,
  yes — 0.9967 Pearson, 0.9974 Spearman, the lowest mean abs diff
  (0.0151) and the lowest top/bottom-decile disagreement (0.104/0.056)
  of all six pairs, each by a wide margin. E006 is retained in the
  active pool only to formally test this in Phase 4 (per the brief's
  explicit "does the pre-tuned XGBoost add anything" question), with the
  expectation of little-to-no gain.
- **Does LightGBM add diversity beyond CatBoost?** Yes, clearly — E008
  vs E003's own pairwise numbers show the *highest* top-decile
  disagreement of any pair measured (42.5%), higher than either model's
  disagreement with E010. CatBoost and LightGBM are not redundant with
  each other despite comparable individual CV scores.

**Equal-weight (50/50) probability blends** (`outputs/ensemble_results.csv`;
classification: Clear >=+0.0005, Marginal >+0.0001, Flat within +/-0.0001,
Worse <-0.0001 — the Clear threshold matches Build 6's own established
meaningful-gain bar, E010's +0.00055 over E006):

| Pair | OOF AUC | Delta vs E010 (0.96499) | Decision |
|---|---|---|---|
| E010 + E008 (50/50) | 0.96414 | -0.00085 | Worse |
| E010 + E003 (50/50) | 0.96417 | -0.00082 | Worse |
| E010 + E006 (50/50) | 0.96495 | -0.00004 | Flat |
| E008 + E003 (50/50) | 0.96187 | -0.00312 | Worse |

Every 50/50 blend involving E010 is worse than E010 alone — expected and
mechanical, not evidence against the diversity found above: averaging a
0.96499 model equally with a ~0.961 model necessarily pulls the mean
down regardless of how genuinely diverse the weaker component's errors
are. E010+E006 is flat, confirming Phase 3's redundancy finding at the
blend level too. Per the brief's own weighted-grid framing ("the better
individual model should generally receive more weight"), the real test
of whether E008/E003's diversity is exploitable is a weight grid that
favors E010 heavily, not the 50/50 point — so all four pairs proceed to
the Phase 5 weighted grid (cheap to run, and E010+E006/E008+E003 close
out the brief's explicit "does the pre-tuned XGBoost add anything" and
"CatBoost+LightGBM" questions with real numbers rather than assumption).

**Weighted probability blends** (coarse grid, `outputs/ensemble_results.csv`):
E010 weight in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} for the three pairs
including E010; E008 weight in {0.3, 0.4, 0.5, 0.6, 0.7} for E008+E003.

| Pair | Best point | OOF AUC | Delta vs E010 | Pattern |
|---|---|---|---|---|
| E010 + E008 | E010=0.90 | 0.96498 | -0.00001 | Monotonic Worse->Flat as E010 weight rises; never exceeds E010 |
| E010 + E003 | E010=0.90-0.95 | 0.96498-0.96500 | -0.00001/+0.00001 | Same pattern, same ceiling |
| E010 + E006 | E010=0.80 | 0.96503 | +0.00004 | Flat plateau around E010=0.6-0.9, tiny positive noise |
| E008 + E003 | E008=0.50 | 0.96187 | -0.00312 | Never approaches E010 at any weight |

No weight, for any pair, reaches the +0.0005 Clear bar. E010+E008 and
E010+E003 both trace the same shape: as E010's weight rises the blend
converges toward E010's own score from below, plateauing at essentially
E010 (0.96498-0.96500) rather than exceeding it -- the genuine diversity
found in Phase 3 does not translate into a net OOF gain once weighted
correctly. E010+E006's small positive plateau (+0.00002 to +0.00004
across weights 0.6-0.9) is well inside the ~0.0005 single-fold noise
band established in Build 6, not a real signal. This is a broad, stable
*null* result (matching the brief's own trustworthiness bar -- a wide
flat region, not an isolated spike), which is itself informative: the
weight search did not overfit to noise, it consistently found nothing.

**Rank averaging:** tested at E010 weight {0.5, 0.8, 0.9} for all three
E010 pairs. Results are essentially identical to probability blending at
every point (e.g. E010=0.8/E006=0.2: rank AUC 0.96503 vs probability AUC
0.96503) -- rank averaging offers no advantage over probability blending
for this candidate set. Not adopted.

**Three-model blend (Phase 6): explicitly skipped.** The brief's own
gate requires both E010+E008 and E010+E003 to show real two-model gains
before attempting a trio -- neither did (both plateau at Flat, never
exceeding E010). Adding a third component on top of two null results
would not manufacture a gain neither pairwise combination could find.

**Fold-level stability** (best surviving candidate -- E010+E006 at
0.80/0.20 probability weights, the single least-negative point found
across every blend tested):

| Fold | E010 | E010+E006 (0.8/0.2) | Delta |
|---|---|---|---|
| 1 | 0.96429 | 0.96433 | +0.00004 |
| 2 | 0.96500 | 0.96503 | +0.00002 |
| 3 | 0.96509 | 0.96513 | +0.00004 |
| 4 | 0.96585 | 0.96591 | +0.00006 |
| 5 | 0.96475 | 0.96477 | +0.00002 |

5/5 folds improved, but every improvement is a fraction of the noise
band (+0.00002 to +0.00006) and CV std is unchanged (0.000508 ->
0.000518). Consistent direction across all 5 folds rules out the gain
being a single-fold fluke, but consistency of a microscopic effect is
still a microscopic effect -- this is exactly the pattern expected from
averaging two highly correlated models from the same architecture family
(mild variance reduction, no real information gain), not evidence of a
usable ensemble.

**Stacking decision gate (Phase 9): rejected.** The weighted grid already
performed an exhaustive linear search over every pairwise combination of
these four candidates and found no region above E010 for any pair. A
`LogisticRegression` meta-model over the same probability inputs is
itself a linear combination -- it cannot discover a solution the coarse
grid's dense coverage of that same linear space did not already surface.
Implementing a nested-CV stacker here would add real complexity
(leakage-safe validation machinery) to re-confirm a null result already
established more simply and more transparently. Per the brief's own
allowance, "Stacking rejected as unnecessary" is a fully valid Build 7
outcome.

**Build 7 core finding: no ensemble beats E010.** Across equal-weight,
weighted, and rank-averaged blending of every candidate pair (and the
explicitly-skipped three-model trio), the frozen E010 configuration
remains the best available model. E008 and E003 are genuinely diverse
from E010 (Phase 3) but too far below its individual strength (~0.004
AUC) for that diversity to net a gain once weighted correctly; E006 is
too redundant with E010 to contribute anything beyond noise-level
variance reduction. This extends the brief's own explicitly-anticipated
outcome ("A perfectly acceptable Build 7 decision is: Stacking rejected
as unnecessary") one level further: no blend of any kind earns inclusion
under the Core Build 7 Rule's bar (strong individual performance +
sufficiently different errors + measurable CV improvement when combined
+ reasonable leaderboard behavior) -- the third criterion was never met
by any candidate.

**Submission selection (Phase 10): no Build 7 submission.** Per the
brief's own "do not submit dominated blends" rule and the core finding
above, no blend earns a Kaggle submission this build -- E010's standing
public LB (0.96653) remains the reference. Confirmed with the user
rather than assumed, given it closes off the build's headline question.
No new row was added to `experiments/experiments.csv`: none of the
eleven blend/rank trials in `outputs/ensemble_results.csv` cleared the
bar for a "formal experiment" (adopted config + submission), mirroring
Build 6's precedent of keeping screening-only trials out of
`experiments.csv` and recording them only in the trial tracker.

**Final status:** complete. E010 remains the frozen best model and best
public LB (0.96653), unchanged by Build 7. `notebooks/07_ensemble.ipynb`
run top-to-bottom via papermill, zero cell errors. `pytest tests/ -v`:
71/71 passing (55 prior + 16 new `tests/test_ensembling.py` tests). No
new features, no base-model hyperparameters changed.

## Build 8 - CV vs Leaderboard Reconciliation

**Objective:** how trustworthy has local cross-validation been relative
to Kaggle's public leaderboard, and what does that imply about
overfitting risk and confidence in E010 entering Build 9's final
submission decision? Analytical only -- no new features, tuning,
ensembling, or Kaggle submissions.

**Experiments reconciled:** the 5 actually-submitted experiments (E001,
E002, E004, E006, E010) -- E003/E005/E007/E008/E009 were validated
locally only (no `public_lb` recorded in `experiments/experiments.csv`)
and excluded from CV-to-LB analysis, per the build's own rule.
`outputs/cv_lb_reconciliation.csv` is the canonical table.

**Rank consistency:** perfect. Spearman rho = Kendall tau = 1.0 across
all 5 submissions -- CV and public LB agree on the exact same ranking
with zero inversions anywhere in the project's history. With only 5
points this is a small sample, but the descriptive fact (no rank
inversions ever occurred) is the meaningful result, not the precision of
the coefficient.

**CV-LB gap behavior:** consistently positive (public LB always above
CV) with mean 0.00159, median 0.00157, range [0.00111, 0.00209], std
0.00035. The four boosting-model gaps (0.00111-0.00163) sit in a
noticeably tighter band than the outlier (E001, Logistic Regression,
0.00209) -- and the gap does *not* widen with model strength; E010's gap
(0.00154) is smaller than E006's (0.00163) despite being the more-tuned
model. Read as a systematic measurement offset (the public LB scores a
fixed, different subset of the true test set than any local CV fold),
not as evidence the leaderboard is "easier" or that later models are
becoming leaderboard-specific.

**Incremental-transfer findings** (`outputs/model_progression.csv`,
chronological submission order E001->E002->E004->E006->E010): every
transition's CV-delta and LB-delta direction matched, all 4 times, with
transfer ratios (delta_lb/delta_cv) of 0.98, 1.14, 1.10, and 0.83 -- all
within roughly +/-20% of 1.0. The final, smallest transition (E006->E010,
delta_cv +0.00054) had a transfer ratio of 0.83 (LB improved *less* than
CV predicted) -- mild conservatism, the opposite direction from the
overfitting-warning pattern (LB improving *more* than CV would).

**E010 vs E006 (Phase 7):** 5/5 folds improved (tight delta range
+0.00042 to +0.00062, no anomalous single fold), CV std improved
(0.00056 -> 0.00051, E010 more stable, not less), and public LB confirmed
the direction (+0.00045). Fold stability across the full winning lineage
(E001->E004->E006->E010) trends *down* in both `cv_std` (0.00081 ->
0.00056 -> 0.00056 -> 0.00051) and fold range (0.00227 -> 0.00166 ->
0.00169 -> 0.00156) -- stronger models did not become less stable.

**Public-LB selection-bias audit:** 5 total submissions, all traced to a
CV comparison finalized *before* the corresponding submission (per-
experiment audit in the notebook Section 8). No parameter was ever
changed in response to an LB score; no rejected CV model was ever
resurrected because of LB; Build 7's ensembling investigation made zero
submissions by explicit design specifically to avoid LB-driven weight
tuning. **Public leaderboard selection-bias risk: Low.**

**Private-LB risk assessment:** lower-risk factors -- perfect CV/LB rank
agreement, improving (not degrading) fold stability, no LB-driven
tuning anywhere in the project's history, no transductive/lookup/
pseudo-labeling tricks (Build 5's forensic generator audit tested for
and rejected such structure), no ensemble weight tuning against LB
(Build 7). Higher-risk factors -- the final tuning gain is small in
absolute terms (+0.00054 CV), the entire winning lineage (E004, E006,
E010) is a single model family (XGBoost) with zero cross-family
diversification by construction (Build 7 found no ensemble beats E010),
and Build 5 found real (if unexploited) generator structure whose
public/private-split behavior is not fully characterized. **Private
leaderboard robustness risk: Moderate** -- process hygiene argues
against "High," but model-family concentration and gain size mean "Low"
would overstate confidence. This is a risk characterization, not a
private-score prediction.

**Hedge candidate assessment (Phase 12):** E008 (CatBoost +
`screen_residual`) is the only structurally plausible hedge -- genuinely
diverse from E010 (Build 7: 0.985 Pearson, 28% top-decile disagreement)
and individually reasonable (CV 0.96104) -- but it was never submitted,
so it carries zero public LB evidence, unlike every other candidate
considered here. Flagged for Build 9's consideration rather than
confirmed or rejected outright: Build 9 should decide whether obtaining
that confirmation is worth a submission given remaining budget. E006 is
explicitly *not* a hedge candidate -- too redundant with E010 (Build 7:
0.9967 Pearson) to offer any diversification benefit despite having LB
evidence.

**Build 9 candidate pool:**
- **Primary:** E010.
- **Hedge (flagged, unconfirmed):** E008.
- **Historical reference:** E001, E002, E004, E006.
- **Rejected:** E003, E005, E007, E009 (never submitted, already
  rejected on CV evidence in their own builds).

**No new Kaggle submissions this build**, per scope. No new experiment
ID assigned -- Build 8 performed no new predictive-configuration
training or evaluation.

**Final status:** complete. `notebooks/08_cv_leaderboard_reconciliation.ipynb`
run top-to-bottom via papermill, zero cell errors. `pytest tests/ -v`:
71/71 passing (no new reusable code introduced -- all calculations are
notebook-local pandas/scipy over existing artifacts, per the build's own
allowance to skip new tests in that case).

## Build 9 - Final Submission Strategy

**Objective:** make the final competition submission decision using the
validated evidence accumulated across Builds 1-8, actual Kaggle
constraints, model diversity, public-LB behavior, and private-LB risk.
Decision and competition-close preparation only -- no new feature
engineering, hyperparameter tuning, model-family benchmarking, blending,
stacking, or exploratory modeling unless a genuine integrity problem was
discovered (none was).

**Live Kaggle constraints verified** (`kaggle.com/competitions/playground-series-s6e8`,
checked 2026-08-24): competition open, 7 days to go; Final Submission
Deadline August 31, 2026, 11:59 PM UTC; daily submission limit 10; up to
2 Final Submissions selectable for judging; prize structure is Kaggle
merchandise only (no points/medals).

**E010 integrity audit:** the committed deliverable
(`deliverables/submission_E010_xgb_tuned.csv`) matches the persisted,
`cv_mean`-verified `outputs/test_predictions/E010.csv` artifact to
within 5e-9 (floating-point rounding only). No reproducibility or
integrity defect found. **E010 retained as primary**, unchanged from
Build 8 -- its CV/LB standing was already established in Builds 6 and 8
and is not reopened here.

**E008 hedge question resolved: retained as the second final
submission.** Evaluated against the Build 9 hedge-decision framework
(strength, stability, diversity, model-family independence, feature
dependence, existing LB evidence) -- see `docs/DECISIONS.md`'s Build 9
entry for the full rationale. Summary: CV 0.96104 (~0.004 below E010,
the same magnitude gap `screen_residual` closed on CatBoost in Build 4),
CV std 0.00055 (same tight band as every other boosting experiment), and
the only historical candidate with genuine prediction diversity from
E010 (Build 7: 0.985 Pearson, 28.1% top-decile disagreement vs E006's
0.997/10.4%) plus real model-family independence (CatBoost vs E010's
XGBoost) -- directly addressing the single-model-family risk factor
Build 8 flagged. E006 was not reconsidered, per Build 7's redundancy
finding.

**E008 submitted exactly as already validated, decided before observing
its public LB.** `deliverables/submission_E008_catboost_screen_residual.csv`
was generated in `notebooks/09_final_submission_strategy.ipynb` by
loading the existing, `cv_mean`-verified `outputs/test_predictions/E008.csv`
artifact (Build 7 regeneration) via `src.ensembling.load_test_pred`,
aligning to `data/sample_submission.csv`'s id order, and validating with
`src.submission_validation.validate_submission` -- no retraining, no
parameter/feature/seed/fold-setup change. Schema validation passed
(296,302 rows, unique ids exactly matching the sample submission,
predictions in [0.000197, 0.999999], no missing/non-finite values). No
new experiment ID was assigned -- E008 already exists in
`experiments/experiments.csv`; this build only closes its missing
public-LB evidence gap. **Public LB: pending manual Kaggle upload and
score entry**, recorded in `docs/DECISIONS.md` as a decision made prior
to observing the result.

**Final candidate set:** primary E010 (CV 0.96499, public LB 0.96653),
second/hedge E008 (CV 0.96104, public LB pending). Documented with full
risk-matrix rationale in `outputs/final_submission_candidates.csv` and
`notebooks/09_final_submission_strategy.ipynb` (Sections 7-8). No new
modeling of any kind was performed -- no features, tuning, blending, or
stacking.

**`docs/COMPETITION_CLOSE_CHECKLIST.md` created** -- the human action
list for final selection before the 2026-08-31 23:59 UTC deadline
(upload both files if not already uploaded, record E008's public LB,
mark exactly E010 and E008 as the two Final Submissions, verify the
marked scores match) and for after competition close (record private
LB, final rank, percentile).

**Final rank, private LB, percentile, and medal/status are explicitly
marked pending** in `docs/COMPETITION_CLOSE_CHECKLIST.md` and
`README.md` -- not estimated, per the build's own rule against premature
final-result claims.

**Final status:** complete. `notebooks/09_final_submission_strategy.ipynb`
run top-to-bottom via papermill, zero cell errors across all 11 code
cells. `pytest tests/ -v`: 71/71 passing (no new reusable `src/` code
introduced -- the submission-generation logic reuses `src.ensembling`
and `src.submission_validation` exactly as Build 7 built them, so no new
tests were required). Merged to `main` via PR (see PR link in this
build's final handoff). Next step: user manually uploads/selects final
submissions on Kaggle per the close checklist, then waits for E008's
public LB and, eventually, competition close, before Build C begins.

## Build C - Consolidation, Reproduction, Documentation, Publication

Not started.
