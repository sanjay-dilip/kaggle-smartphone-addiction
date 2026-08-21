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
directories serving the same purpose. See `deliverables/CONTENTS.md`.

## XGBoost (E004) is the primary Build 4 control

Build 3 benchmarked CatBoost (E002), LightGBM (E003), and XGBoost (E004)
on the same raw feature set under the frozen `StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)` harness (`notebooks/03_model_benchmarks.ipynb`).
All three beat E001 by a wide margin (deltas +0.049 to +0.052, roughly
50x the combined fold-to-fold noise — not a marginal result). XGBoost had
the best CV mean (0.96382) and becomes the primary control future feature
engineering (Build 4+) is measured against.

## CatBoost (E002) is the secondary Build 4 control; LightGBM (E003) is deprioritized

Among the two benchmarks that clearly beat E001 besides the best model,
CatBoost's out-of-fold predictions were the least correlated with
XGBoost's (0.9879 vs LightGBM's 0.9901 — see
`outputs/oof_prediction_correlation.csv`), making it the more useful
secondary control for checking whether Build 4 feature-engineering gains
generalize across model families, despite CatBoost's much longer training
time (see below). LightGBM scored competitively (0.96106) and trained by
far the fastest, but was dominated by XGBoost on CV and added less
diversity than CatBoost — not selected as a control, but not rejected as
a model family; it remains a fast option to revisit if training cost
becomes a constraint later.

## Native categorical/missing handling used for all three boosters, not one-hot encoding

CatBoost, LightGBM, and XGBoost were all given the same raw feature
treatment: numeric predictors with missing values preserved (native
missing-value handling), categorical predictors filled with the same
explicit "Missing" category used for E001 and cast to a pandas `category`
dtype (native categorical splits). This was a deliberate choice to keep
the three benchmarks comparable — using each library's "best" encoding
independently would have mixed model-family effects with preprocessing
effects, which Build 3 was designed to separate.

## CatBoost's Build 3 benchmark did not fully converge within its training budget

An initial timing check found CatBoost still improving at 2000 boosting
iterations. For runtime practicality, all three boosters were capped at
800 iterations / learning_rate=0.1 (shared budget). CatBoost hit that cap
in every fold (`best_iteration=799` in all 5 folds) without early
stopping ever triggering, while LightGBM converged well inside the budget
(best iterations 190-634) and XGBoost mostly hit the cap too (best
iterations 794-799). CatBoost's 0.96040 CV mean is therefore a
resource-capped result, not its ceiling — this does not change its Build
3 role (secondary control) but should be kept in mind if CatBoost's
relative standing matters later.

## `screen_residual` accepted into the Build 4+ feature set; `component_sum` and the `app_opens_per_day` missingness flag rejected

Build 1 (`notebooks/01_eda.ipynb`, Section 12) found `daily_screen_time_hours
>= social_media_hours + gaming_hours + work_study_hours` holds in 100% of
fully-observed rows, with a right-skewed residual correlated with the
target. Build 4 (`notebooks/04_feature_engineering.ipynb`) tested this as
two candidate features against the frozen XGBoost (E004) and CatBoost
(E002) controls:

- `screen_residual` (`daily_screen_time_hours` minus the three components,
  via `add_screen_residual`): **accepted**. E006 (XGBoost) scored CV mean
  0.96445 (std 0.00056), +0.00062 vs E004, 5/5 folds improved, tight range
  [+0.00050, +0.00072]. E008 (CatBoost transfer test) scored CV mean
  0.96104 (std 0.00055), +0.00064 vs E002, 5/5 folds improved — a near-
  identical delta on a second model family, strong evidence the feature
  carries model-independent signal rather than an XGBoost-specific
  artifact. This is the largest and most consistent gain of any Build 4
  candidate and becomes part of the default feature set going forward.
- `component_sum` (`social_media_hours + gaming_hours + work_study_hours`,
  via `add_component_sum`): **rejected as redundant**, not as individually
  useless. E005 (XGBoost, isolated) scored CV mean 0.96408, +0.00026 vs
  E004, 5/5 folds improved — a real but smaller gain than
  `screen_residual`. E007 (XGBoost, both features combined) scored CV mean
  0.96443, statistically indistinguishable from E006 (`screen_residual`
  alone, 0.96445) — combining the two features provided no gain over
  `screen_residual` by itself. `component_sum` is dropped from the default
  feature set; `screen_residual` alone captures the relationship.
- `app_opens_per_day_is_missing` (via `add_missing_flag`): **rejected as
  noise**. E009 (XGBoost) scored CV mean 0.96384, +0.00002 vs E004, only
  3/5 folds improved (mixed sign) — within fold-to-fold noise, consistent
  with Build 1's finding (`docs/DECISIONS.md`, "Missing values are not
  treated as informative by default") that missingness carries little
  target signal in this dataset.

Evidence trace: `experiments/experiments.csv` (E005-E009 rows) and
`outputs/feature_experiments.csv`.

## No Build 5 generator-inspired feature is accepted; `screen_residual` remains the only one

Build 5 (`notebooks/05_synthetic_generator.ipynb`) investigated the
Playground S6E8 synthetic generator across quantization, arithmetic
structure, exact-value target rates, frequency, repeated/near-duplicate
patterns, missingness structure, `id`/batch drift, and cross-feature
constraints. Two real generator artifacts were confirmed:

- The `daily_screen_time_hours >= social_media_hours + gaming_hours +
  work_study_hours` compositional constraint (0 violations across
  421,427 train and 182,287 test complete rows) — this is the mechanism
  behind the already-accepted `screen_residual` feature, characterized
  rather than newly discovered.
- A `sleep_hours + daily_screen_time_hours` clipping artifact: the sum
  never exceeds 24h but spikes sharply at exactly 20.00h (14,132/559,350
  train rows, vs a few hundred at neighboring 0.01 steps), with
  `sleep_hours` and `daily_screen_time_hours` nearly uncorrelated
  (r=0.03). **Rejected as a feature**: target rate at the cap (0.9997) is
  statistically indistinguishable from rows just below it (0.9995) —
  fully redundant with `daily_screen_time_hours`, which the models
  already use directly.

No repeated-profile, near-duplicate, or missingness-pattern structure was
found strong enough to justify a feature:

- Missingness-pattern encoding: **rejected**. 2,925 of a ~4,096-pattern
  space realized (consistent with near-independent per-column
  missingness); target rate flat (0.696-0.719) across all well-supported
  patterns/`row_missing_count` values.
- Near-duplicate/profile-target-lookup on 0.1h-rounded screen-time
  profiles: **rejected**. Groups with 5+ rows have target rates spanning
  the full 0.0-1.0 range — a lookup feature here would fit sampling
  noise, not signal.
- Frequency encoding on `age`, `notifications_per_day`,
  `app_opens_per_day`: **rejected**. Train/test value-frequency
  distributions are highly stable (corr 0.995-0.999) but carry no
  independent target signal (Spearman corr with target rate near zero
  once each value's own effect is accounted for).

No Bucket 3 (lookup-like/source-reconstruction) or Bucket 2
(transductive) technique was implemented. Source-data fingerprint
assessment: **weak evidence** — real generator structure exists (the two
constraints above) but every check for template reuse or a small latent
source dataset returned negative. No formal Phase B model experiment was
run; E006 (XGBoost + `screen_residual`, CV mean 0.96445, public LB
0.96608) remains the best validated model, unchanged by Build 5.

Evidence trace: `outputs/synthetic_generator_findings.csv`,
`outputs/numeric_quantization_audit.csv`,
`outputs/generator_constraints.csv`.
