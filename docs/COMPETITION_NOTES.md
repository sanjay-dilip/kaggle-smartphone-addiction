# Competition Notes

Stable facts about the competition. Derived from what is verifiable locally
plus information provided directly by the user; anything not independently
verifiable from local files is marked as such.

## Identity

- Competition: Kaggle Playground Series - Season 6, Episode 8
- Title: Predicting Smartphone Addiction
- Task: binary classification
- Target column: `addicted_label` (verified in `data/train.csv` header)
- Evaluation metric: ROC AUC (as provided by the user; not independently
  verifiable from local files alone)
- Final deadline: August 31, 2026 (as provided by the user; not
  independently verifiable from local files alone)
- Predictions must be probabilities, not hard class labels (consistent with
  ROC AUC as the evaluation metric and with the placeholder values already
  present in `data/sample_submission.csv`)

## Local files present

- `data/train.csv` — 691,369 data rows
- `data/test.csv` — 296,302 data rows
- `data/sample_submission.csv` — 296,302 data rows

## Verified schema

`train.csv` columns: `id, age, daily_screen_time_hours, social_media_hours,
gaming_hours, work_study_hours, sleep_hours, notifications_per_day,
app_opens_per_day, weekend_screen_time, gender, stress_level,
academic_work_impact, addicted_label`

`test.csv` has the same columns except `addicted_label`.

`sample_submission.csv` columns: `id, addicted_label`.

Missing values are present across every predictor column in both
`train.csv` and `test.csv` (4.2%-19.4% in train). Full missingness
statistics, a missingness-vs-target analysis, and imputation
recommendations were produced in Build 1 — see
`notebooks/01_eda.ipynb` and `docs/BUILD_HISTORY.md`.

## Target balance (Build 1)

`addicted_label` is moderately imbalanced in train: 70.9% positive
(490,474 rows), 29.1% negative (200,895 rows).

## Data structure notes (Build 1)

- `id` is a simple sequential integer, unique and contiguous within each
  split (train 0-691368, test 691369-987670), with no detectable
  relationship to the target or other features.
- `daily_screen_time_hours` is never less than
  `social_media_hours + gaming_hours + work_study_hours` in any of the
  421,427 train rows where all four are present — consistent with, but not
  proof of, a generator that composes daily screen time from named
  components plus an additional non-negative term. See
  `notebooks/01_eda.ipynb`, Section 11, for the full investigation.
- No train/test distribution shift was detected on any predictor (KS tests
  on numeric features, proportion comparisons on categoricals).
- No exact or predictor-only duplicate rows were found within train or
  within test; no leakage was found. See `notebooks/01_eda.ipynb`,
  Sections 5 and 14.
- All continuous hour-based predictors sit on a clean 0.01 grid (negligible
  float noise, under 5 rows per split); `age`, `notifications_per_day`, and
  `app_opens_per_day` are exact integer grids. Train and test use identical
  grids and ranges. See `notebooks/05_synthetic_generator.ipynb`, Section 3,
  and `outputs/numeric_quantization_audit.csv`.
- `sleep_hours + daily_screen_time_hours` never exceeds 24h and empirically
  caps at ~20h with a sharp frequency spike exactly at 20.00h — a real
  generator clipping fingerprint, but redundant with `daily_screen_time_hours`
  for modeling purposes (target rate at the cap is statistically
  indistinguishable from rows just below it). See
  `notebooks/05_synthetic_generator.ipynb`, Section 10, and
  `outputs/generator_constraints.csv`.
- Build 5's broader forensic investigation (quantization, repeated/near-
  duplicate patterns, missingness structure, `id`/batch drift, source-data
  fingerprint) found no evidence supporting a small, reusable latent source
  dataset — see `docs/BUILD_HISTORY.md` (Build 5 entry) for the full
  finding list.
