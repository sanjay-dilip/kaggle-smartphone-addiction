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

Missing values are present across multiple predictor columns in both
`train.csv` and `test.csv`. This is noted only — no missingness analysis,
imputation strategy, or statistics have been produced. That belongs to
Build 1.
