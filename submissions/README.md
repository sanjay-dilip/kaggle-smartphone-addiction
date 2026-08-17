# Submissions

No submissions have been generated yet. This directory holds Kaggle
submission CSVs once they exist, and this file tracks the convention for
recording them.

## Convention

Each submission should trace back to the experiment that produced it. When
a submission is made, record a row in `experiments/experiments.csv` with:

- `experiment_id` — the experiment that generated the submission
- `submission_file` — the filename in this directory
- `cv_mean` / `cv_std` — local cross-validation score behind the submission
- `public_lb` — the public leaderboard score, once known
- `conclusion` — what the result means
- `next_action` — what to try next

Submission filenames should include the experiment ID (e.g.
`E001_submission.csv`) so a file can be traced back to its row without
opening it.
