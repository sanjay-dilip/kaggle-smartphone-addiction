# Data

Raw competition data is **not** committed to this repository. It is
intentionally excluded via `.gitignore` (`data/*.csv`).

## Source

Kaggle Playground Series - Season 6, Episode 8: Predicting Smartphone
Addiction. You must accept the competition rules on Kaggle yourself before
downloading any data — this repository does not reproduce or distribute it.

## Expected files

After downloading and extracting the competition data, place these files
directly in this `data/` directory:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Setup after a fresh clone

1. Accept the competition rules at the Kaggle competition page.
2. Download the competition data (e.g. via the Kaggle CLI or the website).
3. Extract the three CSV files listed above into `data/`.
4. Run the smoke checks in `tests/` to confirm the files are usable.
