# Data

Raw competition data is **not** committed to this repository. It is
intentionally excluded via `.gitignore` (`data/*.csv`).

## Source

The original competition's data, obtained under its own terms — see the
top-level `README.md` for competition context. This repository does not
reproduce or distribute it.

## Expected files

After downloading and extracting the competition data, place these files
directly in this `data/` directory:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Setup after a fresh clone

1. Obtain the competition data from its original source.
2. Extract the three CSV files listed above into `data/`.
3. Run the smoke checks in `tests/` to confirm the files are usable.
