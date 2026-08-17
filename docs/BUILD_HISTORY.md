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

Not started.

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
