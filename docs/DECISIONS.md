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
