# Predicting Smartphone Addiction

A gradient-boosting pipeline that predicts smartphone addiction risk from
self-reported usage data, built for a Kaggle competition.

---

## Overview

The task is binary classification: given a person's daily screen time,
app usage, sleep, and a few categorical lifestyle fields, predict the
probability that they're flagged as addicted (`addicted_label`), scored
by ROC AUC. The dataset is synthetic (generated, not scraped from real
users), which changes what "understanding the data" means: instead of
asking what's true about smartphone use in general, the project spends
real effort asking what's true about *this generator specifically* -
where the signal actually sits, whether it's exploitable, and whether
exploiting it would even be legitimate.

Current state: seven builds in, best validated model still at CV mean
ROC AUC 0.96499 (public leaderboard 0.96653), reached via controlled
XGBoost hyperparameter tuning. A subsequent ensembling investigation
found no combination of models that beats it - see below.

## Why I built this

I wanted a project that forced discipline around validation, not just
model fitting - a competition with a public leaderboard is a good way to
find out fast whether your cross-validation actually agrees with reality.
It's also a deliberate exercise in staged, auditable development: every
build has a stated objective, a frozen control to compare against, and a
paper trail of what was tried and rejected, not just what worked.

## Key features

- A validated 5-fold cross-validation harness that CatBoost, LightGBM,
  and XGBoost all run through identically, so model comparisons aren't
  contaminated by different preprocessing
- An engineered feature (`screen_residual`) derived from a compositional
  relationship found in the data during EDA, validated against two
  separate model families before being accepted
- A forensic investigation notebook that audits the synthetic generator
  itself - quantization, arithmetic constraints, missingness patterns,
  ID drift, near-duplicate structure - and documents what was found,
  rejected, and why
- An experiment tracker (`experiments/experiments.csv`) recording every
  model run with its exact configuration, fold scores, and conclusion, so
  no result exists only as a remembered number

## Tech stack

- Python
- pandas, NumPy
- scikit-learn (preprocessing, cross-validation)
- XGBoost, CatBoost, LightGBM
- SciPy (statistical tests)
- Jupyter (experimentation), pytest (testing)
- Git / GitHub, issue-branch-PR workflow

## Project workflow

1. Data audit - schema, missingness, duplicates, distribution shift,
   leakage check
2. Validation harness - stratified 5-fold CV, shared across every model
3. Model benchmarking - CatBoost, LightGBM, XGBoost on identical raw
   features
4. Feature engineering - hypotheses from the audit, tested against frozen
   controls, accepted or rejected on CV evidence
5. Synthetic-generator investigation - forensic audit of what's real
   generator structure versus ordinary signal versus noise
6. Hyperparameter tuning, ensembling, and final submission strategy -
   complete (see Build 9 above); competition result pending close

## Data source

Competition data - 691,369 training rows, 296,302 test rows, all fields
synthetically generated. It's not included in this repository:
`data/*.csv` is gitignored per the competition's terms, and
`data/README.md` explains how to pull it down yourself.

Every predictor column has some missingness (4-19% depending on the
column), and the target is moderately imbalanced (71% positive). Both are
handled explicitly in the preprocessing rather than assumed away.

---

## Results / outcomes

| Experiment | Model | Feature set | CV mean ROC AUC | Public LB |
|---|---|---|---|---|
| E001 | Logistic Regression | raw predictors | 0.91149 (std 0.00081) | 0.91358 |
| E002 | CatBoost | raw predictors | 0.96040 (std 0.00051) | 0.96151 |
| E004 | XGBoost | raw predictors | 0.96382 (std 0.00056) | 0.96539 |
| E006 | XGBoost | raw + `screen_residual` | 0.96445 (std 0.00056) | 0.96608 |
| **E010** | **XGBoost (tuned)** | **raw + `screen_residual`** | **0.96499 (std 0.00051)** | **0.96653** |

E010 is the current best model: same feature set and architecture as
E006, with `learning_rate` lowered to 0.05 and `n_estimators` raised to
2500 after E006 was found to be iteration-constrained (hitting or nearly
hitting its 800-iteration cap in 4/5 folds). `screen_residual` - the gap
between `daily_screen_time_hours`
and the sum of its three named components - was found during EDA,
tested in isolation against both XGBoost and CatBoost, and only accepted
once it showed a consistent gain on both (+0.00062 and +0.00064
respectively, 5/5 folds improved each time). A second candidate feature
built from the same relationship (`component_sum`) was tested and
rejected as redundant once `screen_residual` was already in the model -
that negative result is recorded too, not just the win.

Every number above traces to a committed artifact
(`experiments/experiments.csv`, `outputs/model_benchmarks.csv`,
`outputs/feature_experiments.csv`), not a remembered figure.

**Synthetic-generator investigation (Build 5):** ran a full forensic audit
of the dataset's generation process - numeric precision grids, the
screen-time compositional constraint behind `screen_residual`, exact-value
target rates, repeated and near-duplicate profiles, missingness patterns,
ID drift, and cross-feature constraints. Found one previously-uncharted
generator artifact (`sleep_hours + daily_screen_time_hours` clips near
20h) and confirmed it's fully redundant with a feature already in the
model. Every other candidate - missingness encoding, near-duplicate
lookup, frequency encoding - was tested and rejected on evidence. No new
feature came out of it, and that's recorded as a real finding
(`outputs/synthetic_generator_findings.csv`), not a gap in the work.

**Controlled XGBoost tuning (Build 6):** with the feature set frozen,
tuned E006's XGBoost hyperparameters under a fixed CV harness. Diagnosed
E006 as iteration-constrained rather than converged (`best_iteration` at
or one below its 800-iteration cap in 4/5 folds), then confirmed that
lowering the learning rate and raising the iteration ceiling (E010:
`learning_rate=0.05`, `n_estimators=2500`) let early stopping find a true
optimum - CV mean 0.96499 vs 0.96445, 5/5 folds improved, no fold hit the
new ceiling. Nine further candidates (tree depth, sampling, regularization)
were screened on a single fold each and rejected as noise before reaching
a full CV run, per the build's stopping rule. E010 became the new frozen
control (`outputs/xgboost_tuning_results.csv`,
`outputs/best_xgboost_params.json`).

**Ensembling investigation (Build 7):** tested whether combining E010
with CatBoost (`screen_residual`), a pre-tuned XGBoost, and LightGBM
(raw features) could beat E010 alone. CatBoost and LightGBM are each
genuinely diverse from E010 - correlated overall (~0.985 Pearson) but
disagreeing on 28-33% of the highest-probability decile, the region that
matters most for ROC AUC - yet individually about 0.004 AUC below it, a
gap too large for that diversity to net a gain once weighted to avoid
dragging the blend down. Equal-weight, weighted-grid, and rank-averaged
blends were all tested; none reached a meaningful improvement, so a
three-model blend and stacking were both explicitly skipped rather than
run anyway. E010 remains the best model - a negative result, evidenced
end to end (`outputs/ensemble_prediction_correlations.csv`,
`outputs/ensemble_results.csv`) rather than assumed.

**Final submission strategy (Build 9):** verified the live competition's
final-submission constraints (up to 2 final submissions selectable
before its posted close date), confirmed E010 as the primary final
submission, and selected E008
(CatBoost + `screen_residual`) as a second final submission - a
deliberate model-family diversity hedge (Build 7: 0.985 Pearson vs E010,
28% top-decile disagreement) against the single-model-family risk Build
8 flagged, not a leaderboard-driven choice. Best Public LB remains
**0.96653** (E010); E008's public LB is pending. Competition final
result (private LB, rank, percentile): **pending competition close.**

**CV vs leaderboard reconciliation (Build 8):** checked whether local
cross-validation has actually been trustworthy across every submitted
experiment (E001, E002, E004, E006, E010), rather than assuming it. CV
and public leaderboard rank every submission identically (Spearman =
Kendall = 1.0, no inversions), and every CV improvement was confirmed by
a leaderboard improvement in the same direction, including E010's small
final gain over E006. No submission or parameter choice in this
project's history was ever driven by a leaderboard score rather than CV
- confirmed from the actual decision history, not asserted
(`outputs/cv_lb_reconciliation.csv`). The one caveat: the entire winning
lineage is a single model family (XGBoost), since Build 7 found no
ensemble beats it - a real, disclosed private-leaderboard risk factor,
not a reason to distrust the CV-driven process itself.

## How to run the project

```bash
git clone <repo-url>
cd kaggle-smartphone-addiction
python -m venv <env-name>
<env-name>\Scripts\activate        # Windows
pip install -r requirements-dev.txt
```

Data isn't included - see `data/README.md` to get the three competition
CSVs into `data/` first. Then:

```bash
pytest tests/
jupyter notebook notebooks/
```

There's no dashboard or app here - the notebooks are the deliverable.
Start with `notebooks/01_eda.ipynb` for the data audit, or
`notebooks/05_synthetic_generator.ipynb` for the generator investigation.

## Repository structure

```text
src/            reusable, accepted logic (imported by notebooks and tests)
notebooks/      experimentation laboratory, one per build
tests/          smoke checks and unit tests
data/           competition CSVs (gitignored; see data/README.md)
experiments/    experiment tracker (experiments.csv) and per-run artifacts
deliverables/   submission-ready CSVs (see deliverables/CONTENTS.md)
docs/           decision log, build history
outputs/        generated tables and artifacts
```

## What I learned

- How much of "feature engineering" is really about characterizing a
  relationship precisely enough to know why it works, not just
  discovering that it does
- That a synthetic dataset needs its own kind of audit - the questions
  that matter for real-world data (is this leakage? is this bias?) get
  replaced by generator-specific ones (is this an artifact of how the
  data was made, and if so, is it safe and worthwhile to use?)
- How much discipline a frozen control adds: every feature candidate this
  project has tested was compared against the same fixed baseline under
  the same CV split, which makes "did this actually help" a much less
  slippery question to answer
- That a negative result (a rejected feature, a null finding from a whole
  build) is worth documenting as carefully as a positive one - the
  decision log here has as many rejections as acceptances

## Future improvements

- Once the competition closes: record the private leaderboard result,
  final rank, and percentile, and consolidate the project (Build C -
  not started)
- The README's data-source and setup sections could use a cold-clone test
  (fresh venv, fresh clone, verbatim commands) before the project is
  called complete

## Contact

[github.com/sanjay-dilip](https://github.com/sanjay-dilip)
