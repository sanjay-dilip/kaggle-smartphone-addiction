# Decision Log

Records why, not what happened chronologically. See `BUILD_HISTORY.md` for
the chronological record.

## This is a competition project, not a deployment project

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

`data/*.csv` is gitignored. The competition data is licensed for
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

Build 0 originally created a `submissions/` directory for
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

## E010 (tuned XGBoost) supersedes E006 as the primary control; joint refinement skipped

Build 6 (`notebooks/06_xgboost_tuning.ipynb`) found E006 was materially
constrained by its 800-iteration training cap (`best_iterations`
`[799, 794, 799, 794, 794]` — 4/5 folds at or one below the cap, not
evidence of convergence). Lowering `learning_rate` to 0.05 and raising
`n_estimators` to 2500 (same feature set, same `early_stopping_rounds`)
let early stopping converge genuinely (best_iterations 2294-2485, no
fold hit the ceiling) and produced **E010**: CV mean 0.96499 (std
0.00051) vs E006's 0.96445 — +0.00055, 5/5 folds improved, range
[+0.00042, +0.00062]. **E010 is accepted as the new primary XGBoost
control, superseding E006.**

Subsequent staged tuning around E010 — tree complexity (`max_depth`,
`min_child_weight`), sampling (`subsample`, `colsample_bytree`), and
regularization (`reg_alpha`, `reg_lambda`) — tested 9 single-fold
candidates across three phases and found nothing exceeding the ~0.0005
single-fold noise band observed throughout the build (best candidate
deltas: +0.00007, -0.00001, +0.00002 respectively). Per the Build 6
stopping rule (several consecutive sensible configurations failing to
improve the current best), **joint refinement was skipped** rather than
running a full 5-fold CV to reconfirm a null result across all three
parameter families at once — E010 is adopted directly as Build 6's
final tuned configuration.

E010's out-of-fold predictions correlate with E006's at Pearson 0.9967
— expected, since E010 refines E006's exact architecture rather than
introducing a diverse alternative; this is a diagnostic only and does
not itself motivate any Build 7 ensembling decision.

Evidence trace: `outputs/xgboost_tuning_results.csv`,
`outputs/best_xgboost_params.json`, `experiments/experiments.csv`
(E010 row).

## Build 7 — Ensembling / Blending

**Candidate pool: E010, E008, E006, E003 (4 active).** E001, E002, E005,
E007, E009 excluded as dominated (same model with a strictly worse
feature set, or a rejected feature, or too weak individually — see
`docs/BUILD_HISTORY.md`'s Build 7 entry for the full inventory).

**E006 retained in the pool despite suspected redundancy, tested anyway.**
Its 0.9967 Pearson correlation with E010 (measured in Build 6) predicted
minimal blend value; Phase 3's fuller diversity analysis confirmed it —
lowest mean abs diff (0.0151) and lowest top/bottom-decile disagreement
(0.104/0.056) of all six pairs measured, by a wide margin. Blending it
with E010 answers the brief's explicit question ("does the pre-tuned
XGBoost add anything despite being same-family") with a clear no: best
blend point +0.00004 vs E010, inside the ~0.0005 noise band.

**CatBoost (E008) and LightGBM (E003) both confirmed as genuinely
diverse relative to E010** (~0.985 Pearson, but 28-33% top-decile
disagreement — meaningful disagreement precisely where ROC AUC is most
sensitive) **and diverse from each other** (42.5% top-decile
disagreement between E008 and E003, the highest of any pair measured).
This diversity is real, not an artifact of one weak, low-correlation
outlier — both are individually strong (CV ~0.961, ~0.004 below E010).

**Probability blending chosen as the primary method; rank averaging
tested and found equivalent, not superior.** At every weight tested,
rank-averaged blends scored within 0.00001 of the corresponding
probability blend. No calibration difference between these four models
is large enough for rank transformation to matter.

**No ensemble beats E010 — this is Build 7's core finding.** Equal-weight
blends of E010 with either diverse candidate are mechanically Worse
(averaging in a ~0.004-weaker model pulls the mean down); the weighted
grid (E010 weight 0.5-0.95) shows both E010+E008 and E010+E003
monotonically approaching, but never exceeding, E010's own OOF AUC as
E010's weight rises. E008+E003 (no E010 component) never comes close to
E010 at any weight. **Rejected: any blend as the Build 7 candidate** —
none clears the Core Build 7 Rule's "measurable CV improvement" bar.

**Three-model blend (E010+E008+E003): rejected, not attempted.** The
brief's own gate requires both constituent pairs to show real two-model
gains first; neither did.

**Stacking: rejected as unnecessary.** The weighted grid already searched
the full linear combination space across every pair and found no region
above E010. A `LogisticRegression` meta-model over the same probability
inputs is itself a linear combination of them — it cannot find a
solution that dense grid search over that exact space did not already
rule out. Implementing leakage-safe nested-CV stacking machinery to
re-confirm this would add complexity without evidentiary value.

**E010 remains the frozen best model entering Build 8**, unchanged by
Build 7. Its public LB (0.96653) stands as the reference until Build 8
examines CV/LB reconciliation directly.

Evidence trace: `outputs/ensemble_prediction_correlations.csv`,
`outputs/ensemble_results.csv`, `outputs/oof_predictions/`,
`outputs/test_predictions/`.

## Build 8 — CV vs Leaderboard Reconciliation

**E010 retained as the primary Build 9 candidate** because CV and
public LB both confirm its improvement over E006: 5/5 folds improved
(tight range +0.00042 to +0.00062), CV std improved (0.00056 -> 0.00051),
and public LB moved in the same direction (+0.00045, transfer ratio
0.83). Across the full submission history (E001, E002, E004, E006,
E010), CV and public LB rank experiments identically (Spearman = Kendall
= 1.0) and every CV improvement was confirmed by an LB improvement in
the same direction — no rank inversions, no divergent-direction
transitions, anywhere.

**Public LB selection-bias risk assessed as Low** because every
submitted experiment's model/parameter/feature choice traces to a CV
comparison finalized *before* its corresponding submission
(audited per-experiment in `notebooks/08_cv_leaderboard_reconciliation.ipynb`
Section 8): no parameter was ever changed in response to an LB score, no
CV-rejected model was ever resurrected because of LB, and Build 7's
ensembling investigation made zero submissions specifically to avoid
LB-driven weight tuning.

**Private LB robustness risk assessed as Moderate, not Low**, despite
the low selection-bias finding, because two structural risk factors
remain regardless of process hygiene: the final tuning gain (E006 ->
E010) is small in absolute terms (+0.00054 CV), and the entire winning
lineage (E004, E006, E010) is a single model family (XGBoost) with zero
cross-family diversification — not from oversight, but because Build 7
found no ensemble combining CatBoost/LightGBM beats E010. This is a risk
characterization, not a private-score prediction.

**E008 (CatBoost) flagged as a possible Build 9 hedge candidate, not
confirmed.** It is the only historical model with genuine prediction
diversity from E010 (Build 7: 0.985 Pearson, 28% top-decile
disagreement) and reasonable individual CV (0.96104), which is exactly
what a private-LB diversification hedge would want. However it was
never submitted and so carries zero public LB evidence, unlike
every other candidate in this reconciliation — Build 9 should decide
whether that gap is worth closing with a submission, not Build 8, which
is barred from submitting without explicit approval.

**E006 rejected as a hedge candidate** despite having public LB
evidence, because Build 7 already established it is too redundant with
E010 (0.9967 Pearson, lowest disagreement of every pair measured) to
offer any private-LB diversification benefit.

**No new submissions were made this build.** All required
evidence already existed in committed artifacts; Build 8's philosophy is
reconciliation, not new experimentation.

Evidence trace: `outputs/cv_lb_reconciliation.csv`,
`outputs/model_progression.csv`, `outputs/figures/cv_vs_public_lb.png`.

## Build 9 -- Final Submission Strategy

**Live competition constraints verified** at the start of this build:
the competition remained open with two final submissions selectable for
judging, ahead of its posted close date.

**E010 confirmed as the primary final submission**, unchanged from
Build 8. Integrity audit this build: the committed deliverable
(`deliverables/submission_E010_xgb_tuned.csv`) matches the persisted,
`cv_mean`-verified `outputs/test_predictions/E010.csv` artifact to
within 5e-9 (floating-point rounding only) -- no drift, no dependency on
uncommitted local code. No reproducibility or integrity defect found;
per the build's own rule, this does not reopen tuning.

**E008 selected as the second final submission (hedge), decided before
observing its public LB.** Rationale, evaluated against the Build 9
hedge-decision framework:

- *Strength*: CV 0.96104 vs E010's 0.96499 -- a real but modest gap
  (~0.004 AUC), the same magnitude `screen_residual` closed on CatBoost
  relative to XGBoost in Build 4 (E008 vs E006's own delta over their
  respective raw-feature controls was near-identical, +0.00064 vs
  +0.00062).
- *Stability*: CV std 0.00055, in the same tight band as every other
  boosting-model experiment in this project (0.00051-0.00056) -- not a
  high-variance outlier.
- *Diversity* (Build 7, `outputs/ensemble_prediction_correlations.csv`):
  the only historical candidate with genuine prediction diversity from
  E010 -- 0.9849 Pearson (vs E006's 0.9967), 28.1% top-decile
  disagreement (vs E006's 10.4%). This is exactly the profile a
  private-LB diversification hedge wants: correlated enough to be a
  sensible model, different enough that its errors are not E010's
  errors.
- *Model-family independence*: CatBoost vs E010's XGBoost -- addresses
  Build 8's flagged risk factor directly (the entire winning lineage,
  E004/E006/E010, is single-family XGBoost with zero cross-family
  diversification).
- *Feature dependence*: identical accepted feature set (raw predictors +
  `screen_residual`) -- the hedge diversifies model family only, not
  feature representation, which keeps the comparison to E010 clean.
- *Existing LB evidence*: none prior to this build -- E008 was validated
  locally only in Build 4 (`experiments/experiments.csv` row) and never
  submitted. This is the one open question in the framework Build 9 was
  specifically tasked with resolving.

E006 was **not** reconsidered as a hedge -- Build 7 already established
it is too redundant with E010 (0.9967 Pearson, lowest disagreement of
any pair measured in that build) to offer any diversification benefit,
and nothing in Build 8 or Build 9 changes that finding.

**E008 submitted exactly as already validated -- no parameter, feature,
seed, categorical handling, fold setup, iteration budget, or prediction
method was changed.** The submission file
(`deliverables/submission_E008_catboost_screen_residual.csv`) was
generated by loading the existing, `cv_mean`-verified
`outputs/test_predictions/E008.csv` artifact (Build 7 regeneration,
CV mean 0.96104 reconfirmed exactly before being trusted) via
`src.ensembling.load_test_pred`, aligning to
`data/sample_submission.csv`'s id order, and validating with
`src.submission_validation.validate_submission` -- no retraining
performed. This decision (submit E008, unchanged) was made and recorded
in this file *before* E008's public LB score was returned, per the
build's own no-leaderboard-chasing rule.

**E008's public LB (0.96220), once reported, reconciled cleanly with
its CV.** LB-CV gap +0.00116, inside the Build 8 boosting-model gap
range [0.00111, 0.00163] -- no rank inversion, no model-family-specific
anomaly. This is evidence collection only; it changes nothing about the
selection decision above, which was already made and recorded before
this score existed. The user selected both E010 and E008 as the two
final submissions for judging.

**No new modeling permitted after this build's final candidate decision.**
Build 9's role is selection under constraints, not development; any
further XGBoost/CatBoost/LightGBM tuning, feature engineering, or
ensembling is explicitly out of scope until the competition closes and
Build C's consolidation begins.

Evidence trace: `outputs/final_submission_candidates.csv`,
`outputs/ensemble_prediction_correlations.csv` (Build 7).
