# Competition Close Checklist

Pre-close checklist for Kaggle Playground Series S6E8 (Predicting
Smartphone Addiction). This is a human action list, not Build C
consolidation -- see `docs/BUILD_HISTORY.md`'s Build 9 entry for the
evidence behind each selection.

**Final Submission Deadline: August 31, 2026, 11:59 PM UTC.**

## Final candidates

- [ ] **Primary** -- E010 (XGBoost tuned + `screen_residual`), file
      `deliverables/submission_E010_xgb_tuned.csv`, public LB 0.96653.
- [ ] **Hedge** -- E008 (CatBoost + `screen_residual`), file
      `deliverables/submission_E008_catboost_screen_residual.csv`, public
      LB pending (submitted this build, score not yet recorded).

## Submission and selection

- [ ] Both files uploaded to Kaggle via Submissions page (E010 may
      already have been uploaded in Build 6 -- check submission history
      before re-uploading).
- [ ] E008 public LB score recorded in `experiments/experiments.csv`
      (E008 row) and reconciled in `docs/BUILD_HISTORY.md`'s Build 9
      entry once available.
- [ ] On Kaggle's Submissions page, mark exactly these two submissions
      (E010 and E008) as the two Final Submissions for judging, before
      the deadline.
- [ ] Cross-check: the two marked submissions' displayed public LB
      scores match 0.96653 (E010) and the recorded E008 score exactly --
      confirms the correct files were selected, not a stale or
      accidental upload.

## Verification

- [ ] Deadline re-confirmed on the live competition page close to the
      actual close date (organizers reserve the right to update the
      timeline).
- [ ] `experiments/experiments.csv` is current (all actually-submitted
      experiments present, no fabricated rows).
- [ ] `README.md` contains no premature final-result claim (final rank,
      private LB, percentile, medal/status) -- only Public LB values,
      explicitly labeled as such.

## After competition close

- [ ] Capture a screenshot of the final Private Leaderboard placement.
- [ ] Record final rank, private LB score, and percentile once Kaggle
      publishes them (`CONTEXT.md` and `README.md`).
- [ ] Confirm which of the two final submissions (E010 or E008, or
      neither) scored best on the private leaderboard -- this is the
      actual test of Build 9's hedge decision.

## Not yet done (explicitly out of scope for Build 9)

- [ ] Final rank / percentile / private LB -- **pending competition
      close**, not estimated.
- [ ] Build C (cold-clone consolidation, repository-wide cleanup, final
      README rewrite, v1.0 tagging, full claim-trace audit) -- **not
      started**.
