"""Regenerates and persists OOF/test predictions for Build 7's candidate pool.

No experiment in this repository has ever saved a raw per-row OOF or test
prediction array to disk (see docs/BUILD_HISTORY.md's Build 7 entry) --
only scalar correlation values survived past their originating notebook
run. This script reconstructs each candidate's *exact* frozen
configuration (never retuned), verifies the reconstruction against its
recorded `cv_mean` in experiments/experiments.csv, and persists OOF/test
predictions via src.ensembling so Build 7's blending work never needs to
retrain a base model again.

Run once, in the background (E008's CatBoost fold and E010's raised
iteration ceiling are both slow on CPU): expect roughly 90-110 minutes
total across all four candidates.
"""

import time

import pandas as pd

from src.benchmarking import run_cv_benchmark
from src.boosting_models import catboost_fold, lightgbm_fold
from src.config import (
    EXPERIMENTS_DIR,
    ID_COLUMN,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from src.ensembling import save_fold_assignments, save_oof, save_test_pred
from src.features import add_screen_residual
from src.preprocessing import build_boosting_frame
from src.tuning import E006_XGB_PARAMS, run_xgboost_trial
from src.validation import get_cv_splitter

CV_MEAN_TOLERANCE = 2e-4


def _recorded_cv_mean(experiment_id: str) -> float:
    experiments = pd.read_csv(EXPERIMENTS_DIR / "experiments.csv")
    row = experiments.loc[experiments["experiment_id"] == experiment_id].iloc[0]
    return float(row["cv_mean"])


def _assert_matches_recorded(experiment_id: str, cv_mean: float) -> None:
    recorded = _recorded_cv_mean(experiment_id)
    delta = abs(cv_mean - recorded)
    assert delta < CV_MEAN_TOLERANCE, (
        f"{experiment_id} reconstruction cv_mean {cv_mean:.5f} does not match "
        f"recorded {recorded:.5f} (delta {delta:.5f} >= tolerance {CV_MEAN_TOLERANCE})"
    )
    print(f"{experiment_id}: reconstructed cv_mean {cv_mean:.5f} matches recorded {recorded:.5f}")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    y = train[TARGET_COLUMN]
    train_ids = train[ID_COLUMN]
    test_ids = test[ID_COLUMN]
    assert (test_ids.to_numpy() == sample_submission[ID_COLUMN].to_numpy()).all(), (
        "test.csv id order does not match sample_submission.csv id order"
    )

    # Canonical fold assignment: StratifiedKFold(seed=42) depends only on
    # row count and y, so it is identical for every candidate regardless
    # of feature set.
    splitter = get_cv_splitter()
    fold_of_row = pd.Series(0, index=train.index)
    for fold_idx, (_, val_idx) in enumerate(splitter.split(train, y), start=1):
        fold_of_row.iloc[val_idx] = fold_idx
    save_fold_assignments(train_ids, fold_of_row.to_numpy())
    print(f"Saved canonical fold assignments for {len(train_ids)} rows.")

    X_raw = build_boosting_frame(train)
    X_test_raw = build_boosting_frame(test)
    X_residual = add_screen_residual(X_raw).drop(columns=["component_sum"])
    X_test_residual = add_screen_residual(X_test_raw).drop(columns=["component_sum"])

    # --- E003: LightGBM, raw predictors only ---
    print("\n=== E003 (LightGBM, raw) ===")
    t0 = time.time()
    e003 = run_cv_benchmark(lightgbm_fold, X_raw, y, X_test=X_test_raw)
    print(f"elapsed: {time.time() - t0:.1f}s")
    _assert_matches_recorded("E003", e003.cv_mean)
    save_oof("E003", train_ids, e003.oof_predictions)
    save_test_pred("E003", test_ids, e003.test_predictions)

    # --- E006: XGBoost pre-tuned, raw + screen_residual ---
    print("\n=== E006 (XGBoost pre-tuned, raw + screen_residual) ===")
    t0 = time.time()
    e006 = run_xgboost_trial(E006_XGB_PARAMS, X_residual, y, X_test=X_test_residual)
    print(f"elapsed: {time.time() - t0:.1f}s")
    _assert_matches_recorded("E006", e006.cv_mean)
    save_oof("E006", train_ids, e006.oof_predictions)
    save_test_pred("E006", test_ids, e006.test_predictions)

    # --- E010: XGBoost tuned, raw + screen_residual (frozen Build 6 control) ---
    print("\n=== E010 (XGBoost tuned, raw + screen_residual) ===")
    E010_PARAMS = {**E006_XGB_PARAMS, "learning_rate": 0.05, "n_estimators": 2500}
    t0 = time.time()
    e010 = run_xgboost_trial(E010_PARAMS, X_residual, y, X_test=X_test_residual)
    print(f"elapsed: {time.time() - t0:.1f}s")
    _assert_matches_recorded("E010", e010.cv_mean)
    save_oof("E010", train_ids, e010.oof_predictions)
    save_test_pred("E010", test_ids, e010.test_predictions)

    # --- E008: CatBoost, raw + screen_residual ---
    print("\n=== E008 (CatBoost, raw + screen_residual) ===")
    t0 = time.time()
    e008 = run_cv_benchmark(catboost_fold, X_residual, y, X_test=X_test_residual)
    print(f"elapsed: {time.time() - t0:.1f}s")
    _assert_matches_recorded("E008", e008.cv_mean)
    save_oof("E008", train_ids, e008.oof_predictions)
    save_test_pred("E008", test_ids, e008.test_predictions)

    print("\nAll four candidates regenerated and persisted successfully.")


if __name__ == "__main__":
    main()
