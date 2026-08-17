"""Generates a Kaggle-submission-ready CSV from a validated experiment.

Currently supports E001 (Logistic Regression baseline, see
experiments/experiments.csv and notebooks/02_baseline.ipynb). The model is
refit on the full training set — cross-validation in the baseline notebook
is for scoring only, not for producing the final fitted model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_COLS,
    DELIVERABLES_DIR,
    ID_COLUMN,
    NUMERIC_COLS,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from src.preprocessing import build_preprocessor

FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS


def build_e001_pipeline() -> Pipeline:
    """Builds the E001 pipeline: Build 1/2 preprocessing + Logistic Regression."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def generate_e001_submission() -> pd.DataFrame:
    """Fits E001 on the full training set and predicts on test.csv.

    Returns a DataFrame matching data/sample_submission.csv's schema
    exactly: `id` (int) and `addicted_label` (predicted probability).
    """
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    pipeline = build_e001_pipeline()
    pipeline.fit(train[FEATURE_COLS], train[TARGET_COLUMN])

    predicted_probabilities = pipeline.predict_proba(test[FEATURE_COLS])[:, 1]

    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN],
            TARGET_COLUMN: predicted_probabilities,
        }
    )

    assert list(submission.columns) == list(sample_submission.columns)
    assert len(submission) == len(sample_submission)
    assert (submission[ID_COLUMN].values == sample_submission[ID_COLUMN].values).all()

    return submission


if __name__ == "__main__":
    DELIVERABLES_DIR.mkdir(exist_ok=True)
    output_path = DELIVERABLES_DIR / "E001_submission.csv"

    submission = generate_e001_submission()
    submission.to_csv(output_path, index=False)

    print(f"Wrote {len(submission)} rows to {output_path}")
