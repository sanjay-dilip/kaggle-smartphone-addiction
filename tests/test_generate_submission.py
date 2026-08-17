"""Tests for src/generate_submission.py."""

import pandas as pd

from src.config import ID_COLUMN, SAMPLE_SUBMISSION_PATH, TARGET_COLUMN
from src.generate_submission import generate_e001_submission


def test_e001_submission_matches_sample_submission_schema() -> None:
    """The generated submission must match sample_submission.csv exactly:
    same columns, same row count, same id values in the same order, and
    probabilities in [0, 1].
    """
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    submission = generate_e001_submission()

    assert list(submission.columns) == list(sample_submission.columns)
    assert len(submission) == len(sample_submission)
    assert (submission[ID_COLUMN].values == sample_submission[ID_COLUMN].values).all()
    assert submission[TARGET_COLUMN].between(0, 1).all()
    assert submission[TARGET_COLUMN].notna().all()
