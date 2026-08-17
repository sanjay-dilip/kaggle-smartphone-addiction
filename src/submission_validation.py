"""Programmatic checks for Kaggle-submission-ready CSVs.

Raises AssertionError with a specific message on the first failed check,
rather than silently writing a malformed submission.
"""

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    id_col: str = ID_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> None:
    """Validates `submission` against `sample_submission`'s schema.

    Checks: matching columns, matching row count, no duplicate/missing/
    reordered ids, no missing or non-finite predictions, and predictions
    within [0, 1].
    """
    assert list(submission.columns) == list(sample_submission.columns), (
        f"column mismatch: {list(submission.columns)} != "
        f"{list(sample_submission.columns)}"
    )
    assert len(submission) == len(sample_submission), (
        f"row count mismatch: {len(submission)} != {len(sample_submission)}"
    )
    assert not submission[id_col].duplicated().any(), "submission contains duplicate ids"
    assert (submission[id_col].values == sample_submission[id_col].values).all(), (
        "submission ids do not exactly match sample_submission ids/order"
    )
    assert submission[target_col].notna().all(), "submission contains missing predictions"
    assert np.isfinite(submission[target_col].to_numpy(dtype=float)).all(), (
        "submission contains non-finite predictions"
    )
    assert submission[target_col].between(0, 1).all(), (
        "submission contains probabilities outside [0, 1]"
    )
