"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLS, ID_COLUMN, NUMERIC_COLS, TARGET_COLUMN
from src.preprocessing import build_preprocessor


def _sample_frame(n_rows: int = 20) -> pd.DataFrame:
    """Builds a small synthetic frame with the real predictor schema."""
    rng = np.random.default_rng(seed=0)
    data = {ID_COLUMN: np.arange(n_rows), TARGET_COLUMN: rng.integers(0, 2, n_rows)}
    for col in NUMERIC_COLS:
        values = rng.uniform(0, 10, n_rows)
        values[::5] = np.nan
        data[col] = values
    for col in CATEGORICAL_COLS:
        values = rng.choice(["A", "B", "C"], n_rows).astype(object)
        values[::5] = np.nan
        data[col] = values
    return pd.DataFrame(data)


def test_preprocessor_output_has_no_missing_values() -> None:
    """Imputation must eliminate all NaNs from the transformed matrix."""
    frame = _sample_frame()
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(frame[NUMERIC_COLS + CATEGORICAL_COLS])

    assert not np.isnan(transformed).any()


def test_preprocessor_excludes_id_and_target_columns() -> None:
    """The ColumnTransformer must only reference NUMERIC_COLS/CATEGORICAL_COLS."""
    preprocessor = build_preprocessor()

    referenced_cols = {
        col
        for _, _, cols in preprocessor.transformers
        for col in cols
    }

    assert ID_COLUMN not in referenced_cols
    assert TARGET_COLUMN not in referenced_cols
    assert referenced_cols == set(NUMERIC_COLS) | set(CATEGORICAL_COLS)


def test_preprocessor_output_row_count_matches_input() -> None:
    """Transforming must not drop or duplicate rows."""
    frame = _sample_frame(n_rows=15)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(frame[NUMERIC_COLS + CATEGORICAL_COLS])

    assert transformed.shape[0] == 15
