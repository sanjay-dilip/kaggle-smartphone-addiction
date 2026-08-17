"""Preprocessing pipeline for the Logistic Regression baseline.

Decisions here trace to docs/DECISIONS.md, frozen from the Build 1 data
audit: median imputation for numeric predictors, an explicit "Missing"
category for categoricals (not mode imputation), one-hot encoding, and
`id` excluded from the feature matrix.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """Builds the numeric/categorical preprocessing pipeline.

    Numeric predictors: median imputation, then standardization.
    Categorical predictors: missing values filled with an explicit
    "Missing" category, then one-hot encoding.

    Only NUMERIC_COLS and CATEGORICAL_COLS are referenced, so `id` and the
    target column are excluded by construction.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
