"""Reusable prediction-blending and diversity-analysis helpers for Build 7.

Operates only on already-computed out-of-fold (OOF) and test predictions
-- no model construction or training lives here (that stays in
src/boosting_models.py and src/tuning.py). Every prediction artifact this
module reads is validated for row count, id alignment, and probability
bounds before use, per the Build 7 rule: never blend predictions by
positional assumption alone when id alignment can be verified explicitly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import ID_COLUMN, OUTPUTS_DIR

OOF_DIR: Path = OUTPUTS_DIR / "oof_predictions"
TEST_PRED_DIR: Path = OUTPUTS_DIR / "test_predictions"
FOLD_ASSIGNMENTS_PATH: Path = OUTPUTS_DIR / "cv_fold_assignments.csv"


def validate_prediction_artifact(
    df: pd.DataFrame, expected_ids: pd.Series, proba_col: str
) -> None:
    """Validates a loaded OOF/test prediction artifact.

    Checks required columns, row count, no duplicate ids, exact id-set
    match against `expected_ids`, no missing/non-finite predictions, and
    predictions within [0, 1].
    """
    assert ID_COLUMN in df.columns and proba_col in df.columns, (
        f"missing required columns: need {ID_COLUMN!r} and {proba_col!r}, "
        f"got {list(df.columns)}"
    )
    assert len(df) == len(expected_ids), (
        f"row count mismatch: {len(df)} != {len(expected_ids)}"
    )
    assert not df[ID_COLUMN].duplicated().any(), "duplicate ids in prediction artifact"
    assert set(df[ID_COLUMN]) == set(expected_ids), (
        "prediction artifact ids do not match the expected id set"
    )
    assert df[proba_col].notna().all(), "prediction artifact contains missing values"
    assert np.isfinite(df[proba_col].to_numpy(dtype=float)).all(), (
        "prediction artifact contains non-finite values"
    )
    assert df[proba_col].between(0, 1).all(), (
        "prediction artifact contains values outside [0, 1]"
    )


def save_oof(experiment_id: str, ids: pd.Series, oof_proba: np.ndarray) -> Path:
    """Persists one experiment's OOF predictions to outputs/oof_predictions/{id}.csv."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID_COLUMN: ids.to_numpy(), "oof_proba": oof_proba})
    validate_prediction_artifact(frame, ids, "oof_proba")
    path = OOF_DIR / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return path


def load_oof(experiment_id: str, expected_ids: pd.Series) -> pd.Series:
    """Loads one experiment's OOF predictions, validated and aligned to `expected_ids` order."""
    frame = pd.read_csv(OOF_DIR / f"{experiment_id}.csv")
    validate_prediction_artifact(frame, expected_ids, "oof_proba")
    aligned = frame.set_index(ID_COLUMN).loc[expected_ids, "oof_proba"]
    return aligned.reset_index(drop=True)


def save_test_pred(experiment_id: str, ids: pd.Series, test_proba: np.ndarray) -> Path:
    """Persists one experiment's test predictions to outputs/test_predictions/{id}.csv."""
    TEST_PRED_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID_COLUMN: ids.to_numpy(), "test_proba": test_proba})
    validate_prediction_artifact(frame, ids, "test_proba")
    path = TEST_PRED_DIR / f"{experiment_id}.csv"
    frame.to_csv(path, index=False)
    return path


def load_test_pred(experiment_id: str, expected_ids: pd.Series) -> pd.Series:
    """Loads one experiment's test predictions, validated and aligned to `expected_ids` order."""
    frame = pd.read_csv(TEST_PRED_DIR / f"{experiment_id}.csv")
    validate_prediction_artifact(frame, expected_ids, "test_proba")
    aligned = frame.set_index(ID_COLUMN).loc[expected_ids, "test_proba"]
    return aligned.reset_index(drop=True)


def save_fold_assignments(ids: pd.Series, folds: np.ndarray) -> Path:
    """Persists the canonical id -> fold mapping shared by every candidate model."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID_COLUMN: ids.to_numpy(), "fold": folds})
    frame.to_csv(FOLD_ASSIGNMENTS_PATH, index=False)
    return FOLD_ASSIGNMENTS_PATH


def load_fold_assignments(expected_ids: pd.Series) -> pd.Series:
    """Loads the canonical id -> fold mapping, validated and aligned to `expected_ids` order."""
    frame = pd.read_csv(FOLD_ASSIGNMENTS_PATH)
    assert len(frame) == len(expected_ids), "fold assignment row count mismatch"
    assert set(frame[ID_COLUMN]) == set(expected_ids), (
        "fold assignment ids do not match the expected id set"
    )
    aligned = frame.set_index(ID_COLUMN).loc[expected_ids, "fold"]
    return aligned.reset_index(drop=True)


def weighted_blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of aligned prediction arrays.

    Weights need not be pre-normalized (they are divided by their sum)
    but must be positive and share exactly the same key set as `preds`.
    Fails loudly on mismatched keys, mismatched array lengths, or NaNs.
    """
    if set(preds) != set(weights):
        raise ValueError(
            f"preds and weights must share the same keys: {set(preds)} != {set(weights)}"
        )
    lengths = {len(v) for v in preds.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"prediction arrays have mismatched lengths: "
            f"{ {k: len(v) for k, v in preds.items()} }"
        )
    for name, arr in preds.items():
        if np.isnan(np.asarray(arr, dtype=float)).any():
            raise ValueError(f"prediction array {name!r} contains NaN values")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive total")

    blend = np.zeros(next(iter(lengths)))
    for name, arr in preds.items():
        blend += (weights[name] / total_weight) * np.asarray(arr, dtype=float)
    return blend


def rank_blend(
    preds: dict[str, np.ndarray], weights: dict[str, float] | None = None
) -> np.ndarray:
    """Weighted average of percentile ranks.

    Percentile rank uses pandas' `rank(pct=True, method="average")`:
    deterministic, ties share the mean rank of their tied group. OOF and
    test sets must each be ranked independently within their own set --
    callers should never concatenate OOF and test predictions before
    calling this function.
    """
    if weights is None:
        weights = {name: 1.0 for name in preds}
    ranked = {
        name: pd.Series(arr).rank(pct=True, method="average").to_numpy()
        for name, arr in preds.items()
    }
    return weighted_blend(ranked, weights)


def pairwise_diversity(
    name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, decile: float = 0.1
) -> dict:
    """Pearson/Spearman correlation, mean absolute difference, and top/bottom
    decile ranking disagreement between two aligned prediction arrays.

    Decile disagreement: among rows in `a`'s top `decile` fraction by
    probability, the fraction that fall outside `b`'s top `decile` (and
    symmetrically for the bottom decile). Two models can be highly
    correlated overall while still disagreeing near the ranking
    boundaries that matter most for ROC AUC -- this is why Pearson alone
    is not sufficient evidence of redundancy.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"mismatched lengths: {len(a)} != {len(b)}")

    n_decile = max(1, int(round(len(a) * decile)))
    top_a = set(np.argsort(a)[-n_decile:])
    top_b = set(np.argsort(b)[-n_decile:])
    bottom_a = set(np.argsort(a)[:n_decile])
    bottom_b = set(np.argsort(b)[:n_decile])

    return {
        "model_a": name_a,
        "model_b": name_b,
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(spearmanr(a, b).statistic),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "top_decile_disagreement": float(1 - len(top_a & top_b) / n_decile),
        "bottom_decile_disagreement": float(1 - len(bottom_a & bottom_b) / n_decile),
    }


def fold_auc_breakdown(oof_proba: np.ndarray, y: pd.Series, folds: pd.Series) -> list[float]:
    """Per-fold ROC AUC of `oof_proba` against `y`, grouped by `folds` (values 1..n_splits)."""
    oof_arr = np.asarray(oof_proba, dtype=float)
    y_arr = np.asarray(y)
    folds_arr = np.asarray(folds)
    return [
        float(roc_auc_score(y_arr[folds_arr == fold_id], oof_arr[folds_arr == fold_id]))
        for fold_id in sorted(pd.unique(folds_arr))
    ]
