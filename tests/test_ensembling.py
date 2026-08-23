"""Tests for src/ensembling.py.

Save/load round trips use monkeypatched artifact directories (pytest's
`tmp_path`) so tests never touch the real outputs/ directory.
"""

import numpy as np
import pandas as pd
import pytest

from src import ensembling


@pytest.fixture(autouse=True)
def _isolated_artifact_dirs(tmp_path, monkeypatch):
    """Redirects OOF/test-prediction/fold-assignment paths to a temp dir."""
    monkeypatch.setattr(ensembling, "OOF_DIR", tmp_path / "oof_predictions")
    monkeypatch.setattr(ensembling, "TEST_PRED_DIR", tmp_path / "test_predictions")
    monkeypatch.setattr(ensembling, "FOLD_ASSIGNMENTS_PATH", tmp_path / "cv_fold_assignments.csv")


def test_save_and_load_oof_round_trips_in_original_order() -> None:
    ids = pd.Series([103, 101, 102])
    proba = np.array([0.9, 0.1, 0.5])

    ensembling.save_oof("E999", ids, proba)
    loaded = ensembling.load_oof("E999", ids)

    assert loaded.to_numpy() == pytest.approx(proba)


def test_load_oof_realigns_to_requested_id_order() -> None:
    """Provenance is retained through id alignment, not row position."""
    write_ids = pd.Series([103, 101, 102])
    proba = np.array([0.9, 0.1, 0.5])
    ensembling.save_oof("E999", write_ids, proba)

    requested_order = pd.Series([101, 102, 103])
    loaded = ensembling.load_oof("E999", requested_order)

    assert loaded.to_numpy() == pytest.approx([0.1, 0.5, 0.9])


def test_load_oof_rejects_id_mismatch() -> None:
    ensembling.save_oof("E999", pd.Series([1, 2, 3]), np.array([0.1, 0.2, 0.3]))

    with pytest.raises(AssertionError, match="ids do not match"):
        ensembling.load_oof("E999", pd.Series([1, 2, 4]))


def test_save_oof_rejects_out_of_bounds_probabilities() -> None:
    with pytest.raises(AssertionError, match="outside \\[0, 1\\]"):
        ensembling.save_oof("E999", pd.Series([1, 2]), np.array([0.5, 1.5]))


def test_save_test_pred_rejects_missing_values() -> None:
    with pytest.raises(AssertionError, match="missing values"):
        ensembling.save_test_pred("E999", pd.Series([1, 2]), np.array([0.5, np.nan]))


def test_fold_assignments_round_trip_and_align() -> None:
    write_ids = pd.Series([30, 10, 20])
    ensembling.save_fold_assignments(write_ids, np.array([3, 1, 2]))

    loaded = ensembling.load_fold_assignments(pd.Series([10, 20, 30]))

    assert loaded.to_numpy().tolist() == [1, 2, 3]


def test_weighted_blend_normalizes_unequal_weights() -> None:
    preds = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}

    blend = ensembling.weighted_blend(preds, {"a": 3, "b": 1})

    assert blend == pytest.approx([0.75, 0.25])


def test_weighted_blend_rejects_mismatched_keys() -> None:
    preds = {"a": np.array([1.0]), "b": np.array([0.0])}
    with pytest.raises(ValueError, match="same keys"):
        ensembling.weighted_blend(preds, {"a": 1.0})


def test_weighted_blend_rejects_mismatched_lengths() -> None:
    preds = {"a": np.array([1.0, 2.0]), "b": np.array([0.0])}
    with pytest.raises(ValueError, match="mismatched lengths"):
        ensembling.weighted_blend(preds, {"a": 1.0, "b": 1.0})


def test_weighted_blend_rejects_nan() -> None:
    preds = {"a": np.array([1.0, np.nan]), "b": np.array([0.0, 0.0])}
    with pytest.raises(ValueError, match="NaN"):
        ensembling.weighted_blend(preds, {"a": 1.0, "b": 1.0})


def test_weighted_blend_rejects_nonpositive_total_weight() -> None:
    preds = {"a": np.array([1.0]), "b": np.array([0.0])}
    with pytest.raises(ValueError, match="positive total"):
        ensembling.weighted_blend(preds, {"a": 1.0, "b": -1.0})


def test_rank_blend_is_deterministic() -> None:
    preds = {"a": np.array([0.3, 0.9, 0.1, 0.5]), "b": np.array([0.7, 0.2, 0.4, 0.6])}

    first = ensembling.rank_blend(preds)
    second = ensembling.rank_blend(preds)

    assert first == pytest.approx(second)


def test_rank_blend_handles_ties_with_average_rank() -> None:
    preds = {"a": np.array([1.0, 1.0, 2.0, 3.0])}

    blend = ensembling.rank_blend(preds)

    # Both tied 1.0 values get the average of ranks 1 and 2 -> percentile 0.375.
    assert blend[0] == pytest.approx(blend[1])
    assert blend[0] < blend[2] < blend[3]


def test_pairwise_diversity_identical_arrays_has_zero_disagreement() -> None:
    a = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.0])

    result = ensembling.pairwise_diversity("m", a, "m", a)

    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["mean_abs_diff"] == pytest.approx(0.0)
    assert result["top_decile_disagreement"] == pytest.approx(0.0)
    assert result["bottom_decile_disagreement"] == pytest.approx(0.0)


def test_pairwise_diversity_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        ensembling.pairwise_diversity("a", np.array([1.0, 2.0]), "b", np.array([1.0]))


def test_fold_auc_breakdown_returns_one_score_per_fold() -> None:
    y = pd.Series([0, 1, 0, 1, 0, 1])
    oof = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6])
    folds = pd.Series([1, 1, 2, 2, 3, 3])

    scores = ensembling.fold_auc_breakdown(oof, y, folds)

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
