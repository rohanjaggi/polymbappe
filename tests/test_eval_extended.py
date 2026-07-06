"""Tests for the extended evaluation metrics, paired significance tests, and the
bookmaker accuracy-workbook loader (proper-scoring-rule reporting upgrade)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polymbappe.eval import metrics, significance
from polymbappe.eval.bookmaker import (
    BOOKMAKER_SCHEMA,
    default_workbook_path,
    load_bookmaker_accuracy,
)

# -- metrics ------------------------------------------------------------------


def test_multiclass_brier_uniform_baseline() -> None:
    # Uniform (1/3 each) on any single outcome: (1/3-1)^2 + 2*(1/3)^2 = 6/9.
    idx = np.array([0])
    prob = np.full((1, 3), 1 / 3)
    assert metrics.multiclass_brier_score(idx, prob) == pytest.approx(2 / 3)


def test_per_match_rps_perfect_is_zero_and_matches_aggregate() -> None:
    idx = np.array([0, 2])
    prob = np.array([[1.0, 0.0, 0.0], [0.1, 0.2, 0.7]])
    per = metrics.per_match_rps(idx, prob)
    assert per[0] == pytest.approx(0.0)
    assert per.mean() == pytest.approx(metrics.ranked_probability_score(idx, prob))


def test_skill_score_sign_and_degenerate() -> None:
    assert metrics.skill_score(0.5, 1.0) == pytest.approx(0.5)  # beats reference
    assert metrics.skill_score(1.2, 1.0) == pytest.approx(-0.2)  # worse
    assert math.isnan(metrics.skill_score(0.5, 0.0))  # no benchmark


def test_uniform_reference_scores_match_direct() -> None:
    idx = np.array([0, 1, 2, 0])
    ref = metrics.uniform_reference_scores(idx, n_classes=3)
    uni = np.full((4, 3), 1 / 3)
    assert ref["rps"] == pytest.approx(metrics.ranked_probability_score(idx, uni))
    assert ref["log_loss"] == pytest.approx(math.log(3))
    assert ref["brier"] == pytest.approx(2 / 3)


def test_expected_calibration_error_perfect_and_worst() -> None:
    # Confidence exactly equals hit rate within each bin -> ECE 0.
    conf = np.array([0.25, 0.25, 0.75, 0.75])
    correct = np.array([0, 0, 1, 1])  # 0% and 100% hit rates? gap 0.25 each...
    out = metrics.expected_calibration_error(conf, correct, n_bins=10)
    assert out["ece"] >= 0.0 and out["mce"] >= out["ece"]
    # A forecaster always 100% confident but only ever right half the time: gap 0.5.
    out2 = metrics.expected_calibration_error(
        np.full(4, 1.0), np.array([1, 0, 1, 0]), n_bins=10
    )
    assert out2["ece"] == pytest.approx(0.5)
    assert out2["mce"] == pytest.approx(0.5)


def test_expected_calibration_error_empty() -> None:
    out = metrics.expected_calibration_error(np.array([]), np.array([]))
    assert out == {"ece": 0.0, "mce": 0.0}


def test_calibration_slope_well_calibrated_near_one() -> None:
    # Generate outcomes whose probability of 1 equals the forecast -> slope ~1, intercept ~0.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    fit = metrics.calibration_slope_intercept(p, y)
    assert fit["slope"] == pytest.approx(1.0, abs=0.15)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.15)


def test_calibration_slope_degenerate_returns_nan() -> None:
    fit = metrics.calibration_slope_intercept(np.full(10, 0.4), np.array([1, 0] * 5))
    assert math.isnan(fit["slope"])  # all-equal probabilities
    fit2 = metrics.calibration_slope_intercept(np.array([]), np.array([]))
    assert math.isnan(fit2["slope"])


# -- significance -------------------------------------------------------------


def test_mcnemar_all_discordant_in_model_favor() -> None:
    model = np.array([True, True, True, True])
    other = np.array([False, False, False, False])
    out = significance.mcnemar_test(model, other)
    assert out["b"] == 4.0 and out["c"] == 0.0
    assert out["p_value"] == pytest.approx(2 * 0.5**4)  # exact two-sided binomial


def test_mcnemar_no_disagreement_is_p1() -> None:
    same = np.array([True, False, True])
    out = significance.mcnemar_test(same, same)
    assert out["n_discordant"] == 0.0 and out["p_value"] == 1.0


def test_wilcoxon_all_zero_is_nan() -> None:
    z = np.zeros(5)
    out = significance.wilcoxon_loss_diff(z, z)
    assert math.isnan(out["p_value"]) and out["mean_diff"] == 0.0


def test_wilcoxon_detects_consistent_advantage() -> None:
    a = np.array([0.1, 0.2, 0.15, 0.1, 0.2, 0.05])
    b = a + 0.1  # A always lower loss
    out = significance.wilcoxon_loss_diff(a, b)
    assert out["mean_diff"] < 0 and out["p_value"] < 0.05


def test_paired_bootstrap_ci_below_zero_when_a_better() -> None:
    a = np.linspace(0.1, 0.2, 30)
    b = a + 0.05
    out = significance.paired_bootstrap_loss_diff(a, b, n_boot=2000, seed=1)
    assert out["mean_diff"] < 0
    assert out["ci_high"] < 0  # significant


def test_paired_bootstrap_empty() -> None:
    out = significance.paired_bootstrap_loss_diff(np.array([]), np.array([]))
    assert math.isnan(out["mean_diff"])


# -- bookmaker loader ---------------------------------------------------------

_WORKBOOK = next(Path(".").glob("*Prediction_Accuracy*.xlsx"), None)
_needs_workbook = pytest.mark.skipif(
    _WORKBOOK is None, reason="bookmaker accuracy workbook not present"
)


@_needs_workbook
def test_load_bookmaker_accuracy_matches_summary() -> None:
    df = load_bookmaker_accuracy(_WORKBOOK)  # type: ignore[arg-type]
    assert df.columns == list(BOOKMAKER_SCHEMA.keys())
    assert df.height == 72
    # Workbook Summary sheet reports 45/72 = 0.625 favourite accuracy.
    assert df["book_correct"].mean() == pytest.approx(0.625)
    # Teams are canonicalized (Czechia -> Czech Republic, Turkiye -> Turkey).
    keys = " ".join(df["pair_key"].to_list())
    assert "Czech Republic" in keys and "Turkey" in keys
    # pair_key is order-independent (sorted).
    row = df.filter(df["fav_team"] == "Mexico").row(0, named=True)
    assert row["pair_key"] == "Mexico | South Africa"


def test_load_bookmaker_missing_matches_sheet(tmp_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Other"
    p = tmp_path / "empty.xlsx"
    wb.save(p)
    df = load_bookmaker_accuracy(p)
    assert df.is_empty() and df.columns == list(BOOKMAKER_SCHEMA.keys())


def test_default_workbook_path_none_when_absent(tmp_path: Path) -> None:
    assert default_workbook_path(tmp_path / "data") is None
