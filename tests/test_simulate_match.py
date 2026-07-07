"""Tests for single-match simulation (scoreline, ET, penalties)."""

from __future__ import annotations

import numpy as np

from polymbappe.simulate.match import (
    hda_marginals,
    knockout_home_winprob,
    knockout_outcome_breakdown,
    penalty_home_winprob,
    reweight_matrix_to_hda,
    sample_scoreline,
    score_matrix_from_rates,
    shrink_penalty_rate,
    simulate_knockout_match,
)


def test_reweight_matrix_to_hda_hits_target() -> None:
    m = score_matrix_from_rates(1.6, 1.1, -0.05, 8)
    target = np.array([0.5, 0.3, 0.2])
    out = reweight_matrix_to_hda(m, target)
    assert abs(out.sum() - 1.0) < 1e-9
    assert np.allclose(hda_marginals(out), target, atol=1e-9)
    # Reweighting preserves scoreline shape within each region (relative order kept).
    assert out.shape == m.shape


def test_score_matrix_normalized() -> None:
    m = score_matrix_from_rates(1.6, 1.1, -0.05, 8)
    assert abs(m.sum() - 1.0) < 1e-9
    assert np.all(m >= 0.0)


def test_sample_scoreline_distribution() -> None:
    rng = np.random.default_rng(0)
    m = score_matrix_from_rates(2.0, 0.5, 0.0, 8)
    samples = [sample_scoreline(m, rng) for _ in range(2000)]
    mean_home = np.mean([h for h, _ in samples])
    mean_away = np.mean([a for _, a in samples])
    assert mean_home > mean_away  # strong home rate


def test_penalty_shrink_and_winprob() -> None:
    # A perfect record (1.0) over 2 shootouts shrinks well below 1.0.
    assert shrink_penalty_rate(1.0, 2.0) < 0.65
    # Even rates: first shooter gets the small edge.
    p_first = penalty_home_winprob(0.5, 0.5, first_shooter_home=True)
    p_second = penalty_home_winprob(0.5, 0.5, first_shooter_home=False)
    assert p_first > 0.5 > p_second


def test_knockout_winprob_monotonic_in_strength() -> None:
    strong = score_matrix_from_rates(2.2, 0.8, 0.0, 8)
    even = score_matrix_from_rates(1.3, 1.3, 0.0, 8)
    et = score_matrix_from_rates(1.0, 1.0, 0.0, 8)
    p_strong = knockout_home_winprob(strong, et)
    p_even = knockout_home_winprob(even, et)
    assert 0.0 < p_even < p_strong < 1.0
    assert abs(p_even - 0.5) < 0.05  # even matchup ~ coin flip


def test_knockout_outcome_breakdown_sums_and_agrees() -> None:
    reg = score_matrix_from_rates(1.8, 1.0, 0.0, 8)
    et = score_matrix_from_rates(1.0, 0.6, 0.0, 8)
    b = knockout_outcome_breakdown(reg, et, home_pen_rate=0.55, away_pen_rate=0.5)
    # Advance probabilities partition the tie.
    assert abs(b.p_home_advance + b.p_away_advance - 1.0) < 1e-9
    # Decided-phase probabilities partition the tie.
    assert abs(b.p_decided_reg + b.p_decided_et + b.p_decided_pens - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in (b.p_decided_reg, b.p_decided_et, b.p_decided_pens))
    # The advance prob agrees with the closed-form used by the engine on identical inputs.
    expected = knockout_home_winprob(reg, et, home_pen_rate=0.55, away_pen_rate=0.5)
    assert abs(b.p_home_advance - expected) < 1e-12
    # Stronger side favoured; regulation is the most common decider here.
    assert b.p_home_advance > 0.5
    assert b.p_decided_reg > b.p_decided_pens


def test_knockout_outcome_breakdown_symmetric_is_even() -> None:
    reg = score_matrix_from_rates(1.3, 1.3, 0.0, 8)
    et = score_matrix_from_rates(0.9, 0.9, 0.0, 8)
    b = knockout_outcome_breakdown(reg, et, first_shooter_home=False)
    assert abs(b.p_home_advance - 0.5) < 0.02


def test_simulate_knockout_always_decides() -> None:
    rng = np.random.default_rng(1)
    reg = score_matrix_from_rates(1.3, 1.3, 0.0, 8)
    decided = {"regulation": 0, "extra_time": 0, "penalties": 0}
    for _ in range(300):
        out = simulate_knockout_match(reg, (1.3, 1.3, 0.0), rng)
        decided[out.decided_by] += 1
        assert isinstance(out.home_advances, bool)
    # An evenly-matched tie should sometimes reach ET and penalties.
    assert decided["penalties"] > 0
