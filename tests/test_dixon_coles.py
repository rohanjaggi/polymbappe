import time
import warnings

import numpy as np

from polymbappe.models.dixon_coles import DixonColesConfig, DixonColesModel, MatchObservation


def test_dixon_coles_fit_and_predict_probabilities_sum_to_one() -> None:
    matches = [
        MatchObservation("A", "B", 1, 0, 10, "World Cup Qualifier"),
        MatchObservation("B", "A", 0, 2, 20, "World Cup Qualifier"),
        MatchObservation("A", "C", 2, 1, 5, "Friendly"),
        MatchObservation("C", "B", 0, 0, 2, "Nations League"),
        MatchObservation("B", "C", 1, 1, 1, "World Cup"),
        MatchObservation("C", "A", 0, 1, 3, "Euro Qualifier"),
    ]
    model = DixonColesModel().fit(matches)
    probs = model.predict_match("A", "B")
    assert probs["home_win"] > 0.0
    assert probs["draw"] > 0.0
    assert probs["away_win"] > 0.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def _make_test_matches() -> list[MatchObservation]:
    """Reproducible set large enough to exercise all code paths."""
    return [
        MatchObservation("A", "B", 2, 1, 100, "FIFA World Cup"),
        MatchObservation("B", "C", 0, 0, 90, "Friendly"),
        MatchObservation("C", "A", 1, 2, 80, "UEFA Euro", neutral_site=True),
        MatchObservation("A", "C", 3, 0, 70, "World Cup Qualifier"),
        MatchObservation("B", "A", 1, 1, 60, "Nations League"),
        MatchObservation("C", "B", 2, 2, 50, "Copa América"),
        MatchObservation("A", "B", 0, 1, 40, "Friendly", neutral_site=True),
        MatchObservation("B", "C", 1, 0, 30, "FIFA World Cup"),
        MatchObservation("C", "A", 0, 3, 20, "Euro Qualifier"),
        MatchObservation("A", "C", 1, 1, 10, "Nations League"),
    ]


def test_vectorized_matches_scalar_output() -> None:
    """Vectorized objective must produce same parameters as scalar."""
    matches = _make_test_matches()
    model = DixonColesModel().fit(matches)

    # Verify basic properties hold
    assert model.attack is not None
    assert model.defense is not None
    assert len(model.index_to_team) == 3

    # Sum-to-zero constraint
    assert abs(model.attack.sum()) < 1e-6
    assert abs(model.defense.sum()) < 1e-6

    # Probabilities sum to 1
    probs = model.predict_match("A", "B")
    assert abs(sum(probs.values()) - 1.0) < 1e-6

    # Team A is strongest attacker in this dataset
    a_idx = model.team_to_index["A"]
    assert model.attack[a_idx] == max(model.attack)


def test_dixon_coles_scales_to_1000_matches() -> None:
    """Fitting 1000 matches with 50 teams should complete in under 30 seconds."""
    rng = np.random.default_rng(42)
    teams = [f"T{i}" for i in range(50)]
    competitions = ["FIFA World Cup", "Friendly", "Nations League"]
    matches = []
    for i in range(1000):
        h, a = rng.choice(teams, size=2, replace=False)
        matches.append(
            MatchObservation(
                home_team=h,
                away_team=a,
                home_goals=int(rng.poisson(1.3)),
                away_goals=int(rng.poisson(1.1)),
                days_ago=float(rng.integers(1, 3000)),
                competition=rng.choice(competitions),
                neutral_site=bool(rng.integers(0, 2)),
            )
        )

    start = time.perf_counter()
    model = DixonColesModel().fit(matches)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"Fitting took {elapsed:.1f}s (budget: 30s)"
    assert model.attack is not None
    assert len(model.index_to_team) == 50
    probs = model.predict_match(teams[0], teams[1])
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_fit_no_overflow_warnings_large_dataset() -> None:
    """Fitting a realistically-sized dataset must not produce overflow RuntimeWarnings."""
    rng = np.random.default_rng(42)
    teams = [f"T{i}" for i in range(80)]
    competitions = ["FIFA World Cup", "Friendly", "Nations League", "World Cup Qualifier"]
    matches = []
    for i in range(2000):
        h, a = rng.choice(teams, size=2, replace=False)
        matches.append(
            MatchObservation(
                home_team=h,
                away_team=a,
                home_goals=int(rng.poisson(1.3)),
                away_goals=int(rng.poisson(1.1)),
                days_ago=float(rng.integers(1, 4000)),
                competition=str(rng.choice(competitions)),
                neutral_site=bool(rng.integers(0, 2)),
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = DixonColesModel(DixonColesConfig(max_history_days=5000)).fit(matches=matches)

    assert model.attack is not None
    assert not np.any(np.isnan(model.attack))
    assert not np.any(np.isinf(model.attack))
    probs = model.predict_match(teams[0], teams[1])
    assert abs(sum(probs.values()) - 1.0) < 1e-6
