"""Smoke + sanity tests for the Bayesian hierarchical Dixon-Coles model.

PyMC is an optional dependency; tests skip cleanly if it is unavailable. Sampler
settings are deliberately tiny — we assert the fit runs, produces a valid posterior
predictive simplex, exposes credible intervals, and orders a strong-vs-weak matchup
correctly, not that it has converged.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc")

from polymbappe.models.bayesian_dc import BayesianConfig, BayesianDixonColesModel
from polymbappe.models.dixon_coles import MatchObservation

_ATTACK = {"A": 1.7, "B": 1.3, "C": 1.0, "D": 0.7}
_CONF = {"A": "UEFA", "B": "UEFA", "C": "CONMEBOL", "D": "CONMEBOL"}


def _observations() -> list[MatchObservation]:
    rng = np.random.default_rng(11)
    obs: list[MatchObservation] = []
    for rep in range(12):
        for home in _ATTACK:
            for away in _ATTACK:
                if home == away:
                    continue
                lam = _ATTACK[home] + 0.2
                mu = _ATTACK[away]
                obs.append(
                    MatchObservation(
                        home_team=home,
                        away_team=away,
                        home_goals=int(rng.poisson(lam)),
                        away_goals=int(rng.poisson(mu)),
                        days_ago=float(rep * 30),
                        competition="FIFA World Cup",
                        neutral_site=True,
                    )
                )
    return obs


def _tiny_config() -> BayesianConfig:
    return BayesianConfig(n_tune=150, n_draws=150, chains=1, cores=1, target_accept=0.9)


def test_fit_and_predict_simplex() -> None:
    model = BayesianDixonColesModel(_tiny_config()).fit(
        matches=_observations(), team_confederation=_CONF
    )
    probs = model.predict_match("A", "D", neutral_site=True)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-6
    # Strong A at home vs weak D: home win clearly more likely than away win.
    assert probs["home_win"] > probs["away_win"]


def test_posterior_draws_and_credible_interval() -> None:
    model = BayesianDixonColesModel(_tiny_config()).fit(
        matches=_observations(), team_confederation=_CONF
    )
    draws = model.predict_proba_draws("A", "D", neutral_site=True, max_draws=50)
    assert draws.shape[1] == 3
    assert np.allclose(draws.sum(axis=1), 1.0, atol=1e-6)
    ci = model.credible_interval("A", "D", neutral_site=True, level=0.9)
    lo, hi = ci["home_win"]
    assert 0.0 <= lo <= hi <= 1.0


def test_unknown_team_returns_uniform() -> None:
    model = BayesianDixonColesModel(_tiny_config()).fit(
        matches=_observations(), team_confederation=_CONF
    )
    probs = model.predict_match("A", "ZZ")
    assert probs == {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
