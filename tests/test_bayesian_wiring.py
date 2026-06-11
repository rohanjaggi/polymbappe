"""Wiring tests for the Bayesian hierarchical DC integration.

The real model needs PyMC (NUTS); these tests stub it out so the *plumbing* — bay_*/ci_*
production, the ensemble stacking the ``bay`` group, the simulate credible-edge branch, and
the +0.003 kill-criterion A/B — is exercised in pymc-free CI. The model itself is covered
by ``test_bayesian_dc.py`` (which skips without pymc).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from polymbappe.eval.backtest import (
    BacktestResult,
    Tournament,
    compare_bayesian_ab,
    run_leave_one_tournament_out,
)
from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs
from polymbappe.simulate import tournament as sim

_CI_COLS = (
    "ci_home_low", "ci_home_high", "ci_draw_low", "ci_draw_high", "ci_away_low", "ci_away_high",
)


class _FakeBayes:
    """Stand-in for BayesianDixonColesModel with deterministic, pymc-free outputs."""

    def __init__(self, config: object | None = None) -> None:
        self.config = config
        self.attack: object | None = None  # set on fit, like the real model

    def fit(self, *args: object, **kwargs: object) -> _FakeBayes:
        self.attack = object()
        return self

    def predict_match(
        self, home_team: str, away_team: str, neutral_site: bool = False
    ) -> dict[str, float]:
        return {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}

    def credible_interval(
        self, home_team: str, away_team: str, neutral_site: bool = False, level: float = 0.9
    ) -> dict[str, tuple[float, float]]:
        return {"home_win": (0.40, 0.60), "draw": (0.20, 0.40), "away_win": (0.10, 0.30)}


TEAMS = ["A", "B", "C", "D"]
_ATTACK = {"A": 1.7, "B": 1.3, "C": 1.0, "D": 0.7}

_TOURNAMENTS = (
    Tournament("WC2016", "FIFA World Cup", date(2016, 6, 1), date(2016, 7, 31)),
    Tournament("EU2018", "UEFA Euro", date(2018, 6, 1), date(2018, 7, 31)),
)


def _make_matches() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    idx = 0

    def add(d: date, home: str, away: str, comp: str, neutral: bool) -> None:
        nonlocal idx
        rows.append(
            {
                "match_id": f"m{idx}", "date": d, "home_team": home, "away_team": away,
                "home_goals": int(rng.poisson(_ATTACK[home] + (0 if neutral else 0.25))),
                "away_goals": int(rng.poisson(_ATTACK[away])),
                "competition": comp, "is_knockout": False, "neutral_site": neutral,
                "group": None,
            }
        )
        idx += 1

    day = date(2008, 1, 1)
    for _ in range(20):
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(day, h, a, "Friendly", False)
                    day += timedelta(days=7)
    for comp, year in (("FIFA World Cup", 2016), ("UEFA Euro", 2018)):
        td = date(year, 6, 10)
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(td, h, a, comp, True)
                    td += timedelta(days=1)
    return pl.DataFrame(rows)


def test_base_probs_emits_bay_and_ci(monkeypatch) -> None:
    monkeypatch.setattr("polymbappe.eval.base_probs.BayesianDixonColesModel", _FakeBayes)
    matches = _make_matches()
    history = matches.filter(pl.col("competition") == "Friendly")
    fixtures = matches.filter(pl.col("competition") == "FIFA World Cup")

    df = compute_tournament_base_probs(
        history, fixtures, tournament="WC2016", config=BaseProbConfig(use_bayesian=True)
    )
    for col in ("bay_home", "bay_draw", "bay_away", *_CI_COLS):
        assert col in df.columns
    # Off by default: no bay_*/ci_* columns leak when the flag is unset.
    df_off = compute_tournament_base_probs(
        history, fixtures, tournament="WC2016", config=BaseProbConfig()
    )
    assert "bay_home" not in df_off.columns
    assert "ci_home_low" not in df_off.columns


def test_backtest_stacks_bay_group(monkeypatch) -> None:
    monkeypatch.setattr("polymbappe.eval.base_probs.BayesianDixonColesModel", _FakeBayes)
    result = run_leave_one_tournament_out(
        _make_matches(), _TOURNAMENTS, base_config=BaseProbConfig(use_bayesian=True)
    )
    for col in ("bay_home", "bay_draw", "bay_away"):
        assert col in result.feature_columns
    # Without the flag the bay group is absent (the original DC+Elo stack).
    off = run_leave_one_tournament_out(_make_matches(), _TOURNAMENTS)
    assert "bay_home" not in off.feature_columns


def _result(rps_by_name: dict[str, float]) -> BacktestResult:
    return BacktestResult(
        per_tournament={n: {"rps": r} for n, r in rps_by_name.items()},
        feature_columns=[],
    )


def test_compare_bayesian_ab_kill_criterion(monkeypatch) -> None:
    names = ["T1", "T2", "T3", "T4"]
    without = _result({n: 0.200 for n in names})

    def fake_run(matches, tournaments=_TOURNAMENTS, *, base_config=None, **kwargs):
        if base_config is not None and base_config.use_bayesian:
            return monkeypatch._with  # type: ignore[attr-defined]
        return without

    monkeypatch.setattr("polymbappe.eval.backtest.run_leave_one_tournament_out", fake_run)

    # Improves every tournament by 0.004 (> 0.003) -> keep, and wins 4/4 -> gate accepts.
    monkeypatch._with = _result({n: 0.196 for n in names})  # type: ignore[attr-defined]
    good = compare_bayesian_ab(pl.DataFrame())
    assert good.delta > good.min_delta
    assert good.keep_bayesian is True
    assert good.wins == 4
    assert good.gate_decision == "accept"

    # Improves by only 0.002 (< 0.003) -> drop the Bayesian model.
    monkeypatch._with = _result({n: 0.198 for n in names})  # type: ignore[attr-defined]
    marginal = compare_bayesian_ab(pl.DataFrame())
    assert marginal.keep_bayesian is False
    assert marginal.gate_decision in {"inconclusive", "reject"}


def test_write_edges_uses_credible_path_when_ci_present(monkeypatch) -> None:
    predictions = pl.DataFrame(
        {
            "match_id": ["2026__A__B"],
            "model_home": [0.62], "model_draw": [0.20], "model_away": [0.18],
            "ci_home_low": [0.55], "ci_home_high": [0.69],
            "ci_draw_low": [0.15], "ci_draw_high": [0.30],
            "ci_away_low": [0.12], "ci_away_high": [0.26],
        }
    )
    market = pl.DataFrame(
        {
            "match_id": ["2026__A__B"],
            "home_win_prob": [0.50], "draw_prob": [0.27], "away_win_prob": [0.23],
        }
    )
    monkeypatch.setattr("polymbappe.data.store.table_exists", lambda *a, **k: True)
    monkeypatch.setattr("polymbappe.data.store.read_table", lambda *a, **k: market)

    class _Logger:
        def info(self, *a: object, **k: object) -> None:  # noqa: D401 - test stub
            pass

    edges = sim._write_edges(predictions, object(), _Logger())
    # Credible-interval columns flow through, and only the CI-excluding home edge survives.
    assert {"ci_low", "ci_high"}.issubset(edges.columns)
    assert edges["match_id"].to_list() == ["2026__A__B"]
    assert edges["outcome"].to_list() == ["H"]


def test_compute_match_predictions_emits_ci(monkeypatch) -> None:
    structure = sim.TournamentStructure(groups={"A": ["A", "B", "C", "D"]})

    class _Model:
        max_goals = 6

        def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
            m = np.full((7, 7), 1.0)
            return m / m.sum()

    preds = sim.compute_match_predictions(structure, _Model(), bayesian_model=_FakeBayes())
    for col in _CI_COLS:
        assert col in preds.columns
    # No Bayesian model -> point predictions only.
    preds_off = sim.compute_match_predictions(structure, _Model())
    assert "ci_home_low" not in preds_off.columns
