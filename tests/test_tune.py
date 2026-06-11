"""Tests for the autotuner: gate, leaderboard, search space, objective, runner."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from polymbappe.config import Settings
from polymbappe.eval.backtest import Tournament
from polymbappe.tune.leaderboard import AcceptanceGate, Leaderboard
from polymbappe.tune.llm_search import default_structural_experiments, propose_structural_experiment
from polymbappe.tune.objective import ExperimentMetrics, config_to_configs
from polymbappe.tune.runner import autotune, parse_budget_to_trials
from polymbappe.tune.search_space import load_search_space

TEAMS = ["A", "B", "C", "D"]
_ATTACK = {"A": 1.7, "B": 1.3, "C": 1.0, "D": 0.7}
_TOURNAMENTS = (
    Tournament("WC2016", "FIFA World Cup", date(2016, 6, 1), date(2016, 7, 31)),
    Tournament("EU2018", "UEFA Euro", date(2018, 6, 1), date(2018, 7, 31)),
    Tournament("CA2020", "Copa América", date(2020, 6, 1), date(2020, 7, 31)),
)


def _matches() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    idx = 0

    def add(d: date, h: str, a: str, comp: str, neutral: bool) -> None:
        nonlocal idx
        rows.append(
            {
                "match_id": f"m{idx}",
                "date": d,
                "home_team": h,
                "away_team": a,
                "home_goals": int(rng.poisson(_ATTACK[h] + (0 if neutral else 0.25))),
                "away_goals": int(rng.poisson(_ATTACK[a])),
                "competition": comp,
                "is_knockout": False,
                "neutral_site": neutral,
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
    for comp, year in (("FIFA World Cup", 2016), ("UEFA Euro", 2018), ("Copa América", 2020)):
        td = date(year, 6, 10)
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(td, h, a, comp, True)
                    td += timedelta(days=1)
    return pl.DataFrame(rows)


def test_acceptance_gate() -> None:
    gate = AcceptanceGate(min_delta=0.003, min_tournaments=2)
    best = ExperimentMetrics(0.210, {"t1": 0.21, "t2": 0.21, "t3": 0.21})
    better = ExperimentMetrics(0.200, {"t1": 0.20, "t2": 0.20, "t3": 0.21})
    worse = ExperimentMetrics(0.230, {"t1": 0.23, "t2": 0.23, "t3": 0.23})
    marginal = ExperimentMetrics(0.209, {"t1": 0.209, "t2": 0.211, "t3": 0.21})
    assert gate.decide(better, best) == "accept"
    assert gate.decide(worse, best) == "reject"
    assert gate.decide(marginal, best) == "inconclusive"
    assert gate.decide(better, None) == "accept"


def test_config_to_configs_maps_params() -> None:
    configs = config_to_configs(
        {
            "dixon_coles.xi": 0.001,
            "features.elo_k_factor": 30.0,
            "ensemble.meta_C": 0.5,
            "ensemble.meta_learner": "isotonic",
            "gbm.enable": True,
            "gbm.num_leaves": 40,
            "contextual.enable_contextual_layer": True,
            "contextual.context_n_estimators": 80,
        }
    )
    assert configs.base.dixon_coles.xi == 0.001
    assert configs.base.elo.k_factor == 30.0
    assert configs.ensemble.meta.C == 0.5
    assert configs.ensemble.meta.learner == "isotonic"
    assert configs.ensemble.use_gbm is True
    assert configs.ensemble.gbm.num_leaves == 40
    assert configs.contextual.enable_contextual_layer is True
    assert configs.contextual.n_estimators == 80


def test_config_to_configs_defaults_to_baseline() -> None:
    configs = config_to_configs({})
    assert configs.ensemble.use_gbm is False
    assert configs.ensemble.meta.learner == "logistic"
    assert configs.contextual.enable_contextual_layer is False


def test_search_space_loads_and_samples() -> None:
    space = load_search_space()
    assert any(p.name == "dixon_coles.xi" for p in space.params)

    class _Trial:
        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2

        def suggest_int(self, name, low, high):
            return (low + high) // 2

        def suggest_categorical(self, name, choices):
            return choices[0]

    sample = space.sample(_Trial())
    assert "dixon_coles.xi" in sample
    assert sample["dixon_coles.max_goals"] == 8


def test_leaderboard_roundtrip(tmp_path) -> None:
    board = Leaderboard(Settings(data_dir=tmp_path))
    m = ExperimentMetrics(0.205, {"WC2016": 0.20})
    board.record("e1", "phase1", "accept", m, {"x": 1}, "hypo")
    board.record("e2", "phase1", "reject", ExperimentMetrics(0.25, {}), {"x": 2})
    df = board.load()
    assert df.height == 2
    best = board.best()
    assert best["experiment_id"].item() == "e1"


def test_structural_fallback_cycles() -> None:
    exps = default_structural_experiments()
    first = propose_structural_experiment([])
    assert first.name == exps[0].name
    second = propose_structural_experiment([{"name": exps[0].name}])
    assert second.name == exps[1].name


def test_parse_budget() -> None:
    assert parse_budget_to_trials("2h", trials_per_hour=10) == 20
    assert parse_budget_to_trials("30m", trials_per_hour=60) == 30
    assert parse_budget_to_trials("garbage", trials_per_hour=42) == 42


def test_autotune_end_to_end(tmp_path) -> None:
    board = Leaderboard(Settings(data_dir=tmp_path))
    result = autotune(
        _matches(),
        tournaments=_TOURNAMENTS,
        n_structural=2,
        n_trials=3,
        leaderboard=board,
    )
    assert np.isfinite(result.baseline_metrics.mean_rps)
    assert np.isfinite(result.best_metrics.mean_rps)
    # Best is never worse than baseline (gate only accepts improvements).
    assert result.best_metrics.mean_rps <= result.baseline_metrics.mean_rps + 1e-9
    assert board.load().height >= 5  # 2 structural + 3 trials + phase2-best
