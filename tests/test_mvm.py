from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from polymbappe.eval.backtest import Tournament, run_leave_one_tournament_out, select_fixtures
from polymbappe.eval.base_probs import (
    BaseProbConfig,
    compute_tournament_base_probs,
    elo_probabilities,
)
from polymbappe.eval.market import compute_edges, kelly_fraction
from polymbappe.models.meta import MetaConfig, MetaLearner
from polymbappe.tune.objective import config_to_configs

TEAMS = ["A", "B", "C", "D"]
_ATTACK = {"A": 1.7, "B": 1.3, "C": 1.0, "D": 0.7}


def _make_matches() -> pl.DataFrame:
    """Deterministic synthetic history + three neutral 'tournaments'."""

    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    idx = 0

    def add(d: date, home: str, away: str, competition: str, neutral: bool) -> None:
        nonlocal idx
        lam_h = _ATTACK[home] + (0.0 if neutral else 0.25)
        lam_a = _ATTACK[away]
        hg = int(rng.poisson(lam_h))
        ag = int(rng.poisson(lam_a))
        rows.append(
            {
                "match_id": f"m{idx}",
                "date": d,
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "competition": competition,
                "is_knockout": False,
                "neutral_site": neutral,
                "group": None,
            }
        )
        idx += 1

    # History: repeated round-robin friendlies 2008-2015.
    day = date(2008, 1, 1)
    for _ in range(20):
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(day, h, a, "Friendly", False)
                    day += timedelta(days=7)

    # Three neutral-site tournaments (round-robin, 12 matches each).
    for comp, year in (("FIFA World Cup", 2016), ("UEFA Euro", 2018), ("Copa América", 2020)):
        td = date(year, 6, 10)
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(td, h, a, comp, True)
                    td += timedelta(days=1)

    return pl.DataFrame(rows)


TOURNAMENTS = (
    Tournament("WC2016", "FIFA World Cup", date(2016, 6, 1), date(2016, 7, 31)),
    Tournament("EU2018", "UEFA Euro", date(2018, 6, 1), date(2018, 7, 31)),
    Tournament("CA2020", "Copa América", date(2020, 6, 1), date(2020, 7, 31)),
)


def test_elo_probabilities_shape_and_ordering() -> None:
    probs = elo_probabilities(np.array([0.5, 0.85, 0.15]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Even matchup carries the most draw mass.
    assert probs[0, 1] > probs[1, 1]
    # Strong home favorite: home > away.
    assert probs[1, 0] > probs[1, 2]


def test_meta_learner_predicts_simplex() -> None:
    df = pl.DataFrame(
        {
            "dc_home": [0.5, 0.2, 0.4, 0.6, 0.33, 0.1],
            "dc_draw": [0.3, 0.3, 0.3, 0.25, 0.34, 0.3],
            "dc_away": [0.2, 0.5, 0.3, 0.15, 0.33, 0.6],
            "label": ["H", "A", "D", "H", "D", "A"],
        }
    )
    meta = MetaLearner(["dc_home", "dc_draw", "dc_away"]).fit(df)
    proba = meta.predict_proba(df)
    assert proba.shape == (6, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)


def test_meta_learner_alternate_families_predict_simplex() -> None:
    df = pl.DataFrame(
        {
            "dc_home": [0.5, 0.2, 0.4, 0.6, 0.33, 0.1],
            "dc_draw": [0.3, 0.3, 0.3, 0.25, 0.34, 0.3],
            "dc_away": [0.2, 0.5, 0.3, 0.15, 0.33, 0.6],
            "elo_home": [0.45, 0.25, 0.42, 0.58, 0.34, 0.12],
            "elo_draw": [0.30, 0.30, 0.28, 0.27, 0.33, 0.28],
            "elo_away": [0.25, 0.45, 0.30, 0.15, 0.33, 0.60],
            "label": ["H", "A", "D", "H", "D", "A"],
        }
    )
    cols = ["dc_home", "dc_draw", "dc_away", "elo_home", "elo_draw", "elo_away"]
    for learner in ("isotonic", "weighted_average"):
        meta = MetaLearner(cols, MetaConfig(learner=learner)).fit(df)
        proba = meta.predict_proba(df)
        assert proba.shape == (6, 3), learner
        assert np.allclose(proba.sum(axis=1), 1.0), learner
        assert np.all(proba >= 0.0), learner


def test_backtest_full_stack_gbm_and_contextual() -> None:
    configs = config_to_configs(
        {
            "gbm.enable": True,
            "ensemble.meta_learner": "logistic",
            "contextual.enable_contextual_layer": True,
        }
    )
    result = run_leave_one_tournament_out(
        _make_matches(),
        TOURNAMENTS,
        base_config=configs.base,
        ensemble_config=configs.ensemble,
        contextual_config=configs.contextual,
    )
    assert set(result.per_tournament) == {"WC2016", "EU2018", "CA2020"}
    for metrics in result.per_tournament.values():
        assert 0.0 <= metrics["rps"] <= 1.2
    assert np.isfinite(result.mean_rps)


def _squad_valuations_for(tournaments) -> pl.DataFrame:
    """Per-(team, tournament) squad values, monotonically increasing A>B>C>D."""

    rows: list[dict[str, object]] = []
    for t in tournaments:
        for mult, team in enumerate(reversed(TEAMS), start=1):
            rows.append(
                {
                    "team": team, "tournament": t.name,
                    "total_value": float(mult * 100_000_000),
                    "median_value": float(mult * 5_000_000),
                    "player_count": 23,
                }
            )
    return pl.DataFrame(rows)


def test_squad_value_ratio_is_point_in_time() -> None:
    """``_squad_value_ratio`` emits home−away log values per fixture, only when both sides
    have that tournament's snapshot."""

    import math

    from polymbappe.eval.backtest import _squad_value_ratio

    matches = _make_matches()
    ratio = _squad_value_ratio(matches, _squad_valuations_for(TOURNAMENTS), TOURNAMENTS)
    assert ratio.columns == ["match_id", "squad_value_ratio"]
    assert not ratio.is_empty()
    # A (total 400M) at home vs D (total 100M): log1p(400M) - log1p(100M) > 0.
    wc16 = select_fixtures(matches, TOURNAMENTS[0])
    a_vs_d = wc16.filter((pl.col("home_team") == "A") & (pl.col("away_team") == "D")).row(
        0, named=True
    )
    val = ratio.filter(pl.col("match_id") == a_vs_d["match_id"]).row(0, named=True)
    assert val["squad_value_ratio"] == math.log1p(400_000_000.0) - math.log1p(100_000_000.0)
    assert val["squad_value_ratio"] > 0


def test_backtest_squad_value_wires_into_gbm() -> None:
    """squad_valuations stacks squad_value_ratio into the backtest GBM (autotuner path):
    it appears in feature_columns when the GBM is on, and is absent when toggled off."""

    matches = _make_matches()
    valuations = _squad_valuations_for(TOURNAMENTS)
    configs = config_to_configs({"gbm.enable": True})

    with_squad = run_leave_one_tournament_out(
        matches, TOURNAMENTS, base_config=configs.base,
        ensemble_config=configs.ensemble, squad_valuations=valuations,
    )
    assert "squad_value_ratio" in with_squad.feature_columns
    assert np.isfinite(with_squad.mean_rps)

    # No valuations -> feature absent (the dead-data baseline).
    without = run_leave_one_tournament_out(
        matches, TOURNAMENTS, base_config=configs.base, ensemble_config=configs.ensemble,
    )
    assert "squad_value_ratio" not in without.feature_columns


def test_objective_toggle_squad_value_off_drops_feature() -> None:
    """``features.toggle_squad_value=false`` makes the objective pass no squad data, so the
    feature stays out even when valuations are available."""

    from polymbappe.tune.objective import config_to_metrics

    matches = _make_matches()
    valuations = _squad_valuations_for(TOURNAMENTS)
    off = config_to_metrics(
        {"gbm.enable": True, "features.toggle_squad_value": False},
        matches, tournaments=TOURNAMENTS, squad_valuations=valuations,
    )
    assert "squad_value_ratio" not in off.feature_columns
    on = config_to_metrics(
        {"gbm.enable": True, "features.toggle_squad_value": True},
        matches, tournaments=TOURNAMENTS, squad_valuations=valuations,
    )
    assert "squad_value_ratio" in on.feature_columns


def _write_backtest_context_tables(settings) -> None:
    """Squads + manager records for the three backtest tournaments (every team)."""

    from polymbappe.data.store import write_table
    from polymbappe.data.tables import Table

    squad_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    for order, t in enumerate(TOURNAMENTS, start=1):
        for team in TEAMS:
            for p, club in enumerate(("City", "City", "Madrid")):
                squad_rows.append(
                    {
                        "team": team, "tournament": t.name, "player": f"{team}p{p}",
                        "club": club, "age": 27.0 + p,
                    }
                )
            record_rows.append(
                {
                    "manager": f"mgr{team}", "team": team, "tournament": t.name,
                    "stage_reached": "FINAL", "knockout_matches": 6, "knockout_wins": 5,
                    "tournament_order": order,
                }
            )
    write_table(Table.SQUADS, pl.DataFrame(squad_rows), settings=settings)
    write_table(Table.MANAGER_RECORDS, pl.DataFrame(record_rows), settings=settings)


def test_prepare_contextual_wires_cohesion_and_manager_groups(tmp_path, monkeypatch) -> None:
    """Contextual-on smoke over real squads/manager data: groups wired + columns joined.

    ``_prepare_contextual`` / ``run_leave_one_tournament_out`` read the ``squads`` /
    ``manager_records`` tables through the *default* ``Settings`` (relative ``data/``), so
    we ``chdir`` into a tmp dir and materialize the tables there. Asserts the contextual
    layer engages, the returned ``feature_groups`` include cohesion + manager, those
    columns are actually joined onto the per-tournament frames with real non-zero data, and
    the full backtest runs without crashing.
    """

    from polymbappe.config import Settings
    from polymbappe.eval.backtest import _prepare_contextual

    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.data_dir == Path("data")  # relative -> resolves under cwd (tmp)
    _write_backtest_context_tables(settings)

    configs = config_to_configs(
        {
            "contextual.enable_contextual_layer": True,
            "contextual.toggle_cohesion": True,
            "contextual.toggle_manager": True,
        }
    )
    matches = _make_matches()
    sorted_tournaments = sorted(TOURNAMENTS, key=lambda t: t.start)
    per_tournament_probs = {
        t.name: compute_tournament_base_probs(
            matches.filter(pl.col("date") < t.start),
            select_fixtures(matches, t),
            tournament=t.name,
            config=configs.base,
        )
        for t in sorted_tournaments
    }

    enabled, feature_groups = _prepare_contextual(
        matches, sorted_tournaments, per_tournament_probs, configs.contextual
    )
    assert enabled is True
    assert "cohesion" in feature_groups
    assert "manager" in feature_groups
    # The cohesion/manager columns are joined with real non-zero data (not just present).
    joined = per_tournament_probs[sorted_tournaments[1].name]
    assert "home_club_cluster_index" in joined.columns
    assert joined.filter(pl.col("home_club_cluster_index") != 0.0).height > 0
    assert joined.filter(pl.col("home_knockout_win_rate") != 0.0).height > 0

    # End-to-end backtest with the contextual layer on runs and scores finitely.
    result = run_leave_one_tournament_out(
        matches,
        TOURNAMENTS,
        base_config=configs.base,
        ensemble_config=configs.ensemble,
        contextual_config=configs.contextual,
    )
    assert set(result.per_tournament) == {"WC2016", "EU2018", "CA2020"}
    assert np.isfinite(result.mean_rps)


def test_compute_tournament_base_probs() -> None:
    matches = _make_matches()
    fixtures = select_fixtures(matches, TOURNAMENTS[0])
    history = matches.filter(pl.col("date") < TOURNAMENTS[0].start)
    probs = compute_tournament_base_probs(
        history, fixtures, tournament="WC2016", config=BaseProbConfig()
    )
    assert probs.height == fixtures.height
    for prefix in ("dc", "elo"):
        triple = probs.select(f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away").to_numpy()
        assert np.allclose(triple.sum(axis=1), 1.0, atol=1e-6)
    assert set(probs["label"].to_list()) <= {"H", "D", "A"}

    # Directional sanity: strong A vs weak D favors A in both base models (no inversion).
    a_vs_d = (
        probs.join(fixtures, on="match_id")
        .filter((pl.col("home_team") == "A") & (pl.col("away_team") == "D"))
        .row(0, named=True)
    )
    assert a_vs_d["dc_home"] > a_vs_d["dc_away"]
    assert a_vs_d["elo_home"] > a_vs_d["elo_away"]


def test_leave_one_tournament_out_runs_and_scores() -> None:
    result = run_leave_one_tournament_out(_make_matches(), TOURNAMENTS)
    assert set(result.per_tournament) == {"WC2016", "EU2018", "CA2020"}
    tier1 = [
        f"{side}_{stat}_{window}"
        for side in ("home", "away")
        for window in (5, 10)
        for stat in ("gs", "ga", "pts")
    ]
    assert result.feature_columns == [
        "dc_home",
        "dc_draw",
        "dc_away",
        "elo_home",
        "elo_draw",
        "elo_away",
        *tier1,
        "h2h_home_winrate",
        "h2h_meetings",
        "home_rest_days",
        "away_rest_days",
    ]
    # Toy data (4 teams, ~24 meta training rows) won't hit RPS<0.21 — the real-data
    # target. Here we assert the pipeline runs and yields finite, in-range scores.
    for metrics in result.per_tournament.values():
        assert 0.0 <= metrics["rps"] <= 1.2
        assert metrics["log_loss"] > 0.0
        assert metrics["n"] == 12.0
    assert np.isfinite(result.mean_rps)


def test_compute_edges_and_kelly() -> None:
    model = pl.DataFrame(
        {
            "match_id": ["x1"],
            "model_home": [0.60],
            "model_draw": [0.25],
            "model_away": [0.15],
        }
    )
    market = pl.DataFrame(
        {
            "match_id": ["x1"],
            "home_win_prob": [0.50],  # 10pp model edge on home
            "draw_prob": [0.27],  # within threshold
            "away_win_prob": [0.23],  # 8pp edge on away (negative)
        }
    )
    edges = compute_edges(model, market, threshold=0.05)
    outcomes = set(edges["outcome"].to_list())
    assert outcomes == {"H", "A"}  # draw within threshold excluded
    home_edge = edges.filter(pl.col("outcome") == "H").row(0, named=True)
    assert home_edge["edge"] > 0.0
    assert home_edge["kelly_fraction"] > 0.0
    assert kelly_fraction(0.4, 0.5) == 0.0  # no positive edge -> no stake
