from datetime import date, timedelta

import numpy as np
import polars as pl

from polymbappe.eval.backtest import Tournament, run_leave_one_tournament_out, select_fixtures
from polymbappe.eval.base_probs import (
    BaseProbConfig,
    compute_tournament_base_probs,
    elo_probabilities,
)
from polymbappe.eval.market import compute_edges, kelly_fraction
from polymbappe.models.meta import MetaLearner

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
    a_vs_d = probs.join(fixtures, on="match_id").filter(
        (pl.col("home_team") == "A") & (pl.col("away_team") == "D")
    ).row(0, named=True)
    assert a_vs_d["dc_home"] > a_vs_d["dc_away"]
    assert a_vs_d["elo_home"] > a_vs_d["elo_away"]


def test_leave_one_tournament_out_runs_and_scores() -> None:
    result = run_leave_one_tournament_out(_make_matches(), TOURNAMENTS)
    assert set(result.per_tournament) == {"WC2016", "EU2018", "CA2020"}
    assert result.feature_columns == [
        "dc_home",
        "dc_draw",
        "dc_away",
        "elo_home",
        "elo_draw",
        "elo_away",
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
