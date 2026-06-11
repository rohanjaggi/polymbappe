"""Tests for the contextual feature builders and the residual adjuster."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polymbappe.context.cohesion import build_cohesion_features, club_cluster_index
from polymbappe.context.draw_pressure import (
    draw_pressure_features,
    low_scoring_probability,
    mutual_qualification_incentive,
    stage_elo_interaction,
)
from polymbappe.context.fatigue import (
    add_fatigue_flag,
    build_season_load_features,
    build_travel_features,
    haversine_km,
    venue_distance,
)
from polymbappe.context.manager import ManagerConfig, build_manager_features, shrink
from polymbappe.context.ppda import build_ppda_features, ppda_difference, ppda_similarity
from polymbappe.context.sentiment import build_xg_overperformance, score_text_vader


def _matches() -> pl.DataFrame:
    rows = []
    day = date(2020, 1, 1)
    for i in range(12):
        rows.append(
            {
                "match_id": f"m{i}", "date": day + timedelta(days=i * 7),
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_goals": 2, "away_goals": 1,
                "competition": "Friendly", "is_knockout": False,
                "neutral_site": False, "group": None,
            }
        )
    return pl.DataFrame(rows)


# -- PPDA ----------------------------------------------------------------------

def test_ppda_difference_and_similarity() -> None:
    assert ppda_difference(8.0, 12.0) == -4.0
    assert ppda_difference(None, 12.0) is None
    assert ppda_similarity(10.0, 10.0) == 1.0
    assert ppda_similarity(0.0, 20.0) == 0.0
    assert ppda_similarity(None, 5.0) is None


def test_build_ppda_features_proxy_and_real() -> None:
    matches = _matches()
    proxy = build_ppda_features(matches)
    assert proxy["ppda_available"].to_list() == [False] * proxy.height
    team_ppda = pl.DataFrame(
        {"team": ["A"] * 6, "date": matches["date"][:6], "ppda": [10.0, 9, 11, 8, 12, 10]}
    )
    real = build_ppda_features(matches, team_ppda)
    assert real.filter(pl.col("ppda_available"))["ppda"].null_count() == 0


# -- cohesion ------------------------------------------------------------------

def test_club_cluster_index() -> None:
    assert club_cluster_index({"X": 3, "Y": 2, "Z": 1}) == 3 + 1 + 0


def test_build_cohesion_features() -> None:
    squads = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B"],
            "tournament": ["2026"] * 5,
            "player": ["p1", "p2", "p3", "p4", "p5"],
            "club": ["City", "City", "Madrid", "PSG", None],
            "age": [27, 29, 31, 25, None],
        }
    )
    out = build_cohesion_features(squads).sort("team")
    a = out.filter(pl.col("team") == "A").row(0, named=True)
    assert a["club_cluster_index"] == 1  # City pair (2 players) -> 1, Madrid -> 0
    assert a["median_age"] == 29.0
    assert a["player_count"] == 3


# -- manager -------------------------------------------------------------------

def test_shrink_pulls_toward_prior() -> None:
    # 1 win in 1 match, prior 0.5 with prior_n=4 -> well below 1.0.
    assert shrink(1.0, 1.0, 0.5, 4.0) == pytest.approx((1 + 2.0) / 5.0)


def test_build_manager_features_shrinkage_and_recency() -> None:
    records = pl.DataFrame(
        {
            "manager": ["X", "X", "Y"],
            "team": ["A", "A", "B"],
            "tournament": ["2018", "2022", "2022"],
            "tournament_order": [1, 2, 2],
            "stage_reached": ["QF", "FINAL", "R16"],
            "knockout_matches": [3, 6, 2],
            "knockout_wins": [2, 5, 0],
        }
    )
    out = build_manager_features(records, ManagerConfig()).sort("manager")
    x = out.filter(pl.col("manager") == "X").row(0, named=True)
    y = out.filter(pl.col("manager") == "Y").row(0, named=True)
    # X has a strong, deep record; Y none -> X ranks higher on both signals.
    assert x["knockout_win_rate"] > y["knockout_win_rate"]
    assert x["deepest_run_weighted"] > y["deepest_run_weighted"]
    assert x["tenure_matches"] == 9


# -- fatigue -------------------------------------------------------------------

def test_haversine_and_venue_distance() -> None:
    # NY to LA is ~3900 km.
    d = venue_distance("New York", "Los Angeles")
    assert d is not None and 3500 < d < 4300
    assert haversine_km((0, 0), (0, 0)) == 0.0
    assert venue_distance("New York", "Atlantis") is None


def test_build_travel_and_load_and_flag() -> None:
    schedule = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "date": [date(2026, 6, 11), date(2026, 6, 16), date(2026, 6, 21)],
            "match_id": ["g1", "g2", "g3"],
            "venue": ["New York", "Los Angeles", "New York"],
        }
    )
    travel = build_travel_features(schedule).sort("match_id")
    assert travel.filter(pl.col("match_id") == "g1")["travel_km"].item() == 0.0
    assert travel.filter(pl.col("match_id") == "g2")["travel_km"].item() > 3000

    minutes = pl.DataFrame(
        {"team": ["A", "B", "C"], "tournament": ["2026"] * 3,
         "season_minutes": [3000.0, 2000.0, 1000.0]}
    )
    load = build_season_load_features(minutes)
    assert load.filter(pl.col("team") == "A")["season_load"].item() > 0

    rest = pl.DataFrame({"match_id": ["x"], "team": ["A"], "rest_days": [3]})
    flagged = add_fatigue_flag(rest)
    assert flagged["fatigued"].item() is True


# -- draw pressure -------------------------------------------------------------

def test_draw_pressure_components() -> None:
    assert mutual_qualification_incentive(True, True, True) == 1
    assert mutual_qualification_incentive(True, True, False) == 0
    assert mutual_qualification_incentive(False, True, True) == 0

    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0.3
    matrix[1, 0] = 0.2
    matrix[0, 1] = 0.2
    matrix[2, 2] = 0.3
    assert low_scoring_probability(matrix) == pytest.approx(0.7)

    # Group stage, small gap -> positive; knockout -> negative.
    assert stage_elo_interaction(False, 20.0) > 0
    assert stage_elo_interaction(True, 20.0) < 0
    assert stage_elo_interaction(False, 500.0) == 0.0


def test_draw_pressure_features_dict() -> None:
    matrix = np.full((4, 4), 1 / 16)
    feats = draw_pressure_features(
        is_final_matchday=True, draw_qualifies_home=True, draw_qualifies_away=True,
        home_ppda=10.0, away_ppda=11.0, score_matrix=matrix,
        is_knockout=False, elo_gap=50.0,
    )
    assert feats["mutual_qual_incentive"] == 1.0
    assert 0.0 <= feats["ppda_similarity"] <= 1.0
    assert set(feats) == {
        "mutual_qual_incentive", "ppda_similarity", "low_scoring_prob",
        "stage_elo_interaction",
    }


# -- sentiment -----------------------------------------------------------------

def test_xg_overperformance_zero_under_proxy() -> None:
    # With no real xG, the proxy makes overperformance ~0 (goals - goals proxy).
    out = build_xg_overperformance(_matches())
    nonnull = out["xg_overperformance"].drop_nulls().to_numpy()
    assert np.allclose(nonnull, 0.0, atol=1e-9)


def test_score_text_vader_graceful() -> None:
    assert score_text_vader([]) == 0.0
    # Returns a float in [-1, 1] whether or not vader is installed.
    val = score_text_vader(["great win", "terrible loss"])
    assert -1.0 <= val <= 1.0
