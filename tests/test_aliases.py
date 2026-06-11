"""Tests for team-name normalization and its application across sources."""

from __future__ import annotations

import polars as pl

from polymbappe.data.aliases import normalize_team_expr, normalize_team_name
from polymbappe.data.normalize import normalize_footballdata_odds, normalize_kaggle_results
from polymbappe.polymarket.adapter import (
    align_polymarket_to_fixtures,
    normalize_polymarket_three_way,
)


def test_normalize_team_name_canonicalizes() -> None:
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("  usa  ") == "United States"
    assert normalize_team_name("Czechia") == "Czech Republic"
    assert normalize_team_name("Korea Republic") == "South Korea"
    # Unknown / already-canonical names pass through (trimmed).
    assert normalize_team_name("Spain") == "Spain"
    assert normalize_team_name(" Brazil ") == "Brazil"


def test_normalize_team_expr_on_frame() -> None:
    df = pl.DataFrame({"team": ["USA", "Spain", "Czechia"]})
    out = df.with_columns(normalize_team_expr("team").alias("team"))
    assert out["team"].to_list() == ["United States", "Spain", "Czech Republic"]


def test_results_normalization_applies_aliases() -> None:
    raw = pl.DataFrame(
        {
            "date": ["2023-06-01"], "home_team": ["USA"], "away_team": ["Czechia"],
            "home_score": [2], "away_score": [1], "tournament": ["Friendly"],
            "neutral": [False],
        }
    )
    out = normalize_kaggle_results(raw).row(0, named=True)
    assert out["home_team"] == "United States"
    assert out["away_team"] == "Czech Republic"
    assert out["match_id"] == "2023-06-01__United States__Czech Republic"


def test_footballdata_normalization_applies_aliases() -> None:
    fd = pl.DataFrame(
        {
            "Date": ["01/06/2024"], "HomeTeam": ["Czechia"], "AwayTeam": ["USA"],
            "B365H": [2.5], "B365D": [3.3], "B365A": [2.7],
        }
    )
    out = normalize_footballdata_odds(fd).row(0, named=True)
    assert out["match_id"] == "2024-06-01__Czech Republic__United States"


def test_polymarket_alignment_reconciles_spellings() -> None:
    # Polymarket uses "USA"; the fixture (from results) uses "United States".
    long = pl.DataFrame(
        {
            "market_id": ["m1"] * 3,
            "question": ["USA vs Spain"] * 3,
            "outcome": ["USA", "Draw", "Spain"],
            "price": [0.40, 0.26, 0.34],
        }
    )
    tw = normalize_polymarket_three_way(long)
    fixtures = pl.DataFrame(
        {
            "match_id": ["2026__United States__Spain"],
            "home_team": ["United States"],
            "away_team": ["Spain"],
        }
    )
    aligned = align_polymarket_to_fixtures(tw, fixtures)
    assert aligned.height == 1  # "USA" reconciled to "United States" -> joined
    row = aligned.row(0, named=True)
    assert row["match_id"] == "2026__United States__Spain"
