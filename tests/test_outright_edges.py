"""Tests for Polymarket futures parsing and outright edge detection."""

from __future__ import annotations

import polars as pl

from polymbappe.eval.market import compute_outright_edges
from polymbappe.polymarket.adapter import parse_team_yes_prices


def _winner_event() -> dict:
    return {
        "slug": "world-cup-winner",
        "markets": [
            {"groupItemTitle": "Spain", "outcomes": '["Yes", "No"]',
             "outcomePrices": '["0.17", "0.83"]'},
            {"groupItemTitle": "Czechia", "outcomes": '["Yes", "No"]',
             "outcomePrices": '["0.03", "0.97"]'},
            {"groupItemTitle": "Other", "outcomes": '["Yes", "No"]',
             "outcomePrices": None},  # skipped
            {"groupItemTitle": "Brazil", "outcomes": ["Yes", "No"],
             "outcomePrices": [0.08, 0.92]},
        ],
    }


def test_parse_team_yes_prices_aliases_and_skips() -> None:
    raw = parse_team_yes_prices(_winner_event(), normalize=False)
    teams = set(raw["team"].to_list())
    assert teams == {"Spain", "Czech Republic", "Brazil"}  # alias applied, "Other" dropped
    assert raw.filter(pl.col("team") == "Spain")["market_prob"].item() == 0.17


def test_parse_team_yes_prices_normalizes() -> None:
    norm = parse_team_yes_prices(_winner_event(), normalize=True)
    assert abs(norm["market_prob"].sum() - 1.0) < 1e-9  # de-vigged


def test_compute_outright_edges() -> None:
    model = pl.DataFrame(
        {"team": ["Spain", "Brazil", "France"], "model_prob": [0.25, 0.10, 0.12]}
    )
    market = pl.DataFrame(
        {"team": ["Spain", "Brazil", "France"], "market_prob": [0.17, 0.09, 0.13]}
    )
    edges = compute_outright_edges(model, market, threshold=0.03)
    # Spain: +8pp edge flagged; Brazil (+1pp) and France (-1pp) within threshold.
    assert edges["team"].to_list() == ["Spain"]
    row = edges.row(0, named=True)
    assert row["edge"] > 0 and row["kelly_fraction"] > 0
