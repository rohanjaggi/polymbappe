"""Tests for the replay trajectory engine and champion-market P&L (eval/trajectory.py)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from polymbappe.eval.trajectory import (
    PNL_SCHEMA,
    TRAJECTORY_SCHEMA,
    compute_champion_market_pnl,
    compute_champion_trajectory,
    replay_dates,
)
from polymbappe.simulate.structure import placeholder_structure_2026
from polymbappe.simulate.tournament import WC2026_START, StrengthModel


def _matches(wc_days: list[date]) -> pl.DataFrame:
    rows = [
        {
            "match_id": "hist", "date": date(2024, 3, 1), "home_team": "X", "away_team": "Y",
            "home_goals": 1, "away_goals": 0, "competition": "Friendly",
            "is_knockout": False, "neutral_site": False, "group": None,
        }
    ]
    for i, d in enumerate(wc_days):
        rows.append(
            {
                "match_id": f"wc{i}", "date": d, "home_team": "X", "away_team": "Y",
                "home_goals": 2, "away_goals": 1, "competition": "FIFA World Cup",
                "is_knockout": False, "neutral_site": True, "group": None,
            }
        )
    return pl.DataFrame(rows)


def test_replay_dates_eve_plus_match_days() -> None:
    d1, d2 = WC2026_START, WC2026_START + timedelta(days=3)
    dates = replay_dates(_matches([d1, d2, d2]))  # duplicate day collapses
    assert dates == [WC2026_START - timedelta(days=1), d1, d2]
    # Pre-2026 history contributes no replay points.
    assert replay_dates(_matches([])) == [WC2026_START - timedelta(days=1)]


def _model(teams: list[str]) -> StrengthModel:
    attack = {t: 0.6 - 0.02 * i for i, t in enumerate(teams)}
    defense = {t: -0.3 + 0.01 * i for i, t in enumerate(teams)}
    return StrengthModel(attack=attack, defense=defense, home_advantage=0.0, rho=-0.03)


def test_trajectory_final_model_points_and_normalization() -> None:
    structure = placeholder_structure_2026()
    matches = _matches([WC2026_START, WC2026_START + timedelta(days=1)])
    out = compute_champion_trajectory(
        matches, pl.DataFrame(), structure,
        n_sims=100, refit=False, seed=7,
        fallback_model=_model(structure.teams),
    )
    assert out.schema == TRAJECTORY_SCHEMA
    # Eve + two match days = three replay points, 48 teams each.
    assert out["date"].n_unique() == 3
    assert out.height == 3 * 48
    # One champion (and two finalists) per simulated tournament at every cutoff.
    sums = out.group_by("date").agg(pl.col("champion").sum(), pl.col("FINAL").sum())
    assert all(abs(v - 1.0) < 1e-9 for v in sums["champion"].to_list())
    assert all(abs(v - 2.0) < 1e-9 for v in sums["FINAL"].to_list())


def test_trajectory_refit_false_requires_model() -> None:
    with pytest.raises(ValueError, match="fallback_model"):
        compute_champion_trajectory(
            _matches([]), pl.DataFrame(), placeholder_structure_2026(),
            refit=False, fallback_model=None,
        )


def test_champion_market_pnl_settles_at_resolution() -> None:
    from polymbappe.eval.market import kelly_fraction

    d1, d2 = date(2026, 6, 15), date(2026, 7, 1)
    trajectory = pl.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "team": ["Spain", "France", "Spain", "France"],
            "SF": [0.5, 0.4, 0.9, 0.1],
            "FINAL": [0.4, 0.3, 0.8, 0.05],
            "champion": [0.30, 0.20, 0.60, 0.02],
        }
    )
    market = pl.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "team": ["Spain", "France", "Spain", "France"],
            # Spain priced under the model both days (edges .10/.10); France over (no bet).
            "price": [0.20, 0.25, 0.50, 0.05],
        }
    )
    pnl, summary = compute_champion_market_pnl(
        trajectory, market, "Spain", edge_threshold=0.03, kelly_scale=1.0
    )
    assert pnl.schema == PNL_SCHEMA
    assert pnl["team"].unique().to_list() == ["Spain"]  # only positive-edge bets
    assert summary["n_bets"] == 2.0
    # Hand-check the first bet: stake = kelly(0.30, 0.20), all-in Yes at 0.20 pays 5x.
    first = pnl.sort("date").row(0, named=True)
    expected_stake = kelly_fraction(0.30, 0.20)
    assert first["stake"] == pytest.approx(expected_stake)
    assert first["payout"] == pytest.approx(expected_stake / 0.20)
    assert summary["total_profit"] == pytest.approx(
        float(pnl["payout"].sum() - pnl["stake"].sum())
    )
    # Losing side: had France been staked it would pay 0 — champion mismatch settles to 0.
    pnl_lose, summary_lose = compute_champion_market_pnl(
        trajectory, market, "France", edge_threshold=0.03, kelly_scale=1.0
    )
    assert float(pnl_lose["payout"].sum()) == 0.0
    assert summary_lose["total_profit"] == pytest.approx(-summary_lose["total_staked"])


def test_champion_market_pnl_empty_inputs() -> None:
    empty_traj = pl.DataFrame(schema=TRAJECTORY_SCHEMA)
    empty_market = pl.DataFrame(schema={"date": pl.Date, "team": pl.Utf8, "price": pl.Float64})
    pnl, summary = compute_champion_market_pnl(empty_traj, empty_market, "Spain")
    assert pnl.is_empty() and summary["n_bets"] == 0.0


def test_parse_team_yes_tokens_from_event_fixture() -> None:
    from polymbappe.polymarket.adapter import parse_team_yes_tokens

    event = {
        "markets": [
            {
                "groupItemTitle": "Spain",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["111", "222"]',
            },
            {
                "groupItemTitle": "Other",  # placeholder -> dropped
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["333", "444"]',
            },
            {
                "groupItemTitle": "France",
                "outcomes": '["No", "Yes"]',  # Yes not first: index respected
                "clobTokenIds": '["555", "666"]',
            },
            {
                "groupItemTitle": "Brazil",  # malformed: token count mismatch
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["777"]',
            },
        ]
    }
    tokens = parse_team_yes_tokens(event)
    assert tokens.sort("team").rows() == [("France", "666"), ("Spain", "111")]
