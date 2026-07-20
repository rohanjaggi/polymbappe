"""Tests for the dashboard data-layer helpers added in the design pass."""

from __future__ import annotations

import polars as pl

from polymbappe.dashboard import data


def _stage_frame(rows: list[dict[str, float | str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=data.STAGE_SCHEMA)


def test_champion_team_none_while_open() -> None:
    df = _stage_frame(
        [
            {"team": "Spain", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 0.55},
            {"team": "Argentina", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 0.45},
        ]
    )
    assert data.champion_team(df) is None


def test_champion_team_decided() -> None:
    df = _stage_frame(
        [
            {"team": "Spain", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 1.0},
            {"team": "Argentina", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 0.0},
        ]
    )
    assert data.champion_team(df) == "Spain"


def test_champion_team_empty() -> None:
    assert data.champion_team(pl.DataFrame(schema=data.STAGE_SCHEMA)) is None


def test_final_standings_orders_by_depth() -> None:
    df = _stage_frame(
        [
            {"team": "GroupExit", "R32": 0.0, "R16": 0.0, "QF": 0.0, "SF": 0.0,
             "FINAL": 0.0, "champion": 0.0},
            {"team": "Spain", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 1.0},
            {"team": "Argentina", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 1.0, "champion": 0.0},
            {"team": "England", "R32": 1.0, "R16": 1.0, "QF": 1.0, "SF": 1.0,
             "FINAL": 0.0, "champion": 0.0},
            {"team": "Brazil", "R32": 1.0, "R16": 1.0, "QF": 0.0, "SF": 0.0,
             "FINAL": 0.0, "champion": 0.0},
        ]
    )
    standings = data.final_standings(df)
    assert standings.columns == ["team", "result"]
    assert standings["team"].to_list() == [
        "Spain", "Argentina", "England", "Brazil", "GroupExit",
    ]
    assert standings["result"].to_list() == [
        "🏆 Champions", "Runners-up", "Semi-finalists", "Round of 16", "Group stage",
    ]


def test_final_standings_empty() -> None:
    standings = data.final_standings(pl.DataFrame(schema=data.STAGE_SCHEMA))
    assert standings.is_empty()
    assert standings.columns == ["team", "result"]
