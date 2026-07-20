"""Tests for the tournament retrospective: per-round accuracy and the markdown builder."""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.dashboard import data
from polymbappe.eval.retrospective import build_retrospective_markdown


def _match_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": ["2026__A__B", "2026__C__D", "2026__A__C"],
            "group": ["G", "G", "KO"],
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "D", "C"],
            "model_home": [0.6, 0.2, 0.5],
            "model_draw": [0.25, 0.3, 0.3],
            "model_away": [0.15, 0.5, 0.2],
        }
    )


def _results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "D", "C"],
            "date": [date(2026, 6, 12), date(2026, 6, 13), date(2026, 6, 29)],
            "home_goals": [2, 0, 1],  # A wins (hit), C-D away win (hit), A wins KO (hit)
            "away_goals": [0, 1, 0],
            "competition": ["FIFA World Cup"] * 3,
        }
    )


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": ["r32"],
            "date": [date(2026, 6, 28)],
            "stage": ["Round of 32"],
            "group": [None],
            "home_team": ["1A"],
            "away_team": ["2B"],
            "city": [None],
        },
        schema_overrides={"group": pl.Utf8, "city": pl.Utf8},
    )


def test_per_round_accuracy_splits_group_and_ko() -> None:
    table = data.per_round_accuracy(_match_df(), _results(), _schedule())
    rows = {r["round"]: r for r in table.iter_rows(named=True)}
    assert rows["Group"]["n"] == 2 and rows["Group"]["accuracy"] == 1.0
    # A-C on 06-29 is past the R32 cutoff (06-28) -> classified R32, model pick correct.
    assert rows["R32"]["n"] == 1 and rows["R32"]["accuracy"] == 1.0
    # Mean P(actual): group = (0.6 + 0.5) / 2, R32 = 0.5.
    assert abs(rows["Group"]["avg_p_actual"] - 0.55) < 1e-9
    assert abs(rows["R32"]["avg_p_actual"] - 0.5) < 1e-9


def test_per_round_accuracy_empty_inputs() -> None:
    empty = pl.DataFrame(schema={"match_id": pl.Utf8})
    assert data.per_round_accuracy(empty, _results(), _schedule()).is_empty()


def _trajectory() -> pl.DataFrame:
    d = [date(2026, 6, 10), date(2026, 6, 20), date(2026, 7, 18), date(2026, 7, 19)]
    rows = []
    for i, day in enumerate(d):
        rows.append({"date": day, "team": "Spain", "SF": 0.5, "FINAL": 0.4,
                     "champion": [0.20, 0.30, 0.55, 1.0][i]})
        rows.append({"date": day, "team": "France", "SF": 0.4, "FINAL": 0.3,
                     "champion": [0.15, 0.20, 0.0, 0.0][i]})
    return pl.DataFrame(rows)


def test_build_retrospective_markdown_full() -> None:
    scorecard = {
        "n": 104.0, "accuracy": 0.70, "brier_score": 0.44, "log_loss": 0.76,
        "rps": 0.19, "rps_skill": 0.15, "log_loss_skill": 0.25, "brier_skill": 0.34,
    }
    per_round = pl.DataFrame(
        {"round": ["Group", "F"], "n": [72, 1],
         "accuracy": [0.639, 1.0], "avg_p_actual": [0.498, 0.6]},
        schema_overrides={"n": pl.Int64},
    )
    upsets = pl.DataFrame(
        {
            "Fixture": ["Brazil vs Norway"], "Score": ["1 – 2"], "Model Pick": ["Brazil"],
            "Pick Confidence": ["62%"], "Actual Result": ["Norway"], "P(Actual)": ["12%"],
            "Upset Magnitude": ["88%"],
        }
    )
    bookmaker = {
        "available": True, "n_overlap": 90.0, "model_accuracy": 0.70,
        "book_accuracy": 0.67, "mcnemar_p": 0.42,
    }
    pnl = pl.DataFrame(
        {
            "date": [date(2026, 6, 20)], "team": ["Spain"], "model_prob": [0.30],
            "market_price": [0.20], "edge": [0.10], "stake": [0.125],
            "payout": [0.625], "profit": [0.5],
        }
    )
    md = build_retrospective_markdown(
        scorecard, per_round, upsets, _trajectory(), bookmaker, pnl
    )
    assert "# FIFA World Cup 2026 — Tournament Retrospective" in md
    assert "**104 matches**" in md and "70.0%" in md
    assert "| Group stage | 72 |" in md and "| Final | 1 |" in md
    assert "**Champion: Spain.**" in md
    assert "Brazil vs Norway" in md
    assert "McNemar's test" in md
    assert "+0.500 units" in md


def test_build_retrospective_markdown_degrades() -> None:
    scorecard = {
        "n": 10.0, "accuracy": 0.5, "brier_score": 0.5, "log_loss": 0.9,
        "rps": 0.2, "rps_skill": 0.1, "log_loss_skill": 0.1, "brier_skill": 0.1,
    }
    empty_upsets = pl.DataFrame(schema={"Fixture": pl.Utf8})
    empty_traj = pl.DataFrame(
        schema={"date": pl.Date, "team": pl.Utf8, "SF": pl.Float64,
                "FINAL": pl.Float64, "champion": pl.Float64}
    )
    md = build_retrospective_markdown(
        scorecard,
        pl.DataFrame(schema={"round": pl.Utf8, "n": pl.Int64,
                             "accuracy": pl.Float64, "avg_p_actual": pl.Float64}),
        empty_upsets,
        empty_traj,
        {"available": False, "reason": "No bookmaker accuracy workbook found."},
        pl.DataFrame(schema={"stake": pl.Float64, "profit": pl.Float64}),
    )
    assert "_No result fell below 25% model probability._" in md
    assert "No bookmaker accuracy workbook found." in md
    assert "no P&L backtest was run" in md
    assert "title race" not in md  # trajectory section skipped entirely
