"""Tests for the real-draw knockout-bracket forecast (``simulate.knockout_bracket``)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from polymbappe.simulate.knockout_bracket import BRACKET_SCHEMA, compute_knockout_bracket
from polymbappe.simulate.tournament import StrengthModel, TournamentStructure


def _model(teams: list[str]) -> StrengthModel:
    # Descending strength so favourites are well-defined (team A strongest).
    attack = {t: 0.6 - 0.05 * i for i, t in enumerate(teams)}
    defense = {t: -0.3 + 0.02 * i for i, t in enumerate(teams)}
    return StrengthModel(attack=attack, defense=defense, home_advantage=0.0, rho=-0.03)


def _mini_schedule() -> pl.DataFrame:
    """A tiny 4-team bracket: two R16 (here labelled ``Round of 16``) feeding one QF via W##.

    R16 matches are numbered 89, 90 by chronological order; the QF references ``W89``/``W90``.
    """

    return pl.DataFrame(
        {
            "match_id": ["m1", "m2", "qf"],
            "date": [date(2026, 7, 4), date(2026, 7, 5), date(2026, 7, 9)],
            "stage": ["Round of 16", "Round of 16", "Quarter-final"],
            "group": [None, None, None],
            "home_team": ["A", "C", "W89"],
            "away_team": ["B", "D", "W90"],
            "city": [None, None, None],
        },
        schema_overrides={"group": pl.Utf8, "city": pl.Utf8},
    )


def _no_results() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "match_id": pl.Utf8, "date": pl.Date, "home_team": pl.Utf8, "away_team": pl.Utf8,
            "home_goals": pl.Int64, "away_goals": pl.Int64, "competition": pl.Utf8,
            "is_knockout": pl.Boolean, "neutral_site": pl.Boolean, "group": pl.Utf8,
        }
    )


def test_bracket_schema_and_probability_partitions() -> None:
    teams = ["A", "B", "C", "D"]
    df = compute_knockout_bracket(_mini_schedule(), _no_results(), _model(teams), TournamentStructure(groups={}))

    assert df.columns == list(BRACKET_SCHEMA.keys())
    # R16 fixtures are concrete (one matchup each, prob 1); the QF fans out into 2x2 = 4.
    r16 = df.filter(pl.col("round") == "R16")
    qf = df.filter(pl.col("round") == "QF")
    assert r16.height == 2 and np.allclose(r16["matchup_prob"].to_numpy(), 1.0)
    assert qf.height == 4
    # The 4 possible QF matchups' occurrence probabilities sum to 1.
    assert abs(qf["matchup_prob"].sum() - 1.0) < 1e-9
    # Advance and decided-phase probabilities each partition every tie.
    adv = df.select("p_a_advance", "p_b_advance").to_numpy()
    assert np.allclose(adv.sum(axis=1), 1.0, atol=1e-9)
    phase = df.select("p_decided_reg", "p_decided_et", "p_decided_pens").to_numpy()
    assert np.allclose(phase.sum(axis=1), 1.0, atol=1e-9)


def test_bracket_most_likely_qf_pairs_favourites() -> None:
    teams = ["A", "B", "C", "D"]  # A strongest, D weakest
    df = compute_knockout_bracket(_mini_schedule(), _no_results(), _model(teams), TournamentStructure(groups={}))
    qf = df.filter(pl.col("round") == "QF").sort("rank")
    top = qf.row(0, named=True)
    # The likeliest QF is the two R16 favourites: A (beats B) vs C (beats D).
    assert {top["team_a"], top["team_b"]} == {"A", "C"}


def test_bracket_locks_played_results() -> None:
    teams = ["A", "B", "C", "D"]
    results = pl.DataFrame(
        {
            "match_id": ["m1"], "date": [date(2026, 7, 4)],
            "home_team": ["A"], "away_team": ["B"], "home_goals": [0], "away_goals": [2],
            "competition": ["FIFA World Cup"], "is_knockout": [True],
            "neutral_site": [True], "group": [None],
        },
        schema_overrides={"group": pl.Utf8},
    )
    df = compute_knockout_bracket(_mini_schedule(), results, _model(teams), TournamentStructure(groups={}))
    qf = df.filter(pl.col("round") == "QF")
    # B won its R16 (2-0 upset), so every possible QF now has B on one side.
    assert all("B" in {r["team_a"], r["team_b"]} for r in qf.iter_rows(named=True))
    # B is locked (prob 1 through that side) -> only 2 QF matchups remain (B vs C, B vs D).
    assert qf.height == 2


def test_bracket_empty_schedule_returns_typed_empty() -> None:
    df = compute_knockout_bracket(
        pl.DataFrame(schema={"stage": pl.Utf8}), _no_results(),
        _model(["A", "B"]), TournamentStructure(groups={}),
    )
    assert df.is_empty()
    assert df.columns == list(BRACKET_SCHEMA.keys())
