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
    df = compute_knockout_bracket(
        _mini_schedule(), _no_results(), _model(teams), TournamentStructure(groups={})
    )

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
    df = compute_knockout_bracket(
        _mini_schedule(), _no_results(), _model(teams), TournamentStructure(groups={})
    )
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
    df = compute_knockout_bracket(
        _mini_schedule(), results, _model(teams), TournamentStructure(groups={})
    )
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


def test_bracket_drawn_tie_with_later_evidence_locks_inferred_winner() -> None:
    teams = ["A", "B", "C", "D"]
    results = pl.DataFrame(
        {
            "match_id": ["m1", "qf"], "date": [date(2026, 7, 4), date(2026, 7, 9)],
            "home_team": ["A", "B"], "away_team": ["B", "C"],
            "home_goals": [1, 2], "away_goals": [1, 0],
            "competition": ["FIFA World Cup"] * 2, "is_knockout": [True] * 2,
            "neutral_site": [True] * 2, "group": [None] * 2,
        },
        schema_overrides={"group": pl.Utf8},
    )
    df = compute_knockout_bracket(
        _mini_schedule(), results, _model(teams), TournamentStructure(groups={})
    )
    # B appearing in the played QF proves B won the drawn R16 on penalties.
    m1 = df.filter(pl.col("match_number") == 89)
    assert m1.height == 1
    row = m1.row(0, named=True)
    assert {row["team_a"], row["team_b"]} == {"A", "B"}
    winner_side = "p_a_advance" if row["team_a"] == "B" else "p_b_advance"
    assert row[winner_side] == 1.0


def test_bracket_unresolved_draw_emits_beyond_regulation_split() -> None:
    teams = ["A", "B", "C", "D"]
    results = pl.DataFrame(
        {
            "match_id": ["m1"], "date": [date(2026, 7, 4)],
            "home_team": ["A"], "away_team": ["B"], "home_goals": [1], "away_goals": [1],
            "competition": ["FIFA World Cup"], "is_knockout": [True],
            "neutral_site": [True], "group": [None],
        },
        schema_overrides={"group": pl.Utf8},
    )
    df = compute_knockout_bracket(
        _mini_schedule(), results, _model(teams), TournamentStructure(groups={})
    )
    m1 = df.filter(pl.col("match_number") == 89).row(0, named=True)
    # Regulation is known to have ended level: only ET/pens remain in the phase split.
    assert m1["p_decided_reg"] == 0.0
    assert abs(m1["p_decided_et"] + m1["p_decided_pens"] - 1.0) < 1e-9
    assert 0.0 < m1["p_a_advance"] < 1.0
    # Both teams stay possible occupants of the QF, weighted by the conditional split.
    qf = df.filter(pl.col("match_number") == 97)
    occupants = set(qf["team_a"].to_list()) | set(qf["team_b"].to_list())
    assert {"A", "B"} <= occupants


def _final_weekend_schedule() -> pl.DataFrame:
    """Two semi-finals feeding the final (W refs) and third place (L refs)."""

    return pl.DataFrame(
        {
            "match_id": ["sf1", "sf2", "tp", "final"],
            "date": [
                date(2026, 7, 14), date(2026, 7, 15), date(2026, 7, 18), date(2026, 7, 19),
            ],
            "stage": ["Semi-final", "Semi-final", "Match for third place", "Final"],
            "group": [None] * 4,
            "home_team": ["A", "C", "L101", "W101"],
            "away_team": ["B", "D", "L102", "W102"],
            "city": [None] * 4,
            "match_number": [101, 102, 103, 104],
        },
        schema_overrides={"group": pl.Utf8, "city": pl.Utf8, "match_number": pl.Int32},
    )


def test_bracket_forecasts_third_place_from_semi_loser_distributions() -> None:
    teams = ["A", "B", "C", "D"]
    df = compute_knockout_bracket(
        _final_weekend_schedule(), _no_results(), _model(teams), TournamentStructure(groups={})
    )
    tp = df.filter(pl.col("round") == "THIRD")
    # One row per possible SF-loser pairing: {A,B} x {C,D}.
    assert tp.height == 4
    assert abs(tp["matchup_prob"].sum() - 1.0) < 1e-9
    sides_a = set(tp["team_a"].to_list())
    sides_b = set(tp["team_b"].to_list())
    assert sides_a == {"A", "B"} and sides_b == {"C", "D"}


def test_bracket_locks_played_third_place() -> None:
    teams = ["A", "B", "C", "D"]
    results = pl.DataFrame(
        {
            "match_id": ["sf1", "sf2", "tp"],
            "date": [date(2026, 7, 14), date(2026, 7, 15), date(2026, 7, 18)],
            "home_team": ["A", "C", "B"], "away_team": ["B", "D", "D"],
            "home_goals": [2, 1, 3], "away_goals": [0, 0, 1],
            "competition": ["FIFA World Cup"] * 3, "is_knockout": [True] * 3,
            "neutral_site": [True] * 3, "group": [None] * 3,
        },
        schema_overrides={"group": pl.Utf8},
    )
    df = compute_knockout_bracket(
        _final_weekend_schedule(), results, _model(teams), TournamentStructure(groups={})
    )
    tp = df.filter(pl.col("round") == "THIRD")
    assert tp.height == 1
    row = tp.row(0, named=True)
    assert {row["team_a"], row["team_b"]} == {"B", "D"}
    winner_side = "p_a_advance" if row["team_a"] == "B" else "p_b_advance"
    assert row[winner_side] == 1.0
    # The final projects the two SF winners.
    final = df.filter(pl.col("round") == "FINAL")
    assert final.height == 1
    assert {final.row(0, named=True)["team_a"], final.row(0, named=True)["team_b"]} == {"A", "C"}


def test_bracket_resolves_position_placeholders_from_complete_groups() -> None:
    teams = ["W", "X", "Y", "Z"]
    structure = TournamentStructure(groups={"A": teams})
    pairs = [("W", "X"), ("W", "Y"), ("W", "Z"), ("X", "Y"), ("X", "Z"), ("Y", "Z")]
    results = pl.DataFrame(
        {
            "match_id": [f"g{i}" for i in range(6)],
            "date": [date(2026, 6, 15)] * 6,
            "home_team": [h for h, _ in pairs],
            "away_team": [a for _, a in pairs],
            # W wins all, X beats Y/Z, Y beats Z: table order is W, X, Y, Z.
            "home_goals": [2, 2, 2, 2, 2, 2],
            "away_goals": [0, 0, 0, 0, 0, 0],
            "competition": ["FIFA World Cup"] * 6,
            "is_knockout": [False] * 6,
            "neutral_site": [True] * 6,
            "group": [None] * 6,
        },
        schema_overrides={"group": pl.Utf8},
    )
    schedule = pl.DataFrame(
        {
            "match_id": ["r32"],
            "date": [date(2026, 6, 28)],
            "stage": ["Round of 32"],
            "group": [None],
            "home_team": ["1A"],
            "away_team": ["2A"],
            "city": [None],
        },
        schema_overrides={"group": pl.Utf8, "city": pl.Utf8},
    )
    df = compute_knockout_bracket(schedule, results, _model(teams), structure)
    row = df.row(0, named=True)
    # Placeholders resolve to the real group standings, not literal "1A"/"2A" teams.
    assert (row["team_a"], row["team_b"]) == ("W", "X")
