"""Tests for the shared real-knockout-tree machinery (simulate/real_bracket.py)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from polymbappe.simulate.match import (
    beyond_regulation_home_winprob,
    score_matrix_from_rates,
)
from polymbappe.simulate.real_bracket import (
    ROUND_BASE,
    assign_thirds,
    attach_played_results,
    build_real_bracket,
    fill_r32_leaves,
    parse_side,
)

# The real 2026 third-place constraint sets (openfootball R32 fixtures).
_REAL_THIRD_SLOTS = [
    (74, frozenset("ABCDF")),
    (77, frozenset("CDFGH")),
    (78, frozenset("CEFHI")),
    (80, frozenset("EHIJK")),
    (81, frozenset("BEFIJ")),
    (82, frozenset("AEHIJ")),
    (83, frozenset("EFGIJ")),
    (87, frozenset("DEIJL")),
]


def _schedule(rows: list[dict], with_numbers: bool = True) -> pl.DataFrame:
    schema = {
        "match_id": pl.Utf8,
        "date": pl.Date,
        "stage": pl.Utf8,
        "group": pl.Utf8,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "city": pl.Utf8,
        "match_number": pl.Int32,
    }
    full = [
        {
            "match_id": f"{r['date']}__{r['home_team']}__{r['away_team']}",
            "group": None,
            "city": r.get("city", ""),
            "match_number": r.get("match_number"),
            **{k: r[k] for k in ("date", "stage", "home_team", "away_team")},
        }
        for r in rows
    ]
    df = pl.DataFrame(full, schema=schema)
    return df if with_numbers else df.drop("match_number")


def _matches(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "match_id": f"{r['date']}__{r['home_team']}__{r['away_team']}",
                "competition": "FIFA World Cup",
                "is_knockout": True,
                "neutral_site": True,
                "group": None,
                "country": "",
                "city": r.get("city", ""),
                **r,
            }
            for r in rows
        ]
    )


def test_parse_side_variants() -> None:
    ref = parse_side("W73")
    assert ref.kind == "winner" and ref.ref == 73
    loser = parse_side("L101")
    assert loser.kind == "loser" and loser.ref == 101
    pos = parse_side("2B")
    assert pos.kind == "position" and pos.rank == 2 and pos.group == "B"
    third = parse_side("3A/B/C/D/F")
    assert third.kind == "third" and third.allowed_groups == frozenset("ABCDF")
    team = parse_side("Brazil")
    assert team.kind == "team" and team.team == "Brazil"


def test_build_real_bracket_honors_match_number_over_string_order() -> None:
    # Three same-date R32 fixtures whose real FIFA numbering disagrees with string sort.
    sched = _schedule(
        [
            {"date": date(2026, 6, 29), "stage": "Round of 32", "home_team": "1E",
             "away_team": "3A/B/C/D/F", "match_number": 74},
            {"date": date(2026, 6, 29), "stage": "Round of 32", "home_team": "1F",
             "away_team": "2C", "match_number": 75},
            {"date": date(2026, 6, 29), "stage": "Round of 32", "home_team": "1C",
             "away_team": "2F", "match_number": 76},
        ]
    )
    bracket = build_real_bracket(sched)
    assert bracket is not None
    assert bracket.nodes[74].home.label == "1E"
    assert bracket.nodes[75].home.label == "1F"
    assert bracket.nodes[76].home.label == "1C"


def test_build_real_bracket_fallback_numbering_is_chronological() -> None:
    sched = _schedule(
        [
            {"date": date(2026, 6, 29), "stage": "Round of 32",
             "home_team": "1B", "away_team": "2A"},
            {"date": date(2026, 6, 28), "stage": "Round of 32",
             "home_team": "1A", "away_team": "2B"},
            {"date": date(2026, 7, 4), "stage": "Round of 16",
             "home_team": "W73", "away_team": "W74"},
        ],
        with_numbers=False,
    )
    bracket = build_real_bracket(sched)
    assert bracket is not None
    assert bracket.nodes[ROUND_BASE["R32"]].home.label == "1A"  # earlier date -> 73
    assert bracket.nodes[ROUND_BASE["R32"] + 1].home.label == "1B"
    assert bracket.nodes[ROUND_BASE["R16"]].home.ref == 73
    assert bracket.consumer_of(73) == (ROUND_BASE["R16"], "home")


def test_build_real_bracket_empty_or_groupstage_only() -> None:
    assert build_real_bracket(pl.DataFrame()) is None
    group_only = _schedule(
        [{"date": date(2026, 6, 12), "stage": "Matchday 1",
          "home_team": "Mexico", "away_team": "Chile"}]
    )
    assert build_real_bracket(group_only) is None


def _mini_bracket() -> pl.DataFrame:
    """Two R32 ties feeding one R16 tie."""

    return _schedule(
        [
            {"date": date(2026, 6, 28), "stage": "Round of 32", "home_team": "1A",
             "away_team": "2B", "city": "Dallas (Arlington)", "match_number": 73},
            {"date": date(2026, 6, 29), "stage": "Round of 32", "home_team": "1B",
             "away_team": "2A", "city": "Boston (Foxborough)", "match_number": 74},
            {"date": date(2026, 7, 4), "stage": "Round of 16", "home_team": "W73",
             "away_team": "W74", "city": "Houston", "match_number": 89},
        ]
    )


def test_attach_pins_by_city_token_and_locks_decisive_winner() -> None:
    bracket = build_real_bracket(_mini_bracket())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 6, 28), "home_team": "Spain", "away_team": "Austria",
             "home_goals": 3, "away_goals": 0, "city": "Arlington"},
        ]
    )
    attach_played_results(bracket, matches)
    node = bracket.nodes[73]
    assert (node.pinned_home, node.pinned_away) == ("Spain", "Austria")
    assert node.winner == "Spain" and node.loser == "Austria"
    assert not node.drawn_unresolved


def test_attach_infers_drawn_tie_winner_from_later_round() -> None:
    bracket = build_real_bracket(_mini_bracket())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 6, 28), "home_team": "Germany", "away_team": "Paraguay",
             "home_goals": 1, "away_goals": 1, "city": "Arlington"},
            {"date": date(2026, 6, 29), "home_team": "France", "away_team": "Sweden",
             "home_goals": 3, "away_goals": 0, "city": "Foxborough"},
            {"date": date(2026, 7, 4), "home_team": "Paraguay", "away_team": "France",
             "home_goals": 0, "away_goals": 1, "city": "Houston"},
        ]
    )
    attach_played_results(bracket, matches)
    node = bracket.nodes[73]
    assert node.winner == "Paraguay" and node.loser == "Germany"
    assert not node.drawn_unresolved


def test_attach_drawn_tie_without_later_evidence_is_unresolved() -> None:
    bracket = build_real_bracket(_mini_bracket())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 6, 28), "home_team": "Germany", "away_team": "Paraguay",
             "home_goals": 1, "away_goals": 1, "city": "Arlington"},
        ]
    )
    attach_played_results(bracket, matches)
    node = bracket.nodes[73]
    assert node.winner is None and node.drawn_unresolved


def test_attach_resolves_drawn_semi_via_third_place_match() -> None:
    sched = _schedule(
        [
            {"date": date(2026, 7, 14), "stage": "Semi-final", "home_team": "Alpha",
             "away_team": "Beta", "city": "Dallas (Arlington)", "match_number": 101},
            {"date": date(2026, 7, 15), "stage": "Semi-final", "home_team": "Gamma",
             "away_team": "Delta", "city": "Atlanta", "match_number": 102},
            {"date": date(2026, 7, 18), "stage": "Match for third place",
             "home_team": "L101", "away_team": "L102", "city": "Miami (Miami Gardens)",
             "match_number": 103},
            {"date": date(2026, 7, 19), "stage": "Final", "home_team": "W101",
             "away_team": "W102", "city": "New York/New Jersey (East Rutherford)",
             "match_number": 104},
        ]
    )
    bracket = build_real_bracket(sched)
    assert bracket is not None and bracket.third_place is not None
    matches = _matches(
        [
            {"date": date(2026, 7, 14), "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 2, "away_goals": 2, "city": "Arlington"},
            {"date": date(2026, 7, 15), "home_team": "Gamma", "away_team": "Delta",
             "home_goals": 1, "away_goals": 0, "city": "Atlanta"},
            {"date": date(2026, 7, 18), "home_team": "Beta", "away_team": "Delta",
             "home_goals": 1, "away_goals": 0, "city": "Miami Gardens"},
        ]
    )
    attach_played_results(bracket, matches)
    semi = bracket.nodes[101]
    assert semi.winner == "Alpha" and semi.loser == "Beta"


def _final_weekend_schedule() -> pl.DataFrame:
    return _schedule(
        [
            {"date": date(2026, 7, 14), "stage": "Semi-final", "home_team": "Alpha",
             "away_team": "Beta", "city": "Dallas (Arlington)", "match_number": 101},
            {"date": date(2026, 7, 15), "stage": "Semi-final", "home_team": "Gamma",
             "away_team": "Delta", "city": "Atlanta", "match_number": 102},
            {"date": date(2026, 7, 18), "stage": "Match for third place",
             "home_team": "L101", "away_team": "L102", "city": "Miami (Miami Gardens)",
             "match_number": 103},
            {"date": date(2026, 7, 19), "stage": "Final", "home_team": "W101",
             "away_team": "W102", "city": "New York/New Jersey (East Rutherford)",
             "match_number": 104},
        ]
    )


def test_drawn_final_is_unresolved_without_override() -> None:
    bracket = build_real_bracket(_final_weekend_schedule())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 7, 14), "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 2, "away_goals": 0, "city": "Arlington"},
            {"date": date(2026, 7, 15), "home_team": "Gamma", "away_team": "Delta",
             "home_goals": 1, "away_goals": 0, "city": "Atlanta"},
            {"date": date(2026, 7, 19), "home_team": "Alpha", "away_team": "Gamma",
             "home_goals": 1, "away_goals": 1, "city": "East Rutherford"},
        ]
    )
    attach_played_results(bracket, matches)
    final = bracket.nodes[104]
    assert final.winner is None and final.drawn_unresolved  # no later round to look at


def test_winner_override_resolves_drawn_final() -> None:
    bracket = build_real_bracket(_final_weekend_schedule())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 7, 14), "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 2, "away_goals": 0, "city": "Arlington"},
            {"date": date(2026, 7, 15), "home_team": "Gamma", "away_team": "Delta",
             "home_goals": 1, "away_goals": 0, "city": "Atlanta"},
            {"date": date(2026, 7, 19), "home_team": "Alpha", "away_team": "Gamma",
             "home_goals": 1, "away_goals": 1, "city": "East Rutherford"},
        ]
    )
    attach_played_results(bracket, matches, winner_overrides={104: "Gamma"})
    final = bracket.nodes[104]
    assert final.winner == "Gamma" and final.loser == "Alpha"
    assert not final.drawn_unresolved


def test_winner_override_ignores_team_not_in_tie() -> None:
    bracket = build_real_bracket(_final_weekend_schedule())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 7, 19), "home_team": "Alpha", "away_team": "Gamma",
             "home_goals": 0, "away_goals": 0, "city": "East Rutherford"},
        ]
    )
    attach_played_results(bracket, matches, winner_overrides={104: "Zeta"})
    final = bracket.nodes[104]
    assert final.winner is None and final.drawn_unresolved


def test_winner_override_resolves_drawn_third_place() -> None:
    bracket = build_real_bracket(_final_weekend_schedule())
    assert bracket is not None
    matches = _matches(
        [
            {"date": date(2026, 7, 14), "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 2, "away_goals": 0, "city": "Arlington"},
            {"date": date(2026, 7, 15), "home_team": "Gamma", "away_team": "Delta",
             "home_goals": 1, "away_goals": 0, "city": "Atlanta"},
            {"date": date(2026, 7, 18), "home_team": "Beta", "away_team": "Delta",
             "home_goals": 2, "away_goals": 2, "city": "Miami Gardens"},
        ]
    )
    attach_played_results(bracket, matches)
    tp = bracket.third_place
    assert tp is not None and tp.winner is None and tp.drawn_unresolved

    bracket2 = build_real_bracket(_final_weekend_schedule())
    assert bracket2 is not None
    attach_played_results(bracket2, matches, winner_overrides={103: "Delta"})
    tp2 = bracket2.third_place
    assert tp2 is not None and tp2.winner == "Delta" and tp2.loser == "Beta"
    assert not tp2.drawn_unresolved


def test_load_ko_winner_overrides_roundtrip(tmp_path, monkeypatch) -> None:
    from polymbappe.simulate.real_bracket import load_ko_winner_overrides

    monkeypatch.chdir(tmp_path)
    assert load_ko_winner_overrides() == {}  # no file yet

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "ko_winner_overrides.yaml").write_text("104:  Spain \n103: France\n")
    assert load_ko_winner_overrides() == {104: "Spain", 103: "France"}


def test_attach_tolerates_duplicate_match_rows() -> None:
    bracket = build_real_bracket(_mini_bracket())
    assert bracket is not None
    dup = _matches(
        [
            {"date": date(2026, 6, 28), "home_team": "Spain", "away_team": "Austria",
             "home_goals": 3, "away_goals": 0, "city": "Arlington"},
            {"date": date(2026, 6, 28), "home_team": "Spain", "away_team": "Austria",
             "home_goals": 3, "away_goals": 0, "city": "Arlington"},
        ]
    )
    attach_played_results(bracket, dup)
    assert bracket.nodes[73].winner == "Spain"


def test_assign_thirds_deterministic_and_feasible_for_all_subsets() -> None:
    from itertools import combinations

    slots = _REAL_THIRD_SLOTS
    first = assign_thirds(list("ABCDEFGH"), slots)
    second = assign_thirds(list("ABCDEFGH"), slots)
    assert first == second  # deterministic without rng
    for combo in combinations("ABCDEFGHIJKL", 8):
        assignment = assign_thirds(list(combo), slots)
        assert assignment is not None, combo
        assert sorted(assignment.values()) == sorted(combo)
        for num, group in assignment.items():
            allowed = dict(slots)[num]
            assert group in allowed


def test_assign_thirds_shuffles_with_rng() -> None:
    slots = _REAL_THIRD_SLOTS
    rng = np.random.default_rng(1)
    seen = {tuple(sorted(assign_thirds(list("ABCDEFGH"), slots, rng).items())) for _ in range(20)}
    assert len(seen) > 1  # candidate shuffling explores several valid matchings


def test_fill_r32_leaves_positions_thirds_and_pin_override() -> None:
    sched = _schedule(
        [
            {"date": date(2026, 6, 28), "stage": "Round of 32", "home_team": "1A",
             "away_team": "2B", "city": "Dallas (Arlington)", "match_number": 73},
            {"date": date(2026, 6, 29), "stage": "Round of 32", "home_team": "1B",
             "away_team": "3A/B", "city": "Atlanta", "match_number": 74},
        ]
    )
    bracket = build_real_bracket(sched)
    assert bracket is not None
    positions = {"1A": "Spain", "2A": "Chile", "3A": "Peru", "4A": "Oman",
                 "1B": "France", "2B": "Ghana", "3B": "Togo", "4B": "Fiji"}
    leaves = fill_r32_leaves(bracket, positions, ["A", "B"])
    assert leaves[73] == ("Spain", "Ghana")
    assert leaves[74][0] == "France" and leaves[74][1] in ("Peru", "Togo")

    # A played result overrides whatever the standings say.
    matches = _matches(
        [
            {"date": date(2026, 6, 28), "home_team": "Spain", "away_team": "Austria",
             "home_goals": 2, "away_goals": 1, "city": "Arlington"},
        ]
    )
    attach_played_results(bracket, matches)
    leaves = fill_r32_leaves(bracket, positions, ["A", "B"])
    assert leaves[73] == ("Spain", "Austria")


def test_beyond_regulation_home_winprob_is_conditional_on_level_after_90() -> None:
    from polymbappe.simulate.match import penalty_home_winprob

    et = score_matrix_from_rates(0.9, 0.5, 0.0, 8)  # home stronger in ET
    p = beyond_regulation_home_winprob(et)
    eh = float(np.tril(et, k=-1).sum())
    ed = float(np.trace(et))
    p_pen = penalty_home_winprob(0.5, 0.5, first_shooter_home=True)
    assert p == pytest.approx(eh + ed * p_pen)
    assert 0.5 < p < 1.0
