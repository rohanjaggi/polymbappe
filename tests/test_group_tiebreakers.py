import numpy as np

from polymbappe.data.schema import Match
from polymbappe.simulate.group import resolve_group_table


def test_h2h_used_after_points_gd_gs() -> None:
    teams = ["A", "B", "C", "D"]
    matches = [
        Match(
            match_id="1",
            date="2026-06-11",
            home_team="A",
            away_team="B",
            home_goals=1,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="2",
            date="2026-06-12",
            home_team="A",
            away_team="C",
            home_goals=0,
            away_goals=1,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="3",
            date="2026-06-13",
            home_team="A",
            away_team="D",
            home_goals=2,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="4",
            date="2026-06-14",
            home_team="B",
            away_team="C",
            home_goals=2,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="5",
            date="2026-06-15",
            home_team="B",
            away_team="D",
            home_goals=1,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="6",
            date="2026-06-16",
            home_team="C",
            away_team="D",
            home_goals=0,
            away_goals=0,
            competition="WC",
            group="A",
        ),
    ]

    table = resolve_group_table("A", teams, matches, np.random.default_rng(123))
    assert [row.team for row in table][:2] == ["A", "B"]


def test_fair_play_used_before_lots() -> None:
    teams = ["A", "B", "C", "D"]
    matches = [
        Match(
            match_id="1",
            date="2026-06-11",
            home_team="A",
            away_team="B",
            home_goals=1,
            away_goals=1,
            competition="WC",
            group="A",
            fair_play_home=0,
            fair_play_away=-1,
        ),
        Match(
            match_id="2",
            date="2026-06-12",
            home_team="A",
            away_team="C",
            home_goals=0,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="3",
            date="2026-06-13",
            home_team="A",
            away_team="D",
            home_goals=0,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="4",
            date="2026-06-14",
            home_team="B",
            away_team="C",
            home_goals=0,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="5",
            date="2026-06-15",
            home_team="B",
            away_team="D",
            home_goals=0,
            away_goals=0,
            competition="WC",
            group="A",
        ),
        Match(
            match_id="6",
            date="2026-06-16",
            home_team="C",
            away_team="D",
            home_goals=1,
            away_goals=0,
            competition="WC",
            group="A",
        ),
    ]

    table = resolve_group_table("A", teams, matches, np.random.default_rng(7))
    a_idx = [row.team for row in table].index("A")
    b_idx = [row.team for row in table].index("B")
    assert a_idx < b_idx
