from polymbappe.data.schema import GroupStanding
from polymbappe.simulate.third_place import rank_third_placed_teams, select_best_third_placed


def test_fair_play_tiebreak_prefers_fewer_deductions() -> None:
    """FIFA fair-play points are negative deductions: -1 (one yellow) beats -3."""

    def standing(group: str, team: str, fair_play: int) -> GroupStanding:
        return GroupStanding(
            group=group, team=team, points=4, goal_difference=0, goals_scored=2,
            goals_against=2, fair_play_score=fair_play, lots_rank=0,
        )

    ranked = rank_third_placed_teams(
        [standing("A", "Dirty", -3), standing("B", "Clean", -1)]
    )
    assert [row.team for row in ranked] == ["Clean", "Dirty"]


def test_select_best_third_placed_keeps_top_eight() -> None:
    rows = [
        GroupStanding(
            group=chr(ord("A") + idx),
            team=f"T{idx}",
            points=6 - (idx // 4),
            goal_difference=3 - (idx % 4),
            goals_scored=idx,
            goals_against=idx,
            fair_play_score=0,
            lots_rank=0,
        )
        for idx in range(12)
    ]

    selected = select_best_third_placed(rows, n_select=8)
    assert len(selected) == 8
    assert selected[0].points >= selected[-1].points
