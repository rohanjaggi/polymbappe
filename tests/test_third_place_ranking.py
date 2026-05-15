from polymbappe.data.schema import GroupStanding
from polymbappe.simulate.third_place import select_best_third_placed


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
