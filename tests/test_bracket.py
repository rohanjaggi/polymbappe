import numpy as np

from polymbappe.data.schema import GroupStanding
from polymbappe.simulate.bracket import seed_round_of_32


def _round_of_meeting(slot_a: int, slot_b: int) -> int:
    if slot_a // 16 != slot_b // 16:
        return 5  # final
    if slot_a // 8 != slot_b // 8:
        return 4  # semifinal
    if slot_a // 4 != slot_b // 4:
        return 3  # quarterfinal
    if slot_a // 2 != slot_b // 2:
        return 2  # R16
    return 1  # R32


def test_top_winners_pathway_constraints() -> None:
    winners = [
        GroupStanding(
            group=chr(ord("A") + i),
            team=f"W{i+1}",
            points=9,
            goal_difference=5 - i,
            goals_scored=10 - i,
            goals_against=1,
            fair_play_score=0,
            lots_rank=0,
        )
        for i in range(12)
    ]
    others = [f"Q{i}" for i in range(20)]
    ties = seed_round_of_32(winners, others, np.random.default_rng(2026))

    slot_by_team: dict[str, int] = {}
    for tie in ties:
        assert tie.slot is not None
        base = tie.slot * 2
        slot_by_team[tie.home_team] = base
        slot_by_team[tie.away_team] = base + 1

    assert _round_of_meeting(slot_by_team["W1"], slot_by_team["W2"]) == 5
    assert _round_of_meeting(slot_by_team["W1"], slot_by_team["W3"]) >= 4
    assert _round_of_meeting(slot_by_team["W1"], slot_by_team["W4"]) >= 4
    assert _round_of_meeting(slot_by_team["W2"], slot_by_team["W3"]) >= 4
    assert _round_of_meeting(slot_by_team["W2"], slot_by_team["W4"]) >= 4
