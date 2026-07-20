"""Best third-placed teams ranking logic."""

from __future__ import annotations

from polymbappe.data.schema import GroupStanding


def rank_third_placed_teams(third_rows: list[GroupStanding]) -> list[GroupStanding]:
    """Rank third-placed teams across groups using FIFA cross-group rules."""

    # fair_play_score uses FIFA's negative deduction points (yellow = -1, ...),
    # so higher (closer to zero) is better and sorts descending like the rest.
    return sorted(
        third_rows,
        key=lambda row: (
            row.points,
            row.goal_difference,
            row.goals_scored,
            row.fair_play_score,
            -row.lots_rank,
        ),
        reverse=True,
    )


def select_best_third_placed(
    third_rows: list[GroupStanding], n_select: int = 8
) -> list[GroupStanding]:
    """Select the best n third-placed sides."""

    ranked = rank_third_placed_teams(third_rows)
    return ranked[:n_select]
