"""Knockout bracket seeding utilities for 2026 format."""

from __future__ import annotations

from typing import Literal

import numpy as np

from polymbappe.data.schema import GroupStanding, KnockoutTie


def seed_round_of_32(
    ranked_group_winners: list[GroupStanding],
    other_qualifiers: list[str],
    rng: np.random.Generator,
) -> list[KnockoutTie]:
    """Seed R32 teams while respecting pathway constraints.

    Top-2 ranked group winners are forced into opposite halves (final-only meeting).
    Top-4 ranked winners are forced into separate quarter-pairs (cannot meet before SF).
    """

    if len(ranked_group_winners) < 12:
        raise ValueError("Expected 12 ranked group winners.")

    all_teams = [row.team for row in ranked_group_winners] + list(other_qualifiers)
    if len(all_teams) != 32:
        raise ValueError("Round of 32 requires exactly 32 teams.")

    slots: list[str | None] = [None] * 32
    protected = {
        0: ranked_group_winners[0].team,
        31: ranked_group_winners[1].team,
        15: ranked_group_winners[2].team,
        16: ranked_group_winners[3].team,
    }
    protected_teams = {team for team in protected.values()}
    for slot, team in protected.items():
        slots[slot] = team

    remaining = [team for team in all_teams if team not in protected_teams]
    shuffled = list(rng.permutation(remaining))
    ptr = 0
    for idx, slot_team in enumerate(slots):
        if slot_team is None:
            slots[idx] = str(shuffled[ptr])
            ptr += 1

    ties: list[KnockoutTie] = []
    for idx in range(0, 32, 2):
        pathway: Literal["A", "B"] = "A" if idx < 16 else "B"
        home_team = slots[idx]
        away_team = slots[idx + 1]
        if home_team is None or away_team is None:
            raise RuntimeError("Incomplete bracket slot assignment.")
        ties.append(
            KnockoutTie(
                round_name="R32",
                home_team=home_team,
                away_team=away_team,
                pathway=pathway,
                slot=idx // 2,
            )
        )
    return ties
