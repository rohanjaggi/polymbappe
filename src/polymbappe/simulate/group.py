"""Group stage simulation and ranking."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from polymbappe.data.schema import GroupStanding, Match


def _match_points(home_goals: int, away_goals: int) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def _head_to_head_key(
    team: str, tied_teams: set[str], matches: list[Match]
) -> tuple[int, int, int]:
    points = 0
    gd = 0
    gs = 0
    for match in matches:
        if {match.home_team, match.away_team}.issubset(tied_teams):
            if team == match.home_team:
                home_pts, _ = _match_points(match.home_goals, match.away_goals)
                points += home_pts
                gd += match.home_goals - match.away_goals
                gs += match.home_goals
            elif team == match.away_team:
                _, away_pts = _match_points(match.home_goals, match.away_goals)
                points += away_pts
                gd += match.away_goals - match.home_goals
                gs += match.away_goals
    return points, gd, gs


def resolve_group_table(
    group: str, teams: list[str], matches: list[Match], rng: np.random.Generator
) -> list[GroupStanding]:
    """Resolve FIFA 2026 group ranking order.

    Tiebreaker order: points, goal difference, goals scored, head-to-head,
    fair play, drawing of lots.
    """

    if len(teams) != 4:
        raise ValueError("A 2026 group must contain exactly four teams.")

    stats: dict[str, dict[str, int]] = {
        team: {"points": 0, "gf": 0, "ga": 0, "fair_play": 0} for team in teams
    }
    for match in matches:
        home_points, away_points = _match_points(match.home_goals, match.away_goals)
        stats[match.home_team]["points"] += home_points
        stats[match.away_team]["points"] += away_points
        stats[match.home_team]["gf"] += match.home_goals
        stats[match.home_team]["ga"] += match.away_goals
        stats[match.away_team]["gf"] += match.away_goals
        stats[match.away_team]["ga"] += match.home_goals
        stats[match.home_team]["fair_play"] += match.fair_play_home
        stats[match.away_team]["fair_play"] += match.fair_play_away

    standings = [
        GroupStanding(
            group=group,
            team=team,
            points=stats[team]["points"],
            goal_difference=stats[team]["gf"] - stats[team]["ga"],
            goals_scored=stats[team]["gf"],
            goals_against=stats[team]["ga"],
            fair_play_score=stats[team]["fair_play"],
        )
        for team in teams
    ]

    standings.sort(key=lambda s: (s.points, s.goal_difference, s.goals_scored), reverse=True)

    grouped: dict[tuple[int, int, int], list[GroupStanding]] = defaultdict(list)
    for row in standings:
        grouped[(row.points, row.goal_difference, row.goals_scored)].append(row)

    resolved: list[GroupStanding] = []
    for key in sorted(grouped.keys(), reverse=True):
        tied_rows = grouped[key]
        if len(tied_rows) == 1:
            resolved.extend(tied_rows)
            continue

        tied_teams = {row.team for row in tied_rows}
        h2h_scores = {
            row.team: _head_to_head_key(row.team, tied_teams, matches) for row in tied_rows
        }

        # fair_play_score uses FIFA's negative deduction points (yellow = -1, ...),
        # so higher (closer to zero) is better and sorts descending like the rest.
        tied_rows.sort(
            key=lambda row: (
                h2h_scores[row.team][0],
                h2h_scores[row.team][1],
                h2h_scores[row.team][2],
                row.fair_play_score,
            ),
            reverse=True,
        )

        i = 0
        while i < len(tied_rows):
            j = i + 1
            while j < len(tied_rows):
                left = tied_rows[i]
                right = tied_rows[j]
                if (
                    h2h_scores[left.team] != h2h_scores[right.team]
                    or left.fair_play_score != right.fair_play_score
                ):
                    break
                j += 1

            if j - i > 1:
                lot_order = rng.permutation(j - i)
                for offset, lot_pos in enumerate(lot_order):
                    tied_rows[i + lot_pos].lots_rank = offset
                tied_rows[i:j] = sorted(tied_rows[i:j], key=lambda row: row.lots_rank)

            resolved.extend(tied_rows[i:j])
            i = j

    return resolved
