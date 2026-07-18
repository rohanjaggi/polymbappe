"""The real 2026 knockout tree, shared by the Monte Carlo engine and the bracket forecaster.

The ingested schedule carries the actual knockout fixtures: R32 sides as group-position
placeholders (``"1A"``, ``"2B"``) or third-place constraint sets (``"3A/B/C/D/F"`` = the
third-placed team from one of those groups), and later rounds as ``W##``/``L##`` slot
references keyed by the official FIFA match number. This module parses that tree once,
pins already-played matches onto their nodes (resolving extra-time/penalty ties whose
feed scoreline is a draw by looking at who appears in later rounds), and fills the R32
leaves from group standings — real or simulated.

Both :mod:`polymbappe.simulate.tournament` (per-iteration leaf filling + winner locking)
and :mod:`polymbappe.simulate.knockout_bracket` (distribution propagation) consume the
same tree, so the two artifacts cannot disagree about the bracket or who advanced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

#: Schedule ``stage`` label -> round key (third-place play-off is handled separately).
KO_STAGE_TO_ROUND: dict[str, str] = {
    "Round of 32": "R32",
    "Round of 16": "R16",
    "Quarter-final": "QF",
    "Semi-final": "SF",
    "Final": "FINAL",
}

THIRD_PLACE_STAGE = "Match for third place"

#: Bracket rounds, broadest to narrowest, and the FIFA match number each round starts at
#: (used only when the schedule lacks an explicit ``match_number`` column). The stage a
#: round's winners reach follows in ``NEXT_STAGE``.
ROUND_ORDER: tuple[str, ...] = ("R32", "R16", "QF", "SF", "FINAL")
ROUND_BASE: dict[str, int] = {"R32": 73, "R16": 89, "QF": 97, "SF": 101, "FINAL": 103}
NEXT_STAGE: dict[str, str] = {
    "R32": "R16",
    "R16": "QF",
    "QF": "SF",
    "SF": "FINAL",
    "FINAL": "champion",
}

#: Fixtures a round must contain for the tree to be well-formed (2026 format).
ROUND_SIZES: dict[str, int] = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "FINAL": 1}

_REF_RE = re.compile(r"^([WL])(\d+)$")
_POSITION_RE = re.compile(r"^([12])([A-Z])$")
_THIRD_RE = re.compile(r"^3([A-Z](?:/[A-Z])*)$")

SideKind = Literal["team", "position", "third", "winner", "loser"]


@dataclass(slots=True)
class Side:
    """Parsed occupant spec of one fixture side."""

    kind: SideKind
    label: str
    team: str | None = None  # kind == "team"
    group: str | None = None  # kind == "position"
    rank: int | None = None  # kind == "position": 1 or 2
    allowed_groups: frozenset[str] = frozenset()  # kind == "third"
    ref: int | None = None  # kind == "winner"/"loser": referenced match number


def parse_side(label: str) -> Side:
    """Parse one schedule side label into its occupant spec."""

    label = label.strip()
    m = _REF_RE.match(label)
    if m is not None:
        kind = "winner" if m.group(1) == "W" else "loser"
        return Side(kind=kind, label=label, ref=int(m.group(2)))
    m = _POSITION_RE.match(label)
    if m is not None:
        return Side(kind="position", label=label, rank=int(m.group(1)), group=m.group(2))
    m = _THIRD_RE.match(label)
    if m is not None:
        return Side(kind="third", label=label, allowed_groups=frozenset(m.group(1).split("/")))
    return Side(kind="team", label=label, team=label)


@dataclass(slots=True)
class BracketNode:
    """One knockout fixture in the real bracket tree."""

    number: int
    round: str
    date: _date | None
    city: str | None
    home: Side
    away: Side
    # Filled by :func:`attach_played_results` (matches-table orientation):
    pinned_home: str | None = None
    pinned_away: str | None = None
    winner: str | None = None  # decisive score or draw-inferred
    loser: str | None = None
    drawn_unresolved: bool = False  # played draw whose winner cannot be inferred yet

    @property
    def pinned(self) -> bool:
        return self.pinned_home is not None


@dataclass(slots=True)
class RealBracket:
    """The parsed knockout tree plus lookup indexes."""

    nodes: dict[int, BracketNode]
    rounds: dict[str, list[int]]  # round -> node numbers in bracket order
    third_place: BracketNode | None = None  # not part of the tree (SF losers)
    _consumers: dict[int, tuple[int, str]] = field(default_factory=dict)

    def consumer_of(self, num: int) -> tuple[int, str] | None:
        """Which ``(node number, side)`` consumes the winner of match ``num``."""

        return self._consumers.get(num)


def build_real_bracket(schedule: pl.DataFrame) -> RealBracket | None:
    """Parse the schedule's knockout fixtures into a :class:`RealBracket`.

    Numbering: the ``match_number`` column is used when present and non-null on every
    knockout row (the official FIFA numbers the ``W##``/``L##`` placeholders reference);
    otherwise fixtures are numbered per round chronologically from :data:`ROUND_BASE` —
    ambiguous when a round plays several fixtures on one date, so a warning is logged.

    Returns ``None`` when the schedule is empty or carries no knockout fixtures.
    """

    if schedule.is_empty() or "stage" not in schedule.columns:
        return None
    ko = schedule.filter(pl.col("stage").is_in(list(KO_STAGE_TO_ROUND)))
    if ko.is_empty():
        return None

    have_numbers = (
        "match_number" in ko.columns and ko["match_number"].null_count() == 0
    )
    nodes: dict[int, BracketNode] = {}
    rounds: dict[str, list[int]] = {}
    for round_name in ROUND_ORDER:
        labels = [k for k, v in KO_STAGE_TO_ROUND.items() if v == round_name]
        rows = ko.filter(pl.col("stage").is_in(labels)).sort(["date", "match_id"])
        if rows.is_empty():
            continue
        if not have_numbers and rows.group_by("date").len()["len"].max() > 1:
            logger.warning(
                "real_bracket.ambiguous_numbering",
                round=round_name,
                reason="several fixtures share a date and the schedule has no match_number",
            )
        for idx, r in enumerate(rows.iter_rows(named=True)):
            num = int(r["match_number"]) if have_numbers else ROUND_BASE[round_name] + idx
            nodes[num] = BracketNode(
                number=num,
                round=round_name,
                date=r.get("date"),
                city=r.get("city"),
                home=parse_side(str(r["home_team"])),
                away=parse_side(str(r["away_team"])),
            )
            rounds.setdefault(round_name, []).append(num)

    third_place: BracketNode | None = None
    third_rows = schedule.filter(pl.col("stage") == THIRD_PLACE_STAGE)
    if not third_rows.is_empty():
        r = third_rows.row(0, named=True)
        third_place = BracketNode(
            number=int(r["match_number"]) if r.get("match_number") is not None else -1,
            round="THIRD",
            date=r.get("date"),
            city=r.get("city"),
            home=parse_side(str(r["home_team"])),
            away=parse_side(str(r["away_team"])),
        )

    consumers = {
        side.ref: (node.number, side_name)
        for node in nodes.values()
        for side_name, side in (("home", node.home), ("away", node.away))
        if side.kind == "winner" and side.ref is not None
    }
    return RealBracket(
        nodes=nodes, rounds=rounds, third_place=third_place, _consumers=consumers
    )


def bracket_compatible(bracket: RealBracket, structure: object) -> bool:
    """Whether the parsed tree is complete and consistent with the draw structure.

    Callers fall back to the random pathway-constrained seeding when this is false, so a
    malformed schedule degrades to the pre-anchoring behavior instead of crashing.
    """

    groups = set(getattr(structure, "groups", {}))
    for round_name, size in ROUND_SIZES.items():
        if len(bracket.rounds.get(round_name, [])) != size:
            logger.warning(
                "real_bracket.incompatible",
                round=round_name,
                expected=size,
                got=len(bracket.rounds.get(round_name, [])),
            )
            return False
    for node in bracket.nodes.values():
        for side in (node.home, node.away):
            leaf_round = node.round == "R32"
            leaf_kind = side.kind in ("team", "position", "third")
            if side.kind == "loser" or (leaf_round and not leaf_kind) or (
                not leaf_round and side.kind not in ("winner", "team")
            ):
                # The main tree never consumes losers (third place is separate); R32
                # sides must be leaves and later rounds must be winner refs (or concrete
                # teams) — anything else means a malformed schedule.
                logger.warning(
                    "real_bracket.incompatible", node=node.number, bad_side=side.label
                )
                return False
            if side.kind == "winner":
                if side.ref not in bracket.nodes:
                    logger.warning(
                        "real_bracket.incompatible", node=node.number, dangling_ref=side.label
                    )
                    return False
            elif side.kind == "position":
                if side.group not in groups:
                    logger.warning(
                        "real_bracket.incompatible", node=node.number, unknown_group=side.label
                    )
                    return False
            elif side.kind == "third":
                if not side.allowed_groups <= groups:
                    logger.warning(
                        "real_bracket.incompatible", node=node.number, unknown_group=side.label
                    )
                    return False
    return True


# ---------------------------------------------------------------------------
# Played-result pinning and shootout-winner inference
# ---------------------------------------------------------------------------


def _city_tokens(city: str | None) -> frozenset[str]:
    """Match tokens for a host-city label: ``"Dallas (Arlington)"`` -> dallas/arlington/full."""

    if not city:
        return frozenset()
    lowered = city.strip().lower()
    tokens = {lowered}
    m = re.match(r"^(.*?)\s*\((.*?)\)$", lowered)
    if m is not None:
        tokens.add(m.group(1).strip())
        tokens.add(m.group(2).strip())
    return frozenset(tokens)


def _played_ko_matches(bracket: RealBracket, matches: pl.DataFrame) -> list[dict[str, object]]:
    """Completed WC2026 knockout rows, one per match_id (newest wins), oldest first."""

    if matches.is_empty() or "is_knockout" not in matches.columns:
        return []
    ko_dates = [n.date for n in bracket.nodes.values() if n.date is not None]
    ko = matches.filter(
        (pl.col("competition") == "FIFA World Cup")
        & pl.col("is_knockout")
        & pl.col("home_goals").is_not_null()
        & pl.col("away_goals").is_not_null()
    )
    if ko_dates:
        ko = ko.filter(pl.col("date") >= min(ko_dates))
    if "match_id" in ko.columns:
        ko = ko.unique(subset=["match_id"], keep="last", maintain_order=True)
    return list(ko.sort("date").iter_rows(named=True))


def attach_played_results(bracket: RealBracket, matches: pl.DataFrame) -> None:
    """Pin played matches onto bracket nodes and lock/infer their winners (mutates nodes).

    Pinning, in priority order per played match: exact team-pair (nodes whose sides are
    concrete teams), unique ``(date, city)`` token match, then unique-remaining-per-date.
    A played match that cannot be pinned is logged and skipped — the tie degrades to
    "simulate it" rather than corrupting the tree.

    Winners: a decisive scoreline locks winner/loser directly. A drawn scoreline (the feed
    records the 90'/120' score, so extra-time/penalty ties land as draws) is resolved by
    who appears further down the tree — the parent node's pinned pair, any later round's
    pinned occupants, or (for a drawn semi-final) the third-place match's participants,
    who are by definition the SF losers. A draw with no later evidence yet is flagged
    ``drawn_unresolved`` for the caller to forecast conditionally (beyond regulation).
    """

    played = _played_ko_matches(bracket, matches)
    if not played:
        return

    pinnable = list(bracket.nodes.values()) + (
        [bracket.third_place] if bracket.third_place is not None else []
    )
    unpinned = {id(n): n for n in pinnable}
    remaining: list[dict[str, object]] = []

    # Pass 1: exact team-pair (nodes already carrying concrete team names).
    by_pair = {
        frozenset((n.home.team, n.away.team)): n
        for n in pinnable
        if n.home.kind == "team" and n.away.kind == "team"
    }
    for row in played:
        pair = frozenset((str(row["home_team"]), str(row["away_team"])))
        node = by_pair.get(pair)
        if node is not None and id(node) in unpinned:
            _pin(node, row)
            del unpinned[id(node)]
        else:
            remaining.append(row)

    # Pass 2: unique (date, city-token) match.
    still: list[dict[str, object]] = []
    for row in remaining:
        city = str(row.get("city") or "").strip().lower()
        candidates = [
            n
            for n in unpinned.values()
            if n.date == row["date"] and city and city in _city_tokens(n.city)
        ]
        if len(candidates) == 1:
            _pin(candidates[0], row)
            del unpinned[id(candidates[0])]
        else:
            still.append(row)

    # Pass 3: a date with exactly one unpinned node and one unpinned match.
    for row in still:
        candidates = [n for n in unpinned.values() if n.date == row["date"]]
        if len(candidates) == 1:
            _pin(candidates[0], row)
            del unpinned[id(candidates[0])]
        else:
            logger.warning(
                "real_bracket.unpinnable_match",
                match_id=row.get("match_id"),
                date=str(row.get("date")),
                city=row.get("city"),
                candidates=len(candidates),
            )

    _resolve_drawn_ties(bracket)


def _pin(node: BracketNode, row: dict[str, object]) -> None:
    home, away = str(row["home_team"]), str(row["away_team"])
    hg, ag = int(row["home_goals"]), int(row["away_goals"])  # type: ignore[arg-type]
    node.pinned_home, node.pinned_away = home, away
    if hg > ag:
        node.winner, node.loser = home, away
    elif ag > hg:
        node.winner, node.loser = away, home


def _resolve_drawn_ties(bracket: RealBracket) -> None:
    """Infer winners of pinned-but-drawn ties from later-round participants."""

    later_pinned: dict[str, set[str]] = {}
    for round_name, nums in bracket.rounds.items():
        teams: set[str] = set()
        for num in nums:
            node = bracket.nodes[num]
            if node.pinned:
                teams.update({node.pinned_home, node.pinned_away})  # type: ignore[arg-type]
        later_pinned[round_name] = teams

    third_participants: set[str] = set()
    if bracket.third_place is not None and bracket.third_place.pinned:
        third_participants = {
            bracket.third_place.pinned_home,  # type: ignore[arg-type]
            bracket.third_place.pinned_away,  # type: ignore[arg-type]
        }

    for node in bracket.nodes.values():
        if not node.pinned or node.winner is not None:
            continue
        pair = {node.pinned_home, node.pinned_away}
        node_round_idx = ROUND_ORDER.index(node.round)
        seen_later = {
            t
            for r in ROUND_ORDER[node_round_idx + 1 :]
            for t in later_pinned.get(r, set())
        }
        advanced = pair & seen_later
        if len(advanced) == 1:
            node.winner = advanced.pop()
            node.loser = (pair - {node.winner}).pop()
            logger.info(
                "real_bracket.draw_inferred",
                match_number=node.number,
                winner=node.winner,
                via="later_round",
            )
            continue
        if node.round == "SF" and third_participants:
            lost = pair & third_participants
            if len(lost) == 1:
                node.loser = lost.pop()
                node.winner = (pair - {node.loser}).pop()
                logger.info(
                    "real_bracket.draw_inferred",
                    match_number=node.number,
                    winner=node.winner,
                    via="third_place_match",
                )
                continue
        node.drawn_unresolved = True
        logger.info(
            "real_bracket.draw_unresolved",
            match_number=node.number,
            teams=sorted(pair),
        )


# ---------------------------------------------------------------------------
# Group positions and third-place slot assignment
# ---------------------------------------------------------------------------


def real_group_positions(
    matches: pl.DataFrame, structure: object
) -> tuple[dict[str, str], list[str]] | None:
    """Real group standings, once every group has played all six fixtures.

    Returns ``({"1A": team, ..., "4L": team}, qualified third-place group letters in FIFA
    ranking order)`` computed from the ingested results via the same tiebreaker machinery
    the simulation uses, or ``None`` while any group is incomplete. Drawing-of-lots ties
    use a fixed RNG so the outcome is deterministic.
    """

    from polymbappe.data.schema import Match
    from polymbappe.simulate.group import resolve_group_table
    from polymbappe.simulate.third_place import rank_third_placed_teams
    from polymbappe.simulate.tournament import build_played_group_results

    played = build_played_group_results(matches, structure)  # type: ignore[arg-type]
    groups: dict[str, list[str]] = getattr(structure, "groups", {})
    if not groups or any(len(played.get(g, {})) != 6 for g in groups):
        return None

    rng = np.random.default_rng(0)
    positions: dict[str, str] = {}
    thirds = []
    for group, members in groups.items():
        fixtures = []
        for k, (pair, goals) in enumerate(played[group].items()):
            t1, t2 = sorted(pair)
            fixtures.append(
                Match(
                    match_id=f"{group}-{k}",
                    date=_date(2026, 6, 11),
                    home_team=t1,
                    away_team=t2,
                    home_goals=goals[t1],
                    away_goals=goals[t2],
                    competition="FIFA World Cup",
                    group=group,
                    neutral_site=True,
                )
            )
        standings = resolve_group_table(group, members, fixtures, rng)
        for rank, row in enumerate(standings, start=1):
            positions[f"{rank}{group}"] = row.team
        thirds.append(standings[2])

    n_thirds = int(getattr(structure, "n_qualify_thirds", 8))
    qualified = [s.group for s in rank_third_placed_teams(thirds)[:n_thirds]]
    return positions, qualified


def assign_thirds(
    qualified_groups: list[str],
    slots: list[tuple[int, frozenset[str]]],
    rng: np.random.Generator | None = None,
) -> dict[int, str] | None:
    """Assign qualified third-place groups to R32 constraint slots (perfect matching).

    ``slots`` are ``(node_number, allowed_groups)`` pairs. Backtracking over slots ordered
    by ascending feasible-candidate count; candidate order is shuffled when ``rng`` is
    given (so the Monte Carlo doesn't systematically favor one valid allocation) and
    sorted for determinism otherwise. FIFA's constraint sets admit a perfect matching for
    every possible qualified-group combination, so ``None`` indicates malformed input.
    """

    ordered = sorted(slots, key=lambda s: len(s[1] & set(qualified_groups)))
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(ordered):
            return True
        num, allowed = ordered[i]
        candidates = [g for g in qualified_groups if g in allowed and g not in used]
        if rng is not None:
            rng.shuffle(candidates)
        else:
            candidates.sort()
        for g in candidates:
            assignment[num] = g
            used.add(g)
            if backtrack(i + 1):
                return True
            used.discard(g)
            del assignment[num]
        return False

    if not backtrack(0):
        logger.warning(
            "real_bracket.third_assignment_failed",
            qualified=qualified_groups,
            slots=[(n, sorted(a)) for n, a in slots],
        )
        return None
    return assignment


def oriented_pin(node: BracketNode, team_group: dict[str, str]) -> tuple[str, str]:
    """A pinned node's occupants oriented to its schedule sides (home, away)."""

    pair = (str(node.pinned_home), str(node.pinned_away))

    def fits(side: Side, team: str) -> bool:
        group = team_group.get(team)
        if side.kind == "position":
            return group == side.group
        if side.kind == "third":
            return group in side.allowed_groups
        if side.kind == "team":
            return team == side.team
        return True

    if fits(node.home, pair[1]) and fits(node.away, pair[0]) and not (
        fits(node.home, pair[0]) and fits(node.away, pair[1])
    ):
        return pair[1], pair[0]
    return pair


def fill_r32_leaves(
    bracket: RealBracket,
    positions: dict[str, str],
    qualified_third_groups: list[str],
    rng: np.random.Generator | None = None,
) -> dict[int, tuple[str, str]]:
    """Concrete ``(home, away)`` occupants for every R32 node.

    Pinned nodes (already played in reality) keep their pinned pair, and ``positions`` is
    reconciled against them first: when the standings disagree with a pin (possible only
    while the group stage is partially simulated), the pinned team takes its real slot and
    the displaced team moves to the pinned team's simulated rank — a within-group swap, so
    every team still occupies exactly one slot and nobody is double-counted. Unpinned
    sides then resolve group positions via ``positions`` and third-place constraint slots
    via one :func:`assign_thirds` matching over the not-yet-consumed qualified groups.
    """

    positions = dict(positions)
    slot_of = {team: slot for slot, team in positions.items()}
    team_group = {team: slot[1:] for slot, team in positions.items()}
    r32_nums = bracket.rounds.get("R32", [])
    pinned_nodes = [bracket.nodes[n] for n in r32_nums if bracket.nodes[n].pinned]

    used_third_groups: set[str] = set()
    for node in pinned_nodes:
        home_occ, away_occ = oriented_pin(node, team_group)
        for side, team in ((node.home, home_occ), (node.away, away_occ)):
            group = team_group.get(team)
            if side.kind == "position":
                slot = f"{side.rank}{side.group}"
            elif side.kind == "third" and group in side.allowed_groups:
                slot = f"3{group}"
                used_third_groups.add(group)
            else:
                continue
            if positions.get(slot) == team:
                continue
            displaced, prev_slot = positions.get(slot), slot_of.get(team)
            positions[slot] = team
            slot_of[team] = slot
            # Same-group swap keeps the map a permutation; anything else (an alias gap)
            # is left inconsistent and logged rather than silently reshuffled.
            if displaced is not None and prev_slot is not None and prev_slot[1:] == slot[1:]:
                positions[prev_slot] = displaced
                slot_of[displaced] = prev_slot
            else:
                logger.warning(
                    "real_bracket.pin_outside_group", team=team, slot=slot, node=node.number
                )

    open_third_slots: list[tuple[int, str, frozenset[str]]] = []  # (num, side_name, allowed)
    for num in r32_nums:
        node = bracket.nodes[num]
        if node.pinned:
            continue
        for side_name, side in (("home", node.home), ("away", node.away)):
            if side.kind == "third":
                open_third_slots.append((num, side_name, side.allowed_groups))

    open_assignment: dict[tuple[int, str], str] = {}
    if open_third_slots:
        free = [g for g in qualified_third_groups if g not in used_third_groups]
        matched = assign_thirds(
            free, [(num, allowed) for num, _s, allowed in open_third_slots], rng
        )
        if matched is not None:
            for num, side_name, _allowed in open_third_slots:
                open_assignment[(num, side_name)] = positions[f"3{matched[num]}"]

    leaves: dict[int, tuple[str, str]] = {}
    for num in r32_nums:
        node = bracket.nodes[num]
        if node.pinned:
            leaves[num] = oriented_pin(node, team_group)
            continue
        occupants: list[str] = []
        for side_name, side in (("home", node.home), ("away", node.away)):
            if side.kind == "team":
                occupants.append(side.team or side.label)
            elif side.kind == "position":
                occupants.append(positions[f"{side.rank}{side.group}"])
            else:  # third
                occupants.append(open_assignment.get((num, side_name), side.label))
        leaves[num] = (occupants[0], occupants[1])
    return leaves
