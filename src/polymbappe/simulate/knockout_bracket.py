"""Knockout-bracket forecast anchored to the *real* World Cup draw (spec 4.3).

The bracket is not simulated with a random seeding — it is read from the ingested tournament
schedule, which carries the actual R32/R16 fixtures (real team names) and the QF → Final
fixtures as bracket-slot placeholders (``W89`` = winner of match 89, ``L101`` = loser of
match 101). Combined with the played-results in the matches table this gives the current
bracket exactly as it stands.

From that fixed tree we project forward: each tie's advance probability comes from the strength
model, and the probability a team *reaches* a future slot is propagated down the bracket
(``P(reach) = P(win previous tie)`` chained through the subtree), conditioned on results already
played. So every future match resolves into the set of matchups that can actually occur — the
"expected outcomes of all possible games in future rounds" — each with its probability of
happening and the model's advance + FT/ET/penalties breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from polymbappe.simulate.match import (
    ET_GOAL_SCALE,
    hda_marginals,
    knockout_outcome_breakdown,
    score_matrix_from_rates,
)

#: Column order / schema of the ``knockout_bracket.parquet`` artifact.
BRACKET_SCHEMA: dict[str, pl.DataType] = {
    "round": pl.Utf8,
    "match_number": pl.Int32,
    "rank": pl.Int32,
    "team_a": pl.Utf8,
    "team_b": pl.Utf8,
    "matchup_prob": pl.Float64,
    "p_a_advance": pl.Float64,
    "p_b_advance": pl.Float64,
    "p_decided_reg": pl.Float64,
    "p_decided_et": pl.Float64,
    "p_decided_pens": pl.Float64,
    "model_a": pl.Float64,
    "model_draw": pl.Float64,
    "model_b": pl.Float64,
    "exp_a_goals": pl.Float64,
    "exp_b_goals": pl.Float64,
}

#: Schedule ``stage`` label -> round key (third-place play-off is excluded from the bracket).
_STAGE_TO_ROUND: dict[str, str] = {
    "Round of 32": "R32",
    "Round of 16": "R16",
    "Quarter-final": "QF",
    "Semi-final": "SF",
    "Final": "FINAL",
}

#: Bracket rounds, broadest to narrowest, and the FIFA slot number each round starts at. The
#: placeholders (``W89``..) reference these numbers, so R16=89, QF=97, SF=101 must line up.
_ROUND_ORDER: tuple[str, ...] = ("R32", "R16", "QF", "SF", "FINAL")
_ROUND_BASE: dict[str, int] = {"R32": 73, "R16": 89, "QF": 97, "SF": 101, "FINAL": 103}

_PLACEHOLDER = re.compile(r"^([WL])(\d+)$")
#: Reach-probability entries below this are pruned (then renormalised) to bound the fan-out of
#: possible matchups in the deepest rounds.
_PRUNE_EPS = 1e-4


@dataclass(slots=True)
class _Node:
    """One knockout fixture in the real bracket tree."""

    match_number: int
    round: str
    side_a: str  # concrete team name or a ``W##`` / ``L##`` slot placeholder
    side_b: str
    played_winner: str | None = None  # set when the matches table has a real result
    played_loser: str | None = None


def _build_nodes(schedule: pl.DataFrame, played: dict[frozenset[str], dict[str, object]]) -> dict[int, _Node]:
    """Parse the schedule's knockout fixtures into a ``match_number -> _Node`` tree.

    Fixtures are numbered per round by chronological order (the FIFA/openfootball convention the
    ``W##`` placeholders follow). Concrete-team fixtures already played are tagged with their
    real winner/loser from ``played``.
    """

    nodes: dict[int, _Node] = {}
    for round_name in _ROUND_ORDER:
        labels = [k for k, v in _STAGE_TO_ROUND.items() if v == round_name]
        rows = schedule.filter(pl.col("stage").is_in(labels)).sort(["date", "match_id"])
        for idx, r in enumerate(rows.iter_rows(named=True)):
            num = _ROUND_BASE[round_name] + idx
            a, b = str(r["home_team"]), str(r["away_team"])
            node = _Node(match_number=num, round=round_name, side_a=a, side_b=b)
            if _PLACEHOLDER.match(a) is None and _PLACEHOLDER.match(b) is None:
                res = played.get(frozenset((a, b)))
                if res is not None and res["advanced"] is not None:
                    node.played_winner = str(res["advanced"])
                    node.played_loser = b if res["advanced"] == a else a
            nodes[num] = node
    return nodes


class _BracketForecaster:
    """Propagates reach/advance probabilities through the fixed bracket tree."""

    def __init__(self, nodes: dict[int, _Node], model: object, structure: object) -> None:
        self._nodes = nodes
        self._model = model
        self._pen: dict[str, float] = getattr(structure, "penalty_rate", {}) or {}
        self._max_goals: int = model.max_goals  # type: ignore[attr-defined]
        self._rho: float = model.rho  # type: ignore[attr-defined]
        self._grid = np.arange(self._max_goals + 1)
        self._advance_cache: dict[tuple[str, str], float] = {}
        self._winner_cache: dict[int, dict[str, float]] = {}
        self._loser_cache: dict[int, dict[str, float]] = {}

    # -- model primitives -----------------------------------------------------

    def _matrices(self, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
        lam, mu = self._model.rates(a, b, neutral=True)  # type: ignore[attr-defined]
        reg = score_matrix_from_rates(lam, mu, self._rho, self._max_goals)
        et = score_matrix_from_rates(
            lam * ET_GOAL_SCALE, mu * ET_GOAL_SCALE, self._rho, self._max_goals
        )
        return reg, et

    def advance_prob(self, a: str, b: str) -> float:
        """P(a advances past b) at a neutral venue, cached per ordered pair."""

        key = (a, b)
        if key not in self._advance_cache:
            reg, et = self._matrices(a, b)
            self._advance_cache[key] = knockout_outcome_breakdown(
                reg, et, self._pen.get(a, 0.5), self._pen.get(b, 0.5)
            ).p_home_advance
        return self._advance_cache[key]

    # -- reach-distribution propagation ---------------------------------------

    def _side_dist(self, side: str) -> dict[str, float]:
        """Distribution over which team fills a fixture side (concrete team or slot ref)."""

        m = _PLACEHOLDER.match(side)
        if m is None:
            return {side: 1.0}
        kind, num = m.group(1), int(m.group(2))
        child = self._nodes.get(num)
        if child is None:  # unknown reference -> treat the label itself as a team
            return {side: 1.0}
        return self._winner_dist(num) if kind == "W" else self._loser_dist(num)

    def _resolve(self, num: int) -> None:
        """Populate winner/loser caches for one node from its children (memoised)."""

        node = self._nodes[num]
        if node.played_winner is not None and node.played_loser is not None:
            self._winner_cache[num] = {node.played_winner: 1.0}
            self._loser_cache[num] = {node.played_loser: 1.0}
            return
        a_dist, b_dist = self._side_dist(node.side_a), self._side_dist(node.side_b)
        wd: dict[str, float] = {}
        ld: dict[str, float] = {}
        for a, pa in a_dist.items():
            for b, pb in b_dist.items():
                if a == b:
                    continue
                p = pa * pb
                pa_win = self.advance_prob(a, b)
                wd[a] = wd.get(a, 0.0) + p * pa_win
                wd[b] = wd.get(b, 0.0) + p * (1.0 - pa_win)
                ld[a] = ld.get(a, 0.0) + p * (1.0 - pa_win)
                ld[b] = ld.get(b, 0.0) + p * pa_win
        self._winner_cache[num] = _prune(wd)
        self._loser_cache[num] = _prune(ld)

    def _winner_dist(self, num: int) -> dict[str, float]:
        if num not in self._winner_cache:
            self._resolve(num)
        return self._winner_cache[num]

    def _loser_dist(self, num: int) -> dict[str, float]:
        if num not in self._loser_cache:
            self._resolve(num)
        return self._loser_cache[num]

    # -- row emission ---------------------------------------------------------

    def rows_for(self, num: int, top_n: int) -> list[dict[str, object]]:
        """Emit one row per possible matchup at a fixture, most-probable first."""

        node = self._nodes[num]
        a_dist, b_dist = self._side_dist(node.side_a), self._side_dist(node.side_b)
        combos = [
            (a, b, pa * pb)
            for a, pa in a_dist.items()
            for b, pb in b_dist.items()
            if a != b
        ]
        combos.sort(key=lambda x: -x[2])
        rows: list[dict[str, object]] = []
        for rank, (a, b, mp) in enumerate(combos[:top_n], start=1):
            reg, et = self._matrices(a, b)
            breakdown = knockout_outcome_breakdown(
                reg, et, self._pen.get(a, 0.5), self._pen.get(b, 0.5)
            )
            h, d, aw = hda_marginals(reg)
            rows.append(
                {
                    "round": node.round,
                    "match_number": num,
                    "rank": rank,
                    "team_a": a,
                    "team_b": b,
                    "matchup_prob": mp,
                    "p_a_advance": breakdown.p_home_advance,
                    "p_b_advance": breakdown.p_away_advance,
                    "p_decided_reg": breakdown.p_decided_reg,
                    "p_decided_et": breakdown.p_decided_et,
                    "p_decided_pens": breakdown.p_decided_pens,
                    "model_a": h,
                    "model_draw": d,
                    "model_b": aw,
                    "exp_a_goals": float((reg.sum(axis=1) * self._grid).sum()),
                    "exp_b_goals": float((reg.sum(axis=0) * self._grid).sum()),
                }
            )
        return rows


def _prune(dist: dict[str, float]) -> dict[str, float]:
    """Drop negligible entries and renormalise a reach distribution to sum to 1."""

    kept = {t: p for t, p in dist.items() if p >= _PRUNE_EPS}
    total = sum(kept.values())
    if total <= 0:
        return dist
    return {t: p / total for t, p in kept.items()}


def _played_lookup(matches: pl.DataFrame) -> dict[frozenset[str], dict[str, object]]:
    """Order-independent ``{frozenset(teams): result}`` for played WC2026 knockout ties."""

    from polymbappe.simulate.tournament import WC2026_START

    lookup: dict[frozenset[str], dict[str, object]] = {}
    if matches.is_empty() or "is_knockout" not in matches.columns:
        return lookup
    ko = matches.filter(
        (pl.col("competition") == "FIFA World Cup")
        & pl.col("is_knockout")
        & (pl.col("date") >= WC2026_START)
    )
    for r in ko.iter_rows(named=True):
        hg, ag = r.get("home_goals"), r.get("away_goals")
        if hg is None or ag is None:
            continue
        home, away, hg, ag = str(r["home_team"]), str(r["away_team"]), int(hg), int(ag)
        advanced = home if hg > ag else away if ag > hg else None
        lookup[frozenset((home, away))] = {"advanced": advanced}
    return lookup


def compute_knockout_bracket(
    schedule: pl.DataFrame,
    matches: pl.DataFrame,
    model: object,
    structure: object,
    top_n: int = 30,
) -> pl.DataFrame:
    """Per-fixture knockout forecast for the real bracket (advance + FT/ET/pens).

    Args:
        schedule: ingested tournament schedule (``stage``/``home_team``/``away_team``/``date``);
            supplies the real R32/R16 fixtures and the QF→Final ``W##``/``L##`` slot placeholders.
        matches: ingested results; locks already-played knockout ties to their real winner.
        model: ``StrengthModel`` with ``.rates()``, ``.rho`` and ``.max_goals``.
        structure: ``TournamentStructure`` with a ``.penalty_rate`` mapping.
        top_n: cap on the number of possible matchups emitted per fixture (deepest rounds fan out).

    Returns:
        DataFrame with :data:`BRACKET_SCHEMA` columns: one row per possible matchup at each
        fixture, ``rank`` 1 = most-probable occupant of that slot. Empty (typed) frame when the
        schedule carries no knockout fixtures.
    """

    if schedule.is_empty() or "stage" not in schedule.columns:
        return pl.DataFrame(schema=BRACKET_SCHEMA)

    played = _played_lookup(matches)
    nodes = _build_nodes(schedule, played)
    if not nodes:
        return pl.DataFrame(schema=BRACKET_SCHEMA)

    forecaster = _BracketForecaster(nodes, model, structure)
    rows: list[dict[str, object]] = []
    for num in sorted(nodes):
        rows.extend(forecaster.rows_for(num, top_n))

    if not rows:
        return pl.DataFrame(schema=BRACKET_SCHEMA)

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("match_number").cast(pl.Int32), pl.col("rank").cast(pl.Int32))
        .select(list(BRACKET_SCHEMA.keys()))
    )
