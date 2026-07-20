"""Knockout-bracket forecast anchored to the *real* World Cup draw (spec 4.3).

The bracket is not simulated with a random seeding — it is the shared real tree from
:mod:`polymbappe.simulate.real_bracket`, read from the ingested tournament schedule
(group-position / third-place placeholders for R32, ``W##``/``L##`` slot references for
later rounds) with played results pinned onto it. The same machinery drives the Monte
Carlo engine, so this artifact and ``stage_probabilities`` cannot disagree about the
bracket wiring or who advanced.

From that fixed tree we project forward: each tie's advance probability comes from the
strength model, and the probability a team *reaches* a future slot is propagated down the
bracket (``P(reach) = P(win previous tie)`` chained through the subtree), conditioned on
results already played — including extra-time/penalty ties whose winner is inferred from
later rounds, and, failing that, forecast *beyond regulation* (the 90' draw is a fact).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

from polymbappe.simulate.match import (
    ET_GOAL_SCALE,
    beyond_regulation_home_winprob,
    hda_marginals,
    knockout_outcome_breakdown,
    score_matrix_from_rates,
)
from polymbappe.simulate.real_bracket import (
    BracketNode,
    RealBracket,
    attach_played_results,
    build_real_bracket,
    fill_r32_leaves,
    oriented_pin,
    real_group_positions,
)

logger = structlog.get_logger(__name__)

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

#: Reach-probability entries below this are pruned (then renormalised) to bound the fan-out of
#: possible matchups in the deepest rounds.
_PRUNE_EPS = 1e-4


class _BracketForecaster:
    """Propagates reach/advance probabilities through the fixed bracket tree."""

    def __init__(
        self,
        bracket: RealBracket,
        model: object,
        structure: object,
        positions: dict[str, str] | None,
        qualified_third_groups: list[str] | None,
    ) -> None:
        self._bracket = bracket
        self._model = model
        self._pen: dict[str, float] = getattr(structure, "penalty_rate", {}) or {}
        self._max_goals: int = model.max_goals  # type: ignore[attr-defined]
        self._rho: float = model.rho  # type: ignore[attr-defined]
        self._grid = np.arange(self._max_goals + 1)
        self._advance_cache: dict[tuple[str, str], float] = {}
        self._winner_cache: dict[int, dict[str, float]] = {}
        self._loser_cache: dict[int, dict[str, float]] = {}
        self._team_group = (
            {team: slot[1:] for slot, team in positions.items()} if positions else {}
        )
        # Deterministic R32 leaf occupants once group standings exist (real, not simulated).
        self._r32_leaves: dict[int, tuple[str, str]] | None = None
        if positions is not None and qualified_third_groups is not None:
            self._r32_leaves = fill_r32_leaves(
                bracket, positions, qualified_third_groups, rng=None
            )
        elif bracket.rounds.get("R32"):
            unpinned = [
                n for n in bracket.rounds["R32"] if not bracket.nodes[n].pinned
            ]
            if unpinned:
                logger.info(
                    "knockout_bracket.placeholder_leaves",
                    unpinned=len(unpinned),
                    reason="group stage incomplete; unplayed R32 sides stay as labels",
                )

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

    def _beyond_regulation_prob(self, a: str, b: str) -> float:
        """P(a advances past b | level after 90') — for played but unresolved draws."""

        _reg, et = self._matrices(a, b)
        return beyond_regulation_home_winprob(
            et, self._pen.get(a, 0.5), self._pen.get(b, 0.5)
        )

    # -- reach-distribution propagation ---------------------------------------

    def _side_dist(self, node: BracketNode, side_name: str) -> dict[str, float]:
        """Distribution over which team fills a fixture side."""

        side = node.home if side_name == "home" else node.away
        if side.kind == "winner" and side.ref in self._bracket.nodes:
            return self._winner_dist(side.ref)  # type: ignore[arg-type]
        if side.kind == "loser" and side.ref in self._bracket.nodes:
            # Third-place play-off sides: the losers of the two semi-finals.
            return self._loser_dist(side.ref)  # type: ignore[arg-type]
        if node.pinned:
            home_occ, away_occ = oriented_pin(node, self._team_group)
            return {home_occ if side_name == "home" else away_occ: 1.0}
        if side.kind == "team":
            return {side.team or side.label: 1.0}
        if node.round == "R32" and self._r32_leaves is not None:
            occ = self._r32_leaves[node.number]
            return {occ[0] if side_name == "home" else occ[1]: 1.0}
        return {side.label: 1.0}  # placeholder passthrough (pre-completion)

    def _resolve(self, num: int) -> None:
        """Populate winner/loser caches for one node from its children (memoised)."""

        node = self._bracket.nodes[num]
        if node.winner is not None and node.loser is not None:
            self._winner_cache[num] = {node.winner: 1.0}
            self._loser_cache[num] = {node.loser: 1.0}
            return
        if node.drawn_unresolved and node.pinned:
            a, b = oriented_pin(node, self._team_group)
            p_a = self._beyond_regulation_prob(a, b)
            self._winner_cache[num] = {a: p_a, b: 1.0 - p_a}
            self._loser_cache[num] = {a: 1.0 - p_a, b: p_a}
            return
        a_dist, b_dist = self._side_dist(node, "home"), self._side_dist(node, "away")
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

    def _row(
        self,
        node: BracketNode,
        rank: int,
        a: str,
        b: str,
        matchup_prob: float,
        p_a_advance: float | None = None,
        phase: tuple[float, float, float] | None = None,
    ) -> dict[str, object]:
        reg, et = self._matrices(a, b)
        breakdown = knockout_outcome_breakdown(
            reg, et, self._pen.get(a, 0.5), self._pen.get(b, 0.5)
        )
        h, d, aw = hda_marginals(reg)
        p_a = breakdown.p_home_advance if p_a_advance is None else p_a_advance
        p_reg, p_et, p_pens = (
            (breakdown.p_decided_reg, breakdown.p_decided_et, breakdown.p_decided_pens)
            if phase is None
            else phase
        )
        return {
            "round": node.round,
            "match_number": node.number,
            "rank": rank,
            "team_a": a,
            "team_b": b,
            "matchup_prob": matchup_prob,
            "p_a_advance": p_a,
            "p_b_advance": 1.0 - p_a,
            "p_decided_reg": p_reg,
            "p_decided_et": p_et,
            "p_decided_pens": p_pens,
            "model_a": h,
            "model_draw": d,
            "model_b": aw,
            "exp_a_goals": float((reg.sum(axis=1) * self._grid).sum()),
            "exp_b_goals": float((reg.sum(axis=0) * self._grid).sum()),
        }

    def rows_for(self, node: BracketNode, top_n: int) -> list[dict[str, object]]:
        """Emit one row per possible matchup at a fixture, most-probable first.

        Takes the node itself (not its number) so the third-place play-off — stored
        outside ``bracket.nodes`` — can be forecast through the same path.
        """

        if node.pinned:
            a, b = oriented_pin(node, self._team_group)
            if node.winner is not None:
                # Played and decided (or shootout winner inferred): a known outcome.
                return [self._row(node, 1, a, b, 1.0, p_a_advance=float(node.winner == a))]
            if node.drawn_unresolved:
                # Played, level after 90', winner not yet inferable: forecast the
                # continuation only — regulation is known not to have decided it.
                _reg, et = self._matrices(a, b)
                ed = float(np.trace(et))
                return [
                    self._row(
                        node, 1, a, b, 1.0,
                        p_a_advance=self._beyond_regulation_prob(a, b),
                        phase=(0.0, 1.0 - ed, ed),
                    )
                ]
        a_dist, b_dist = self._side_dist(node, "home"), self._side_dist(node, "away")
        combos = [
            (a, b, pa * pb)
            for a, pa in a_dist.items()
            for b, pb in b_dist.items()
            if a != b
        ]
        combos.sort(key=lambda x: -x[2])
        return [
            self._row(node, rank, a, b, mp)
            for rank, (a, b, mp) in enumerate(combos[:top_n], start=1)
        ]


def _prune(dist: dict[str, float]) -> dict[str, float]:
    """Drop negligible entries and renormalise a reach distribution to sum to 1."""

    kept = {t: p for t, p in dist.items() if p >= _PRUNE_EPS}
    total = sum(kept.values())
    if total <= 0:
        return dist
    return {t: p / total for t, p in kept.items()}


def compute_knockout_bracket(
    schedule: pl.DataFrame,
    matches: pl.DataFrame,
    model: object,
    structure: object,
    top_n: int = 30,
    winner_overrides: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Per-fixture knockout forecast for the real bracket (advance + FT/ET/pens).

    Args:
        schedule: ingested tournament schedule (``stage``/``home_team``/``away_team``/``date``,
            plus ``match_number`` when available); supplies the real bracket tree.
        matches: ingested results; locks already-played knockout ties to their real (or
            shootout-inferred) winner.
        model: ``StrengthModel`` with ``.rates()``, ``.rho`` and ``.max_goals``.
        structure: ``TournamentStructure`` with ``.penalty_rate`` / ``.groups`` mappings.
        top_n: cap on the number of possible matchups emitted per fixture (deepest rounds fan out).
        winner_overrides: manual ``{match_number: winner}`` for played draws inference
            can't settle (final / third place) — see
            :func:`~polymbappe.simulate.real_bracket.load_ko_winner_overrides`.

    Returns:
        DataFrame with :data:`BRACKET_SCHEMA` columns: one row per possible matchup at each
        fixture (the third-place play-off emitted as round ``"THIRD"``), ``rank`` 1 =
        most-probable occupant of that slot. Empty (typed) frame when the schedule carries
        no knockout fixtures.
    """

    bracket = build_real_bracket(schedule)
    if bracket is None:
        return pl.DataFrame(schema=BRACKET_SCHEMA)
    attach_played_results(bracket, matches, winner_overrides)

    positions_qualified = real_group_positions(matches, structure)
    positions, qualified = positions_qualified if positions_qualified else (None, None)

    forecaster = _BracketForecaster(bracket, model, structure, positions, qualified)
    rows: list[dict[str, object]] = []
    for num in sorted(bracket.nodes):
        rows.extend(forecaster.rows_for(bracket.nodes[num], top_n))
    if bracket.third_place is not None:
        rows.extend(forecaster.rows_for(bracket.third_place, top_n))

    if not rows:
        return pl.DataFrame(schema=BRACKET_SCHEMA)

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("match_number").cast(pl.Int32), pl.col("rank").cast(pl.Int32))
        .select(list(BRACKET_SCHEMA.keys()))
    )
