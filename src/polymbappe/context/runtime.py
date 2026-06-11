"""Runtime contextual features shared by training and simulation (spec 2.2, 4.1).

The contextual adjuster must see the *same* feature columns when it is fit (on historical
matches) and when it is applied per simulated match. This module is the single source of
truth for that feature set. The data-light signals are derivable from match results and
Elo alone, so the contextual layer works without extra ingestion:

* ``home_xg_overperf`` / ``away_xg_overperf`` — rolling goals-minus-xG overperformance
  (proxy from goals when real xG is absent; spec 2.2 Group E permanent signal).
* ``draw_pressure`` — group-stage Elo-gap draw signal (spec 2.2 Group F).

When the ``squads`` / ``manager_records`` tables have been ingested, two further groups are
assembled here from point-in-time lookups (history only, leakage-guarded):

* ``cohesion`` — ``home/away_club_cluster_index``, ``home/away_median_age`` (spec 2.2 B).
* ``manager`` — ``home/away_knockout_win_rate``, ``home/away_deepest_run_weighted`` (2.2 C).

All feature production funnels through :func:`fixture_feature_row`, fed by a precomputed
:class:`FixtureContext` bundle of team-level lookups, so the fit frame and the live
simulation hook emit an identical column set. The adjuster's per-group toggles map onto
:data:`FEATURE_GROUPS`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from polymbappe.context.draw_pressure import stage_elo_interaction

#: Contextual feature columns, in fixed order, used by the simulation-time adjuster. The
#: append order is a contract (see orchestration decision #5): the original 3 data-light
#: columns first, then the 4 cohesion columns, then the 4 manager columns.
SIM_CONTEXT_FEATURES: tuple[str, ...] = (
    "home_xg_overperf",
    "away_xg_overperf",
    "draw_pressure",
    "home_club_cluster_index",
    "away_club_cluster_index",
    "home_median_age",
    "away_median_age",
    "home_knockout_win_rate",
    "away_knockout_win_rate",
    "home_deepest_run_weighted",
    "away_deepest_run_weighted",
)

#: Group -> columns mapping for the adjuster's toggle gating. ``cohesion`` / ``manager`` are
#: registered here unconditionally (train/backtest read this constant generically); the
#: *coverage gate* (:func:`gated_feature_groups`) decides whether thin-data groups are kept
#: in the active fit, and the autotuner toggle can re-enable a gated group later.
FEATURE_GROUPS: dict[str, list[str]] = {
    "xg_overperformance": ["home_xg_overperf", "away_xg_overperf"],
    "draw_pressure": ["draw_pressure"],
    "cohesion": [
        "home_club_cluster_index",
        "away_club_cluster_index",
        "home_median_age",
        "away_median_age",
    ],
    "manager": [
        "home_knockout_win_rate",
        "away_knockout_win_rate",
        "home_deepest_run_weighted",
        "away_deepest_run_weighted",
    ],
}

#: Minimum number of tournaments with non-zero data a group needs before the coverage gate
#: keeps it in the active fit (orchestration decision #4).
COVERAGE_GATE_K: int = 3


@dataclass(slots=True)
class FixtureContext:
    """Precomputed team-level lookups feeding :func:`fixture_feature_row`.

    The same bundle is built per tournament for the fit frame and (Phase E) per the live
    2026 snapshot, so both call sites emit an identical column set. ``cohesion`` and
    ``manager`` may be empty (0-fill) when their source tables are absent.
    """

    overperf: dict[str, float] = field(default_factory=dict)
    elo: dict[str, float] = field(default_factory=dict)
    #: team -> (club_cluster_index, median_age)
    cohesion: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: team -> {"knockout_win_rate": float, "deepest_run_weighted": float}
    manager: dict[str, dict[str, float]] = field(default_factory=dict)


def latest_overperformance(
    matches: pl.DataFrame, team_xg: pl.DataFrame | None = None, as_of_date: date | None = None
) -> dict[str, float]:
    """Latest rolling xG-overperformance per team from history (0.0 if none)."""

    from polymbappe.context.sentiment import build_xg_overperformance

    overperf = build_xg_overperformance(matches, team_xg, as_of_date)
    latest = (
        overperf.drop_nulls("xg_overperformance")
        .sort(["team", "date"])
        .group_by("team")
        .agg(pl.col("xg_overperformance").last())
    )
    return {r["team"]: float(r["xg_overperformance"]) for r in latest.iter_rows(named=True)}


def cohesion_lookup(squads: pl.DataFrame, tournament: object) -> dict[str, tuple[float, float]]:
    """Per-team ``(club_cluster_index, median_age)`` for one tournament (history only).

    Filters ``squads`` to its ``tournament == tournament.name`` snapshot — the
    pre-tournament call-up rows, a data contract enforced at ingestion — and runs
    :func:`~polymbappe.context.cohesion.build_cohesion_features`. Teams absent from the
    snapshot are simply omitted (0-filled downstream by :func:`fixture_feature_row`).
    """

    from polymbappe.context.cohesion import build_cohesion_features

    name = tournament.name  # type: ignore[attr-defined]
    snapshot = squads.filter(pl.col("tournament") == name)
    if snapshot.is_empty():
        return {}
    feats = build_cohesion_features(snapshot)
    out: dict[str, tuple[float, float]] = {}
    for r in feats.iter_rows(named=True):
        age = r["median_age"]
        out[r["team"]] = (
            float(r["club_cluster_index"]),
            float(age) if age is not None else 0.0,
        )
    return out


def manager_lookup(records: pl.DataFrame, tournament: object) -> dict[str, dict[str, float]]:
    """Per-team manager pedigree for one tournament, leakage-guarded (decision #1).

    ``Tournament`` has no ``.order``; the cutoff is derived from the records themselves —
    the minimum ``tournament_order`` of rows whose ``tournament == tournament.name``. When
    ``tournament.name`` is absent (the live 2026 case) all records are used (cutoff =
    ``+inf``). Pedigree is computed from
    :func:`~polymbappe.context.manager.build_manager_features` on records strictly
    *before* the cutoff, so a tournament can never see its own record. Each team's manager
    is identified from the team's most-recent record at/just before the cutoff (knowing who
    manages a team at T is not leakage; only their *record* would be).

    Returns ``team -> {"knockout_win_rate", "deepest_run_weighted"}``; teams whose manager
    has no pre-cutoff history are omitted (0-filled downstream).
    """

    from polymbappe.context.manager import build_manager_features

    if records.is_empty():
        return {}

    name = tournament.name  # type: ignore[attr-defined]
    own = records.filter(pl.col("tournament") == name)
    if own.is_empty():
        cutoff = math.inf
    else:
        cutoff = float(own["tournament_order"].min())  # type: ignore[arg-type]

    # Identity: per team, the manager from the latest record at/just before the cutoff. The
    # current-tournament row (order == cutoff) is allowed for identity only.
    identity_src = records.filter(pl.col("tournament_order") <= cutoff)
    team_manager: dict[str, str] = {}
    for (team,), group in identity_src.group_by(["team"]):
        latest = group.sort("tournament_order").row(group.height - 1, named=True)
        team_manager[team] = latest["manager"]

    # Pedigree: strictly pre-cutoff records only (the critical leakage guard).
    history = records.filter(pl.col("tournament_order") < cutoff)
    if history.is_empty():
        return {}
    pedigree = build_manager_features(history)
    by_manager = {
        r["manager"]: {
            "knockout_win_rate": float(r["knockout_win_rate"]),
            "deepest_run_weighted": float(r["deepest_run_weighted"]),
        }
        for r in pedigree.iter_rows(named=True)
    }

    out: dict[str, dict[str, float]] = {}
    for team, manager in team_manager.items():
        ped = by_manager.get(manager)
        if ped is not None:
            out[team] = ped
    return out


def fixture_feature_row(
    home: str,
    away: str,
    ctx: FixtureContext,
    *,
    is_knockout: bool = False,
) -> dict[str, float]:
    """Build the contextual feature row for one fixture from a :class:`FixtureContext`.

    Emits the :data:`SIM_CONTEXT_FEATURES` columns in their fixed order. Missing teams
    yield ``0.0`` explicitly (the adjuster also ``fill_null``s, but emitting 0.0 keeps the
    schema dense).
    """

    gap = ctx.elo.get(home, 1500.0) - ctx.elo.get(away, 1500.0)
    home_coh = ctx.cohesion.get(home, (0.0, 0.0))
    away_coh = ctx.cohesion.get(away, (0.0, 0.0))
    home_mgr = ctx.manager.get(home, {})
    away_mgr = ctx.manager.get(away, {})
    return {
        "home_xg_overperf": ctx.overperf.get(home, 0.0),
        "away_xg_overperf": ctx.overperf.get(away, 0.0),
        "draw_pressure": stage_elo_interaction(is_knockout, gap),
        "home_club_cluster_index": home_coh[0],
        "away_club_cluster_index": away_coh[0],
        "home_median_age": home_coh[1],
        "away_median_age": away_coh[1],
        "home_knockout_win_rate": home_mgr.get("knockout_win_rate", 0.0),
        "away_knockout_win_rate": away_mgr.get("knockout_win_rate", 0.0),
        "home_deepest_run_weighted": home_mgr.get("deepest_run_weighted", 0.0),
        "away_deepest_run_weighted": away_mgr.get("deepest_run_weighted", 0.0),
    }


def gated_feature_groups(context_features: pl.DataFrame) -> dict[str, list[str]]:
    """Coverage-gated copy of :data:`FEATURE_GROUPS` (orchestration decision #4).

    A group is dropped from the active fit unless at least one of its columns carries
    non-zero data for ``>= COVERAGE_GATE_K`` distinct tournaments in ``context_features``.
    Near-constant (all-zero) columns inject noise into the residual adjuster, so thin-data
    groups are omitted until coverage is real; the autotuner toggle can re-enable them.

    Requires a ``tournament`` column in ``context_features`` to count distinct tournaments;
    if absent, all groups pass (the gate is a safeguard, not a hard requirement).
    """

    if "tournament" not in context_features.columns or context_features.is_empty():
        return dict(FEATURE_GROUPS)

    gated: dict[str, list[str]] = {}
    for group, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in context_features.columns]
        if not present:
            continue
        # Always keep the always-available data-light groups; gate the ingestion-backed ones.
        if group not in {"cohesion", "manager"}:
            gated[group] = cols
            continue
        nonzero_expr = pl.any_horizontal([pl.col(c) != 0.0 for c in present])
        covered = (
            context_features.filter(nonzero_expr)
            .select(pl.col("tournament").n_unique())
            .item()
        )
        if covered >= COVERAGE_GATE_K:
            gated[group] = cols
    return gated


def build_tournament_context_features(
    matches: pl.DataFrame,
    tournaments: object,
    settings: object | None = None,
    *,
    include_tournament: bool = False,
) -> pl.DataFrame:
    """Per-fixture contextual features (keyed by ``match_id``) for a set of tournaments.

    For each tournament, computes xG-overperformance and Elo as of its start (history
    only), plus point-in-time cohesion / manager lookups when the ``squads`` /
    ``manager_records`` tables are present, then the per-fixture feature row — the same
    :data:`SIM_CONTEXT_FEATURES` columns the simulation builds at prediction time, so the
    contextual adjuster sees an identical feature set when fit (here) and applied live.

    The signature stays backward-compatible: callers passing only ``(matches, tournaments)``
    get cohesion/manager 0-filled when the tables are absent. Tables are read via the
    default :class:`~polymbappe.config.Settings` (or ``settings`` if given) and skipped
    gracefully when missing.

    The public output is ``match_id`` + the 11 :data:`SIM_CONTEXT_FEATURES` columns. Pass
    ``include_tournament=True`` to retain the assembly-internal ``tournament`` column for
    :func:`gated_feature_groups` (the coverage gate).
    """

    from polymbappe.eval.backtest import select_fixtures
    from polymbappe.features.elo import build_elo_snapshots

    squads, records = _load_context_tables(settings)

    rows: list[dict[str, object]] = []
    for tournament in tournaments:  # type: ignore[attr-defined]
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            continue
        overperf = latest_overperformance(history)
        snaps = (
            build_elo_snapshots(history)
            .sort(["team", "date"])
            .group_by("team")
            .agg(pl.col("rating").last())
        )
        elo = {r["team"]: float(r["rating"]) for r in snaps.iter_rows(named=True)}
        ctx = FixtureContext(
            overperf=overperf,
            elo=elo,
            cohesion=cohesion_lookup(squads, tournament) if squads is not None else {},
            manager=manager_lookup(records, tournament) if records is not None else {},
        )
        for fx in fixtures.iter_rows(named=True):
            feats = fixture_feature_row(fx["home_team"], fx["away_team"], ctx)
            rows.append(
                {"match_id": fx["match_id"], "tournament": tournament.name, **feats}
            )
    cols = {
        "match_id": pl.Utf8,
        "tournament": pl.Utf8,
        **{c: pl.Float64 for c in SIM_CONTEXT_FEATURES},
    }
    frame = pl.DataFrame(rows, schema=cols)
    if include_tournament:
        return frame
    # Public contract: ``match_id`` + the 11 ``SIM_CONTEXT_FEATURES`` columns only. The
    # assembly-internal ``tournament`` column (used by the coverage gate) is dropped so the
    # generic consumers — which join on ``match_id`` then select feature columns — see no
    # extra column and cannot collide with the base frame's own ``tournament``.
    return frame.drop("tournament")


def _load_context_tables(
    settings: object | None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Read the ``squads`` / ``manager_records`` tables when materialized, else ``None``."""

    try:
        from polymbappe.config import Settings
        from polymbappe.data.store import read_table, table_exists
        from polymbappe.data.tables import Table
    except Exception:  # noqa: BLE001 - data layer optional in minimal test contexts
        return None, None

    resolved = settings if settings is not None else Settings()
    squads = (
        read_table(Table.SQUADS, resolved)  # type: ignore[arg-type]
        if table_exists(Table.SQUADS, resolved)  # type: ignore[arg-type]
        else None
    )
    records = (
        read_table(Table.MANAGER_RECORDS, resolved)  # type: ignore[arg-type]
        if table_exists(Table.MANAGER_RECORDS, resolved)  # type: ignore[arg-type]
        else None
    )
    return squads, records
