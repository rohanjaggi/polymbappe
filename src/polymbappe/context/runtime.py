"""Runtime contextual features shared by training and simulation (spec 2.2, 4.1).

The contextual adjuster must see the *same* feature columns when it is fit (on historical
matches) and when it is applied per simulated match. This module is the single source of
truth for that minimal, data-light feature set — the signals derivable from match results
and Elo alone, so the contextual layer works without extra ingestion:

* ``home_xg_overperf`` / ``away_xg_overperf`` — rolling goals-minus-xG overperformance
  (proxy from goals when real xG is absent; spec 2.2 Group E permanent signal).
* ``draw_pressure`` — group-stage Elo-gap draw signal (spec 2.2 Group F).

The adjuster's per-group toggles map onto :data:`FEATURE_GROUPS`.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.context.draw_pressure import stage_elo_interaction

#: Contextual feature columns, in fixed order, used by the simulation-time adjuster.
SIM_CONTEXT_FEATURES: tuple[str, ...] = ("home_xg_overperf", "away_xg_overperf", "draw_pressure")

#: Group -> columns mapping for the adjuster's toggle gating.
FEATURE_GROUPS: dict[str, list[str]] = {
    "xg_overperformance": ["home_xg_overperf", "away_xg_overperf"],
    "draw_pressure": ["draw_pressure"],
}


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


def fixture_feature_row(
    home: str,
    away: str,
    overperf: dict[str, float],
    elo: dict[str, float],
    *,
    is_knockout: bool = False,
) -> dict[str, float]:
    """Build the contextual feature row for one fixture."""

    gap = elo.get(home, 1500.0) - elo.get(away, 1500.0)
    return {
        "home_xg_overperf": overperf.get(home, 0.0),
        "away_xg_overperf": overperf.get(away, 0.0),
        "draw_pressure": stage_elo_interaction(is_knockout, gap),
    }


def build_tournament_context_features(matches: pl.DataFrame, tournaments: object) -> pl.DataFrame:
    """Per-fixture contextual features (keyed by ``match_id``) for a set of tournaments.

    For each tournament, computes xG-overperformance and Elo as of its start (history
    only), then the per-fixture feature row — the same :data:`SIM_CONTEXT_FEATURES`
    columns the simulation builds at prediction time, so the contextual adjuster sees an
    identical feature set when fit (here) and when applied live. Shared by training
    (:mod:`polymbappe.models.train`) and the backtest objective.
    """

    from polymbappe.eval.backtest import select_fixtures
    from polymbappe.features.elo import build_elo_snapshots

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
        for fx in fixtures.iter_rows(named=True):
            feats = fixture_feature_row(fx["home_team"], fx["away_team"], overperf, elo)
            rows.append({"match_id": fx["match_id"], **feats})
    cols = {"match_id": pl.Utf8, **{c: pl.Float64 for c in SIM_CONTEXT_FEATURES}}
    return pl.DataFrame(rows, schema=cols)
