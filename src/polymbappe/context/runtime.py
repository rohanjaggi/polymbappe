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
