"""Rolling xG features (Tier 2).

Real team-level xG is available from FBref for 2018+ matches. For earlier periods the
spec calls for an opponent-strength-adjusted goals proxy; here the proxy is the team's
rolling goals scored/conceded (a clean, backtestable stand-in). Both paths are
point-in-time: the current match is excluded from its own window.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.features.context import team_match_long


def build_xg_features(
    matches: pl.DataFrame,
    team_xg: pl.DataFrame | None = None,
    as_of_date: date | None = None,
    window: int = 10,
) -> pl.DataFrame:
    """Rolling xG-for / xG-against per team appearance.

    Args:
        matches: Frame with the ``matches`` schema.
        team_xg: Optional FBref team-match xG table with columns ``[team, date, xg, xga]``.
            When provided, real xG is rolled; otherwise the goals proxy is used.
        as_of_date: When set, only matches strictly before this date are used.
        window: Rolling window length.

    Returns:
        Frame keyed by ``(match_id, team)`` with ``[match_id, team, date, xg_for,
        xg_against, xg_is_proxy]``.
    """

    long = team_match_long(matches, as_of_date)

    if team_xg is not None:
        joined = long.join(
            team_xg.select(
                pl.col("team"),
                pl.col("date").cast(pl.Date),
                pl.col("xg").cast(pl.Float64),
                pl.col("xga").cast(pl.Float64),
            ),
            on=["team", "date"],
            how="left",
        )
        for_col, against_col, is_proxy = "xg", "xga", False
    else:
        joined = long.with_columns(
            pl.col("goals_for").cast(pl.Float64).alias("xg"),
            pl.col("goals_against").cast(pl.Float64).alias("xga"),
        )
        for_col, against_col, is_proxy = "xg", "xga", True

    return (
        joined.with_columns(
            pl.col(for_col)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .over("team")
            .alias("xg_for"),
            pl.col(against_col)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .over("team")
            .alias("xg_against"),
        )
        .with_columns(pl.lit(is_proxy).alias("xg_is_proxy"))
        .select(["match_id", "team", "date", "xg_for", "xg_against", "xg_is_proxy"])
    )
