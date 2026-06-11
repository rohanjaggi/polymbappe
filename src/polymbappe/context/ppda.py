"""Pressing intensity (PPDA) features (spec 2.2 Group A).

PPDA — passes allowed per defensive action — is FBref's pressing-intensity metric, lower
meaning a more aggressive high press. The contextual feature is the raw match-level
difference ``PPDA_home - PPDA_away``; PPDA *similarity* additionally feeds the draw-pressure
group (two similar pressing styles draw more).

Real PPDA is available from FBref for 2018+ matches. Earlier periods have no PPDA; the
builder marks those rows so the adjuster can rely on its other features there. Both paths
are point-in-time (the current match is excluded from its own rolling window), mirroring
:mod:`polymbappe.features.xg`.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.features.context import team_match_long

#: Reasonable ceiling for the PPDA range used to normalize the similarity score.
DEFAULT_PPDA_RANGE: float = 20.0


def build_ppda_features(
    matches: pl.DataFrame,
    team_ppda: pl.DataFrame | None = None,
    as_of_date: date | None = None,
    window: int = 10,
) -> pl.DataFrame:
    """Rolling team-level PPDA per match appearance.

    Args:
        matches: Frame with the ``matches`` schema.
        team_ppda: Optional FBref team-match PPDA table ``[team, date, ppda]``. When
            absent, every row's PPDA is null and ``ppda_available`` is False.
        as_of_date: When set, only matches strictly before this date are used.
        window: Rolling window length.

    Returns:
        Frame keyed by ``(match_id, team)`` with ``[match_id, team, date, ppda,
        ppda_available]``.
    """

    long = team_match_long(matches, as_of_date)
    if team_ppda is None:
        return long.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ppda"),
            pl.lit(False).alias("ppda_available"),
        ).select(["match_id", "team", "date", "ppda", "ppda_available"])

    joined = long.join(
        team_ppda.select(
            pl.col("team"), pl.col("date").cast(pl.Date), pl.col("ppda").cast(pl.Float64)
        ),
        on=["team", "date"],
        how="left",
    )
    return (
        joined.with_columns(
            pl.col("ppda")
            .shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .over("team")
            .alias("ppda_roll")
        )
        .with_columns(
            pl.col("ppda_roll").alias("ppda"),
            pl.col("ppda_roll").is_not_null().alias("ppda_available"),
        )
        .select(["match_id", "team", "date", "ppda", "ppda_available"])
    )


def ppda_difference(home_ppda: float | None, away_ppda: float | None) -> float | None:
    """Raw pressing difference ``PPDA_home - PPDA_away`` (None if either is missing)."""

    if home_ppda is None or away_ppda is None:
        return None
    return float(home_ppda) - float(away_ppda)


def ppda_similarity(
    home_ppda: float | None,
    away_ppda: float | None,
    max_range: float = DEFAULT_PPDA_RANGE,
) -> float | None:
    """Similarity in ``[0, 1]``: ``1 - |PPDA_home - PPDA_away| / max_range``.

    1 means identical pressing styles (draw-prone), 0 means maximally different. Clamped
    to ``[0, 1]``. Returns None when either PPDA is missing.
    """

    diff = ppda_difference(home_ppda, away_ppda)
    if diff is None:
        return None
    return float(max(0.0, min(1.0, 1.0 - abs(diff) / max_range)))
