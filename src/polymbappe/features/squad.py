"""Squad market-value features (Tier 1).

Derived from the ``squad_valuations`` table (Transfermarkt). The per-team log total
value is the model-facing feature; the home/away ratio is formed at join time in the
feature pipeline. Squad valuations are per-tournament (not time-series), so leakage is
controlled by selecting the valuation snapshot for the tournament being predicted.
"""

from __future__ import annotations

import polars as pl


def build_squad_features(valuations: pl.DataFrame) -> pl.DataFrame:
    """Compute per-team log squad value from a ``squad_valuations`` frame.

    Args:
        valuations: Frame with columns ``[team, tournament, total_value, median_value,
            player_count]``.

    Returns:
        Frame keyed by ``(team, tournament)`` with ``[team, tournament, log_total_value,
        log_median_value, player_count]``. ``log1p`` is used so zero values are safe.
    """

    return valuations.select(
        pl.col("team"),
        pl.col("tournament"),
        pl.col("total_value").cast(pl.Float64).log1p().alias("log_total_value"),
        pl.col("median_value").cast(pl.Float64).log1p().alias("log_median_value"),
        pl.col("player_count").cast(pl.Int64),
    )
