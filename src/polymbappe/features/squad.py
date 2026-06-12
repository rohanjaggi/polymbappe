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


def build_squad_match_features(
    matches: pl.DataFrame,
    valuations: pl.DataFrame,
    tournaments: object,
) -> pl.DataFrame:
    """Map per-``(team, tournament)`` squad valuations onto each tournament's fixtures.

    Squad valuations are per-tournament snapshots (pre-tournament call-ups), so each fixture
    is assigned *its own* tournament's snapshot — point-in-time safe by construction. The
    returned ``(match_id, team, date, log_total_value)`` table plugs into the feature
    pipeline's team-table join, which forms the home/away ``squad_value_ratio`` at join time
    (spec 2.1 Tier 1: ``log(value_home / value_away)``).

    Args:
        matches: Frame with the ``matches`` schema (``competition``/``date`` locate each
            fixture's tournament window).
        valuations: ``squad_valuations`` frame (per :func:`build_squad_features`).
        tournaments: Iterable of backtest ``Tournament`` objects supplying each snapshot's
            ``name`` plus competition/date window.

    Returns:
        ``(match_id, team, date, log_total_value)``, one row per fixture-team that has a
        snapshot value. Teams absent from a snapshot (or fixtures outside every tournament
        window) yield no row and are left null downstream. Empty, correctly-typed frame when
        no fixture matches a valuation.
    """

    from polymbappe.eval.backtest import select_fixtures

    per_team = build_squad_features(valuations)
    rows: list[dict[str, object]] = []
    for tournament in tournaments:  # type: ignore[attr-defined]
        snapshot = per_team.filter(pl.col("tournament") == tournament.name)
        if snapshot.is_empty():
            continue
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        value = {r["team"]: r["log_total_value"] for r in snapshot.iter_rows(named=True)}
        for fx in fixtures.iter_rows(named=True):
            for side in ("home_team", "away_team"):
                team = fx[side]
                if team in value:
                    rows.append(
                        {
                            "match_id": fx["match_id"],
                            "team": team,
                            "date": fx["date"],
                            "log_total_value": value[team],
                        }
                    )
    return pl.DataFrame(
        rows,
        schema={
            "match_id": pl.Utf8,
            "team": pl.Utf8,
            "date": pl.Date,
            "log_total_value": pl.Float64,
        },
    )
