"""Canonical normalized table registry.

All ingested data lands as parquet tables under ``data/processed`` and is queryable
through DuckDB (see :mod:`polymbappe.data.store`). This module is the single source
of truth for table names and their on-disk locations so the rest of the codebase
never hard-codes paths.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from polymbappe.config import Settings


class Table(StrEnum):
    """Canonical normalized table names."""

    MATCHES = "matches"
    ELO_SNAPSHOTS = "elo_snapshots"
    MARKET_ODDS = "market_odds"
    SQUAD_VALUATIONS = "squad_valuations"
    SQUADS = "squads"
    MANAGER_RECORDS = "manager_records"
    TEAM_XG = "team_xg"


#: Columns each normalized table is expected to expose. Used for empty-frame
#: construction and as lightweight documentation of the contract producers must meet.
TABLE_COLUMNS: dict[Table, tuple[str, ...]] = {
    Table.MATCHES: (
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "competition",
        "is_knockout",
        "neutral_site",
        "group",
    ),
    Table.ELO_SNAPSHOTS: ("team", "date", "rating"),
    Table.MARKET_ODDS: (
        "match_id",
        "source",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "timestamp",
    ),
    Table.SQUAD_VALUATIONS: (
        "team",
        "tournament",
        "total_value",
        "median_value",
        "player_count",
    ),
    Table.SQUADS: ("team", "tournament", "player", "club", "age"),
    Table.MANAGER_RECORDS: (
        "manager",
        "team",
        "tournament",
        "stage_reached",
        "knockout_matches",
        "knockout_wins",
        "tournament_order",
    ),
    Table.TEAM_XG: ("team", "date", "xg", "xga"),
}


def table_path(table: Table, settings: Settings | None = None) -> Path:
    """Resolve the parquet path for a normalized table."""

    settings = settings or Settings()
    return settings.processed_data_dir / f"{table.value}.parquet"
