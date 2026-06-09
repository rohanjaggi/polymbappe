"""Agent state persistence in DuckDB (spec section 5.3).

Four persistent tables back the live agent:

* ``agent_runs`` — one row per cycle (timestamp, duration, items scanned / acted on).
* ``agent_player_statuses`` — current availability per player (the agent's memory).
* ``agent_changelog`` — timestamped change log with the reasoning chain.
* ``agent_decisions`` — full per-node decision trace for dashboard transparency.

State lives in a persistent DuckDB database file (``data/processed/agent_state.duckdb``).
The changelog is additionally exported to ``data/outputs/agent_changelog.parquet`` for the
dashboard's Agent Activity page, with the columns that page expects
(``timestamp, team, player, change, reasoning, prob_shift``).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from polymbappe.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id BIGINT, timestamp TIMESTAMP, duration_s DOUBLE,
    items_scanned INTEGER, items_acted INTEGER
);
CREATE TABLE IF NOT EXISTS agent_player_statuses (
    player VARCHAR, team VARCHAR, status VARCHAR, last_updated TIMESTAMP,
    source VARCHAR, confidence DOUBLE
);
CREATE TABLE IF NOT EXISTS agent_changelog (
    timestamp TIMESTAMP, team VARCHAR, player VARCHAR, change VARCHAR,
    reasoning VARCHAR, prob_shift DOUBLE
);
CREATE TABLE IF NOT EXISTS agent_decisions (
    run_id BIGINT, timestamp TIMESTAMP, node VARCHAR, payload VARCHAR
);
"""


class AgentState:
    """DuckDB-backed persistent state for the live agent."""

    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.path = path or (self.settings.processed_data_dir / "agent_state.duckdb")
        self.con = duckdb.connect(str(self.path))
        self.con.execute(_SCHEMA)

    # -- runs ------------------------------------------------------------------

    def record_run(
        self,
        timestamp: datetime,
        duration_s: float,
        items_scanned: int,
        items_acted: int,
    ) -> int:
        """Insert a cycle record and return its run id."""

        next_id = self.con.execute(
            "SELECT COALESCE(MAX(run_id), 0) + 1 FROM agent_runs"
        ).fetchone()[0]
        self.con.execute(
            "INSERT INTO agent_runs VALUES (?, ?, ?, ?, ?)",
            [next_id, timestamp, duration_s, items_scanned, items_acted],
        )
        return int(next_id)

    # -- player statuses -------------------------------------------------------

    def get_player_status(self, player: str) -> dict[str, Any] | None:
        """Current status row for a player, or None."""

        row = self.con.execute(
            "SELECT player, team, status, last_updated, source, confidence "
            "FROM agent_player_statuses WHERE player = ?",
            [player],
        ).fetchone()
        if row is None:
            return None
        cols = ["player", "team", "status", "last_updated", "source", "confidence"]
        return dict(zip(cols, row, strict=True))

    def upsert_player_status(
        self,
        player: str,
        team: str,
        status: str,
        last_updated: datetime,
        source: str,
        confidence: float,
    ) -> None:
        """Insert or replace a player's status (keyed by player)."""

        self.con.execute("DELETE FROM agent_player_statuses WHERE player = ?", [player])
        self.con.execute(
            "INSERT INTO agent_player_statuses VALUES (?, ?, ?, ?, ?, ?)",
            [player, team, status, last_updated, source, confidence],
        )

    def recently_assessed(self, player: str, now: datetime, within_hours: float = 12.0) -> bool:
        """Whether ``player`` was updated within the cooling period (spec 5.2)."""

        current = self.get_player_status(player)
        if current is None:
            return False
        last = current["last_updated"]
        if not isinstance(last, datetime):
            last = datetime.fromisoformat(str(last))
        return (now - last) < timedelta(hours=within_hours)

    # -- changelog / decisions -------------------------------------------------

    def append_changelog(
        self,
        timestamp: datetime,
        team: str,
        player: str,
        change: str,
        reasoning: str,
        prob_shift: float = 0.0,
    ) -> None:
        """Append a changelog entry."""

        self.con.execute(
            "INSERT INTO agent_changelog VALUES (?, ?, ?, ?, ?, ?)",
            [timestamp, team, player, change, reasoning, prob_shift],
        )

    def record_decision(
        self, run_id: int, timestamp: datetime, node: str, payload: dict[str, Any]
    ) -> None:
        """Record one node's decision payload (JSON) for the dashboard trace."""

        self.con.execute(
            "INSERT INTO agent_decisions VALUES (?, ?, ?, ?)",
            [run_id, timestamp, node, json.dumps(payload, default=str)],
        )

    # -- reads -----------------------------------------------------------------

    def _df(self, table: str) -> pl.DataFrame:
        return self.con.execute(f"SELECT * FROM {table}").pl()

    def runs_df(self) -> pl.DataFrame:
        return self._df("agent_runs")

    def player_statuses_df(self) -> pl.DataFrame:
        return self._df("agent_player_statuses")

    def changelog_df(self) -> pl.DataFrame:
        return self._df("agent_changelog")

    def decisions_df(self) -> pl.DataFrame:
        return self._df("agent_decisions")

    def export_changelog_parquet(self) -> Path:
        """Export the changelog to ``data/outputs/agent_changelog.parquet`` for the dashboard."""

        self.settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
        out = self.settings.outputs_data_dir / "agent_changelog.parquet"
        self.changelog_df().write_parquet(out)
        return out

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> AgentState:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
