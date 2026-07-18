"""Persistence layer for parquet and DuckDB.

Normalized tables live as parquet under ``data/processed`` (one file per
:class:`~polymbappe.data.tables.Table`) and are exposed to DuckDB as views via
:func:`connect`, satisfying "all data lands in DuckDB" while staying file-backed
and trivially inspectable.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from polymbappe.config import Settings
from polymbappe.data.tables import Table, table_path


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a dataframe from parquet."""

    return pl.read_parquet(path)


# Natural keys used to de-duplicate append-mode writes. A re-ingested row whose key
# already exists REPLACES the stored row (newest wins) — e.g. a match whose inferred
# ``is_knockout`` flag flipped, or a refreshed odds quote — instead of accumulating
# near-duplicate full rows that fan out every downstream join.
_APPEND_KEYS: dict[Table, tuple[str, ...]] = {
    Table.MATCHES: ("match_id",),
    Table.MARKET_ODDS: ("match_id", "source"),
}


def write_table(
    table: Table, df: pl.DataFrame, *, mode: str = "overwrite", settings: Settings | None = None
) -> Path:
    """Persist a normalized table to its canonical parquet path.

    Args:
        table: Canonical table to write.
        df: Normalized frame.
        mode: ``"overwrite"`` (default) or ``"append"``. Append concatenates with the
            existing table (if any), then de-duplicates on the table's natural key
            (see ``_APPEND_KEYS``, newest row wins) or on identical full rows for
            tables without one.
        settings: Optional settings override (path resolution).

    Returns:
        The path written to.
    """

    path = table_path(table, settings)
    if mode == "append" and path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, df.select(existing.columns)], how="vertical")
        keys = _APPEND_KEYS.get(table)
        if keys is not None:
            df = combined.unique(subset=list(keys), keep="last", maintain_order=True)
        else:
            df = combined.unique(maintain_order=True)
    elif mode not in {"overwrite", "append"}:
        raise ValueError(f"Unknown write mode: {mode!r}")
    write_parquet(df, path)
    return path


def read_table(table: Table, settings: Settings | None = None) -> pl.DataFrame:
    """Read a normalized table from its canonical parquet path."""

    return pl.read_parquet(table_path(table, settings))


def table_exists(table: Table, settings: Settings | None = None) -> bool:
    """Whether a normalized table has been materialized."""

    return table_path(table, settings).exists()


def connect(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with every materialized table registered as a view.

    View names match :class:`~polymbappe.data.tables.Table` values, e.g. ``matches``.
    Missing tables are simply not registered.
    """

    con = duckdb.connect()
    for table in Table:
        path = table_path(table, settings)
        if path.exists():
            con.execute(
                f"CREATE VIEW {table.value} AS SELECT * FROM read_parquet('{path.as_posix()}')"
            )
    return con


def query_duckdb(sql: str, parquet_paths: list[Path]) -> pl.DataFrame:
    """Run ad-hoc SQL over parquet files with DuckDB."""

    con = duckdb.connect()
    try:
        for idx, path in enumerate(parquet_paths):
            con.execute(f"CREATE VIEW t{idx} AS SELECT * FROM read_parquet('{path.as_posix()}')")
        result = pl.from_arrow(con.execute(sql).fetch_arrow_table())
        if not isinstance(result, pl.DataFrame):
            raise TypeError("Expected DuckDB query to return a tabular dataframe.")
        return result
    finally:
        con.close()
