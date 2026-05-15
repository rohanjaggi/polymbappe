"""Persistence layer for parquet and DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a dataframe from parquet."""

    return pl.read_parquet(path)


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
