"""Pure data-access layer for the Streamlit dashboard (spec sections 6.2 & 11).

Reads the forecasting engine's parquet output artifacts from ``data/outputs`` via
Polars and exposes them as Polars DataFrames. This module is deliberately free of
any ``streamlit``/``plotly`` import so it can be unit-tested in isolation — the
Streamlit layer (spec section 6.1) consumes these functions.

Output artifacts (spec section 11):

- ``stage_probabilities.parquet`` — team x stage-reaching probabilities
- ``group_probabilities.parquet`` — team x group-finish probabilities
- ``match_predictions.parquet`` — per-match H/D/A probabilities
- ``edges.parquet`` — model vs. market divergences (spec 3.6)
- ``agent_changelog.parquet`` — LangGraph agent activity history (spec 5.3)

Every loader returns an **empty** DataFrame with the correct schema when its
artifact is missing, so pages can render a graceful "no data yet" state rather
than crashing before the first ``polymbappe simulate`` run.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import polars as pl

from polymbappe.config import Settings

#: Schemas for each output artifact (spec section 11). Used to construct correctly
#: typed empty frames when an artifact has not been produced yet, and to document
#: the contract the producing pipelines must meet.
STAGE_SCHEMA: dict[str, pl.DataType] = {
    "team": pl.Utf8,
    "R32": pl.Float64,
    "R16": pl.Float64,
    "QF": pl.Float64,
    "SF": pl.Float64,
    "FINAL": pl.Float64,
    "champion": pl.Float64,
}

GROUP_SCHEMA: dict[str, pl.DataType] = {
    "team": pl.Utf8,
    "finish_1": pl.Float64,
    "finish_2": pl.Float64,
    "finish_3": pl.Float64,
    "finish_4": pl.Float64,
}

MATCH_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "group": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "model_home": pl.Float64,
    "model_draw": pl.Float64,
    "model_away": pl.Float64,
}

EDGES_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "outcome": pl.Utf8,
    "model_prob": pl.Float64,
    "market_prob": pl.Float64,
    "edge": pl.Float64,
    "edge_bps": pl.Float64,
    "kelly_fraction": pl.Float64,
}

CHANGELOG_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Utf8,
    "team": pl.Utf8,
    "player": pl.Utf8,
    "change": pl.Utf8,
    "reasoning": pl.Utf8,
    "prob_shift": pl.Float64,
}

#: Stage-reaching column order, broadest to narrowest (mirrors simulate.tournament.STAGES).
STAGE_COLUMNS: tuple[str, ...] = ("R32", "R16", "QF", "SF", "FINAL", "champion")


def _empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Construct an empty DataFrame with the given schema."""

    return pl.DataFrame(schema=schema)


def _read_or_empty(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Read a parquet artifact, or return a typed empty frame if it is absent.

    Keeps the dashboard resilient before the first ``polymbappe simulate`` run
    (spec section 6.2) — a missing file is an expected state, not an error.
    """

    if not path.exists():
        return _empty(schema)
    return pl.read_parquet(path)


def _output_path(settings: Settings, filename: str) -> Path:
    """Resolve a parquet artifact path under ``data/outputs`` (spec section 11)."""

    return settings.outputs_data_dir / filename


# -- loaders ------------------------------------------------------------------


def load_stage_probabilities(settings: Settings) -> pl.DataFrame:
    """Load per-team stage-reaching probabilities (``stage_probabilities.parquet``).

    Columns: ``team, R32, R16, QF, SF, FINAL, champion`` (spec 4.3 / 11).
    """

    return _read_or_empty(_output_path(settings, "stage_probabilities.parquet"), STAGE_SCHEMA)


def load_group_probabilities(settings: Settings) -> pl.DataFrame:
    """Load per-team group-finish probabilities (``group_probabilities.parquet``).

    Columns: ``team, finish_1, finish_2, finish_3, finish_4`` (spec 4.3 / 11).
    """

    return _read_or_empty(_output_path(settings, "group_probabilities.parquet"), GROUP_SCHEMA)


def load_match_predictions(settings: Settings) -> pl.DataFrame:
    """Load per-match H/D/A predictions (``match_predictions.parquet``).

    Columns: ``match_id, group, home_team, away_team, model_home, model_draw,
    model_away`` (spec 4.3 / 11).
    """

    return _read_or_empty(_output_path(settings, "match_predictions.parquet"), MATCH_SCHEMA)


def load_edges(settings: Settings) -> pl.DataFrame:
    """Load market-edge rows (``edges.parquet``, spec 3.6 / 11).

    Columns: ``match_id, outcome, model_prob, market_prob, edge, edge_bps,
    kelly_fraction`` (mirrors :func:`polymbappe.eval.market.compute_edges`).
    """

    return _read_or_empty(_output_path(settings, "edges.parquet"), EDGES_SCHEMA)


def load_agent_changelog(settings: Settings) -> pl.DataFrame:
    """Load the LangGraph agent activity history (``agent_changelog.parquet``).

    Columns: ``timestamp, team, player, change, reasoning, prob_shift``
    (spec 5.3 / 11). Drives the Agent Activity page (spec 6.1, page 6).
    """

    return _read_or_empty(_output_path(settings, "agent_changelog.parquet"), CHANGELOG_SCHEMA)


# -- helpers ------------------------------------------------------------------


def top_contenders(df: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """Return the ``n`` teams most likely to win the trophy (spec 6.1, page 1).

    Sorts the stage-probabilities frame by ``champion`` descending. Returns an
    empty frame unchanged so callers can render a "no data" state.
    """

    if df.is_empty() or "champion" not in df.columns:
        return df
    return df.sort("champion", descending=True).head(n)


def available_teams(stage_df: pl.DataFrame) -> list[str]:
    """Sorted list of teams present in the stage-probabilities frame.

    Powers the team-selector dropdowns (spec 6.1, pages 2 & 3).
    """

    if stage_df.is_empty() or "team" not in stage_df.columns:
        return []
    return sorted(stage_df["team"].unique().to_list())


def team_stage_row(stage_df: pl.DataFrame, team: str) -> dict[str, float]:
    """Stage-reaching probabilities for a single team as a ``stage -> prob`` map.

    Feeds the stage-reaching waterfall on the Team Deep Dive page (spec 6.1,
    page 2). Returns an empty mapping if the team is absent.
    """

    if stage_df.is_empty() or "team" not in stage_df.columns:
        return {}
    row = stage_df.filter(pl.col("team") == team)
    if row.is_empty():
        return {}
    record = row.row(0, named=True)
    return {stage: float(record[stage]) for stage in STAGE_COLUMNS if stage in record}


def match_row(match_df: pl.DataFrame, home_team: str, away_team: str) -> dict[str, object] | None:
    """Look up the prediction row for a specific (home, away) fixture.

    Powers the Match Predictor page (spec 6.1, page 3). Returns ``None`` when no
    such fixture exists in the predictions table.
    """

    if match_df.is_empty():
        return None
    hit = match_df.filter(
        (pl.col("home_team") == home_team) & (pl.col("away_team") == away_team)
    )
    if hit.is_empty():
        return None
    return hit.row(0, named=True)


def upset_candidates(
    stage_df: pl.DataFrame,
    elo: dict[str, float] | None = None,
    *,
    n: int = 15,
    min_elo_gap: float = 300.0,
) -> pl.DataFrame:
    """Rank teams whose advancement chances look high relative to their Elo (spec 6.1, page 5).

    "Upset Watch" surfaces underdogs the simulation gives an unusually strong run
    to. We score each team by ``R16`` advancement probability per unit of Elo
    deficit relative to the strongest team in the field — a team far below the top
    Elo that still advances often is a live upset candidate.

    Args:
        stage_df: Stage-reaching probabilities frame.
        elo: Optional ``team -> Elo`` map. When provided, ``elo_gap`` (deficit vs.
            the max Elo) is added and only teams with a gap of at least
            ``min_elo_gap`` are considered. When absent, ranks purely by ``R16``.
        n: Maximum rows to return.
        min_elo_gap: Minimum Elo deficit (vs. field max) to qualify as an underdog.

    Returns:
        A frame sorted by upset score (descending), at most ``n`` rows.
    """

    if stage_df.is_empty() or "team" not in stage_df.columns or "R16" not in stage_df.columns:
        return stage_df

    df = stage_df
    if elo:
        max_elo = max(elo.values())
        df = df.with_columns(
            pl.col("team")
            .map_elements(lambda t: max_elo - elo.get(t, max_elo), return_dtype=pl.Float64)
            .alias("elo_gap")
        )
        df = df.filter(pl.col("elo_gap") >= min_elo_gap)
        if df.is_empty():
            return df
        df = df.with_columns(
            (pl.col("R16") * (pl.col("elo_gap") / max(max_elo, 1.0))).alias("upset_score")
        )
        sort_col = "upset_score"
    else:
        df = df.with_columns(pl.col("R16").alias("upset_score"))
        sort_col = "upset_score"

    return df.sort(sort_col, descending=True).head(n)


def edges_by_priority(edges_df: pl.DataFrame, n: int | None = None) -> pl.DataFrame:
    """Sort edges by ``|edge_bps| * kelly_fraction`` (spec 6.1, page 4).

    The Market Edges page sorts by edge magnitude weighted by conviction (Kelly
    stake stands in for the "confidence" weighting). Returns an empty frame
    unchanged.
    """

    if edges_df.is_empty() or "edge_bps" not in edges_df.columns:
        return edges_df
    scored = edges_df.with_columns(
        (pl.col("edge_bps").abs() * pl.col("kelly_fraction")).alias("priority")
    ).sort("priority", descending=True)
    if n is not None:
        scored = scored.head(n)
    return scored


def data_freshness(settings: Settings) -> dict[str, str]:
    """Map each output artifact to its last-modified timestamp (spec 6.1, page 1).

    Surfaces "last simulation timestamp / data freshness" on the Overview page.
    Missing artifacts map to ``"missing"``.
    """

    from datetime import datetime

    artifacts = (
        "stage_probabilities.parquet",
        "group_probabilities.parquet",
        "match_predictions.parquet",
        "edges.parquet",
        "agent_changelog.parquet",
    )
    freshness: dict[str, str] = {}
    for name in artifacts:
        path = _output_path(settings, name)
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            freshness[name] = mtime.isoformat(timespec="seconds")
        else:
            freshness[name] = "missing"
    return freshness
