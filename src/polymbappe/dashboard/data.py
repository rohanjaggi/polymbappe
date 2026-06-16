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

import math
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

KO_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int32,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "matchup_prob": pl.Float64,
    "model_home": pl.Float64,
    "model_draw": pl.Float64,
    "model_away": pl.Float64,
    "exp_home_goals": pl.Float64,
    "exp_away_goals": pl.Float64,
}

CHANGELOG_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Utf8,
    "team": pl.Utf8,
    "player": pl.Utf8,
    "change": pl.Utf8,
    "reasoning": pl.Utf8,
    "prob_shift": pl.Float64,
}

#: Recorded-results schema — the subset of the ``matches`` normalized table (spec 11)
#: the Match Predictor page needs to mark a scheduled fixture as played and show its
#: scoreline. Mirrors ``polymbappe.data.tables.TABLE_COLUMNS[Table.MATCHES]``.
RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "date": pl.Date,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_goals": pl.Int64,
    "away_goals": pl.Int64,
    "competition": pl.Utf8,
    "is_knockout": pl.Boolean,
    "neutral_site": pl.Boolean,
    "group": pl.Utf8,
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


def load_knockout_predictions(settings: Settings) -> pl.DataFrame:
    """Load R32 probable matchups (``knockout_predictions.parquet``).

    Columns: ``rank, home_team, away_team, matchup_prob, model_home, model_draw,
    model_away, exp_home_goals, exp_away_goals``.
    """

    return _read_or_empty(_output_path(settings, "knockout_predictions.parquet"), KO_SCHEMA)


def load_recorded_results(settings: Settings) -> pl.DataFrame:
    """Load played-match results from the ``matches`` normalized table (spec 11).

    Unlike the other loaders this reads ``data/processed`` (ingested results), not
    ``data/outputs`` (simulation artifacts). Returns a typed empty frame when no
    matches have been ingested yet, so the Match Predictor page degrades gracefully.
    """

    from polymbappe.data.tables import Table, table_path

    return _read_or_empty(table_path(Table.MATCHES, settings), RESULTS_SCHEMA)


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


def _model_pick_expr() -> pl.Expr:
    """Polars expression for the model's most-likely outcome (``home``/``draw``/``away``).

    Ties break toward the home win, then the away win — matching how the H/D/A bar is
    read. Operates on the ``model_home``/``model_draw``/``model_away`` columns.
    """

    return (
        pl.when(
            (pl.col("model_home") >= pl.col("model_draw"))
            & (pl.col("model_home") >= pl.col("model_away"))
        )
        .then(pl.lit("home"))
        .when(
            (pl.col("model_away") >= pl.col("model_draw"))
            & (pl.col("model_away") >= pl.col("model_home"))
        )
        .then(pl.lit("away"))
        .otherwise(pl.lit("draw"))
    )


def tournament_results(
    results_df: pl.DataFrame,
    *,
    year: int = 2026,
    competition_substr: str | None = None,
) -> pl.DataFrame:
    """Narrow recorded results to the tournament currently being forecast.

    A scheduled fixture (e.g. group-stage ``Brazil vs Serbia``) shares its team pair
    with decades of historical friendlies, so we must restrict recorded results to the
    tournament before treating a match as "played". The discriminator is the match
    ``date`` year (``>= year``); an optional case-insensitive ``competition`` substring
    narrows further when the source labels the competition reliably.

    Returns the (possibly empty) filtered frame; an empty input passes through unchanged.
    """

    if results_df.is_empty():
        return results_df
    df = results_df
    if "date" in df.columns:
        df = df.filter(pl.col("date").dt.year() >= year)
    if competition_substr and "competition" in df.columns:
        df = df.filter(
            pl.col("competition").cast(pl.Utf8).str.contains(f"(?i){competition_substr}")
        )
    return df


def split_fixtures(
    match_df: pl.DataFrame, results_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split scheduled fixtures into ``(upcoming, finished)`` using recorded results.

    Powers the Match Predictor page (spec 6.1, page 3): the page only forecasts real
    fixtures, so we partition the predictions table by whether a recorded result exists
    for the same ``(home_team, away_team)`` pairing in ``results_df`` (which the caller
    should first narrow with :func:`tournament_results`). When a pairing has multiple
    recorded results the most recent by ``date`` wins.

    Both frames carry a ``model_pick`` column (the model's favoured outcome). The
    ``finished`` frame additionally carries ``home_goals``, ``away_goals``, ``date``,
    ``actual_outcome`` and ``model_correct`` (whether the model's pick matched reality).
    Returns the input frame for both halves when there are no fixtures.
    """

    if match_df.is_empty():
        return match_df, match_df

    fixtures = match_df.with_columns(_model_pick_expr().alias("model_pick"))

    result_cols = ["home_team", "away_team", "home_goals", "away_goals", "date"]
    if results_df.is_empty():
        results_slim = pl.DataFrame(
            schema={c: RESULTS_SCHEMA[c] for c in result_cols}
        )
    else:
        results_slim = (
            results_df.select(result_cols)
            .sort("date")
            .group_by(["home_team", "away_team"], maintain_order=True)
            .last()
        )

    joined = fixtures.join(results_slim, on=["home_team", "away_team"], how="left")
    played = pl.col("home_goals").is_not_null()

    upcoming = joined.filter(~played).select(fixtures.columns)
    finished = (
        joined.filter(played)
        .with_columns(
            pl.when(pl.col("home_goals") > pl.col("away_goals"))
            .then(pl.lit("home"))
            .when(pl.col("home_goals") < pl.col("away_goals"))
            .then(pl.lit("away"))
            .otherwise(pl.lit("draw"))
            .alias("actual_outcome")
        )
        .with_columns(
            (pl.col("model_pick") == pl.col("actual_outcome")).alias("model_correct")
        )
    )
    return upcoming, finished


def all_fixtures_with_results(
    match_df: pl.DataFrame, results_df: pl.DataFrame
) -> pl.DataFrame:
    """All fixtures with predictions; unplayed rows have null result columns.

    Like :func:`split_fixtures` but returns a single unified frame instead of two.
    Unplayed fixtures carry null ``home_goals``/``away_goals``/``actual_outcome``/
    ``model_correct`` so the Match Predictor page can render one table with ⏳ for
    pending matches and ✅/❌ for finished ones.
    """

    if match_df.is_empty():
        return match_df

    fixtures = match_df.with_columns(_model_pick_expr().alias("model_pick"))

    result_cols = ["home_team", "away_team", "home_goals", "away_goals", "date"]
    if results_df.is_empty():
        results_slim = pl.DataFrame(schema={c: RESULTS_SCHEMA[c] for c in result_cols})
    else:
        results_slim = (
            results_df.select(result_cols)
            .sort("date")
            .group_by(["home_team", "away_team"], maintain_order=True)
            .last()
        )

    joined = fixtures.join(results_slim, on=["home_team", "away_team"], how="left")
    played = pl.col("home_goals").is_not_null()

    return (
        joined
        .with_columns(
            pl.when(played)
            .then(
                pl.when(pl.col("home_goals") > pl.col("away_goals"))
                .then(pl.lit("home"))
                .when(pl.col("home_goals") < pl.col("away_goals"))
                .then(pl.lit("away"))
                .otherwise(pl.lit("draw"))
            )
            .alias("actual_outcome")
        )
        .with_columns(
            pl.when(played)
            .then(pl.col("model_pick") == pl.col("actual_outcome"))
            .alias("model_correct")
        )
    )


#: Per-outcome accuracy frame schema (output of :func:`accuracy_by_outcome`).
OUTCOME_ACCURACY_SCHEMA: dict[str, pl.DataType] = {
    "actual_outcome": pl.Utf8,
    "n": pl.Int64,
    "hits": pl.Int64,
    "accuracy": pl.Float64,
}

#: Calibration-bin frame schema (output of :func:`calibration_bins`).
CALIBRATION_SCHEMA: dict[str, pl.DataType] = {
    "bin_lower": pl.Float64,
    "bin_upper": pl.Float64,
    "mean_confidence": pl.Float64,
    "hit_rate": pl.Float64,
    "count": pl.Int64,
}


def prediction_scorecard(finished: pl.DataFrame) -> dict[str, float]:
    """Aggregate accuracy / Brier / log-loss over finished fixtures (spec 6.1, page 7).

    Scores the model's pre-match H/D/A probabilities against recorded outcomes. Expects
    the ``finished`` frame returned by :func:`split_fixtures` (carrying ``model_home``/
    ``model_draw``/``model_away``, ``actual_outcome`` and ``model_correct``). All metrics
    are lower-is-better except ``accuracy``:

    - ``accuracy``: share of matches where the model's top pick matched the outcome.
    - ``brier_score``: mean multiclass Brier score (sum of squared errors over H/D/A),
      ranging 0 (perfect) to 2 (worst).
    - ``log_loss``: mean negative log-probability assigned to the realized outcome.

    Returns a zeroed scorecard (``n == 0``) for an empty frame so the page can render a
    "no data yet" state.
    """

    if finished.is_empty():
        return {"n": 0.0, "accuracy": 0.0, "brier_score": 0.0, "log_loss": 0.0}

    n = finished.height
    eps = 1e-12
    brier_total = 0.0
    log_total = 0.0
    for r in finished.iter_rows(named=True):
        probs = {
            "home": float(r["model_home"]),
            "draw": float(r["model_draw"]),
            "away": float(r["model_away"]),
        }
        actual = str(r["actual_outcome"])
        for outcome, p in probs.items():
            target = 1.0 if outcome == actual else 0.0
            brier_total += (p - target) ** 2
        log_total += -math.log(max(probs[actual], eps))

    return {
        "n": float(n),
        "accuracy": float(finished["model_correct"].sum()) / n,
        "brier_score": brier_total / n,
        "log_loss": log_total / n,
    }


def accuracy_by_outcome(finished: pl.DataFrame) -> pl.DataFrame:
    """Top-pick accuracy grouped by realized outcome (spec 6.1, page 7).

    Splits the ``finished`` frame (see :func:`split_fixtures`) by ``actual_outcome`` and
    reports the count, hits and accuracy of the model's top pick within each. Returns a
    typed empty frame when there are no finished fixtures.
    """

    if finished.is_empty():
        return pl.DataFrame(schema=OUTCOME_ACCURACY_SCHEMA)

    return (
        finished.group_by("actual_outcome")
        .agg(
            pl.len().alias("n"),
            pl.col("model_correct").cast(pl.Int64).sum().alias("hits"),
        )
        .with_columns((pl.col("hits") / pl.col("n")).alias("accuracy"))
        .sort("actual_outcome")
    )


def calibration_bins(finished: pl.DataFrame, *, n_bins: int = 5) -> pl.DataFrame:
    """Reliability bins of forecast confidence vs. observed hit rate (spec 6.1, page 7).

    Bins the finished fixtures by the probability the model assigned to its favoured
    outcome (its "confidence", the max of H/D/A) into ``n_bins`` equal-width buckets over
    ``[0, 1]``, and reports the mean confidence, observed hit rate and count per non-empty
    bucket. A well-calibrated model tracks the diagonal (mean confidence ≈ hit rate).
    Returns a typed empty frame when there are no finished fixtures.
    """

    if finished.is_empty():
        return pl.DataFrame(schema=CALIBRATION_SCHEMA)

    df = finished.with_columns(
        pl.max_horizontal("model_home", "model_draw", "model_away").alias("confidence")
    )
    df = df.with_columns(
        pl.min_horizontal(
            (pl.col("confidence") * n_bins).floor().cast(pl.Int64),
            pl.lit(n_bins - 1),
        ).alias("_bin")
    )
    return (
        df.group_by("_bin")
        .agg(
            pl.col("confidence").mean().alias("mean_confidence"),
            pl.col("model_correct").cast(pl.Float64).mean().alias("hit_rate"),
            pl.len().alias("count"),
        )
        .sort("_bin")
        .with_columns(
            (pl.col("_bin") / n_bins).alias("bin_lower"),
            ((pl.col("_bin") + 1) / n_bins).alias("bin_upper"),
        )
        .select(list(CALIBRATION_SCHEMA.keys()))
    )


def xg_error_summary(finished: pl.DataFrame) -> dict[str, float]:
    """MAE of predicted xG vs actual goals scored over finished matches.

    Compares ``exp_home_goals``/``exp_away_goals`` (model Poisson mean) against
    ``home_goals``/``away_goals`` (actual result). Returns zeroed dict when xG
    columns are absent or the frame is empty.
    """

    needed = {"exp_home_goals", "exp_away_goals", "home_goals", "away_goals"}
    if finished.is_empty() or not needed.issubset(finished.columns):
        return {"n": 0.0, "home_mae": 0.0, "away_mae": 0.0, "total_mae": 0.0}

    home_mae = float(
        (finished["exp_home_goals"] - finished["home_goals"].cast(pl.Float64)).abs().mean()
    )
    away_mae = float(
        (finished["exp_away_goals"] - finished["away_goals"].cast(pl.Float64)).abs().mean()
    )
    return {
        "n": float(finished.height),
        "home_mae": home_mae,
        "away_mae": away_mae,
        "total_mae": (home_mae + away_mae) / 2,
    }


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
