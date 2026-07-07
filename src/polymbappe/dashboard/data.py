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


def load_match_xg(settings: Settings) -> pl.DataFrame:
    """Load per-match actual xG from the ``match_xg`` normalized table.

    Populated by ``polymbappe ingest --live`` (scrapes FBref) or a local
    ``data/raw/match_xg.csv``. Returns a typed empty frame when absent so the
    xG analysis section degrades gracefully before any live xG is ingested.
    """

    from polymbappe.data.tables import Table, table_path

    schema: dict[str, pl.DataType] = {
        "match_id": pl.Utf8,
        "date": pl.Date,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "home_xg": pl.Float64,
        "away_xg": pl.Float64,
    }
    return _read_or_empty(table_path(Table.MATCH_XG, settings), schema)


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

    # Join on exact (home, away) first
    joined = fixtures.join(results_slim, on=["home_team", "away_team"], how="left")

    # For unmatched rows, try the reversed pairing (neutral-venue fixtures may have
    # home/away swapped between the prediction schedule and the results feed).
    unmatched = joined.filter(pl.col("home_goals").is_null()).select(fixtures.columns)
    if not unmatched.is_empty() and not results_slim.is_empty():
        reversed_results = results_slim.rename(
            {"home_team": "away_team", "away_team": "home_team",
             "home_goals": "away_goals", "away_goals": "home_goals"}
        )
        reverse_joined = unmatched.join(
            reversed_results, on=["home_team", "away_team"], how="left"
        )
        matched_fwd = joined.filter(pl.col("home_goals").is_not_null())
        joined = pl.concat([matched_fwd, reverse_joined], how="diagonal_relaxed")

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

    # Try reversed pairing for unmatched rows (neutral-venue home/away swaps).
    unmatched = joined.filter(pl.col("home_goals").is_null()).select(fixtures.columns)
    if not unmatched.is_empty() and not results_slim.is_empty():
        reversed_results = results_slim.rename(
            {"home_team": "away_team", "away_team": "home_team",
             "home_goals": "away_goals", "away_goals": "home_goals"}
        )
        reverse_joined = unmatched.join(
            reversed_results, on=["home_team", "away_team"], how="left"
        )
        matched_fwd = joined.filter(pl.col("home_goals").is_not_null())
        joined = pl.concat([matched_fwd, reverse_joined], how="diagonal_relaxed")

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
        return {"n": 0.0, "accuracy": 0.0, "brier_score": 0.0, "log_loss": 0.0, "rps": 0.0}

    n = finished.height
    eps = 1e-12
    brier_total = 0.0
    log_total = 0.0
    rps_total = 0.0
    outcome_idx = {"home": 0, "draw": 1, "away": 2}
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
        prob_list = [probs["home"], probs["draw"], probs["away"]]
        actual_vec = [0.0, 0.0, 0.0]
        actual_vec[outcome_idx[actual]] = 1.0
        rps_total += sum(
            (sum(prob_list[: k + 1]) - sum(actual_vec[: k + 1])) ** 2
            for k in range(3)
        ) / 2

    return {
        "n": float(n),
        "accuracy": float(finished["model_correct"].sum()) / n,
        "brier_score": brier_total / n,
        "log_loss": log_total / n,
        "rps": rps_total / n,
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


def xg_error_summary(
    finished: pl.DataFrame,
    match_xg: pl.DataFrame | None = None,
) -> dict[str, float]:
    """MAE of model predicted xG vs actual goals and (optionally) vs actual match xG.

    Always computes model-vs-goals MAE. When ``match_xg`` is provided and contains rows
    matching the finished fixtures, additionally computes:

    - ``model_vs_xg_home/away_mae``: model predicted xG vs actual FBref xG (pure model
      quality, removes finishing-luck noise).
    - ``xg_vs_goals_home/away_mae``: actual FBref xG vs actual goals (finishing luck).

    Returns zeroed dict when required columns are absent.
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
    result: dict[str, float] = {
        "n": float(finished.height),
        "home_mae": home_mae,
        "away_mae": away_mae,
        "total_mae": (home_mae + away_mae) / 2,
    }

    if match_xg is None or match_xg.is_empty():
        return result

    # Join actual xG onto finished fixtures by (home_team, away_team), with a
    # reversed-pairing fallback for neutral-venue matches where FBref and the
    # prediction schedule list home/away in opposite order.
    xg_slim = match_xg.select(["home_team", "away_team", "home_xg", "away_xg"])
    joined = finished.join(xg_slim, on=["home_team", "away_team"], how="inner")
    unmatched = finished.join(xg_slim, on=["home_team", "away_team"], how="anti")
    if not unmatched.is_empty():
        xg_rev = xg_slim.rename(
            {"home_team": "away_team", "away_team": "home_team",
             "home_xg": "away_xg", "away_xg": "home_xg"}
        )
        rev_joined = unmatched.join(xg_rev, on=["home_team", "away_team"], how="inner")
        if not rev_joined.is_empty():
            joined = pl.concat([joined, rev_joined], how="diagonal_relaxed")
    if joined.is_empty():
        return result

    result["xg_n"] = float(joined.height)
    result["model_vs_xg_home_mae"] = float(
        (joined["exp_home_goals"] - joined["home_xg"]).abs().mean()
    )
    result["model_vs_xg_away_mae"] = float(
        (joined["exp_away_goals"] - joined["away_xg"]).abs().mean()
    )
    result["model_vs_xg_mae"] = (
        result["model_vs_xg_home_mae"] + result["model_vs_xg_away_mae"]
    ) / 2
    result["xg_vs_goals_home_mae"] = float(
        (joined["home_xg"] - joined["home_goals"].cast(pl.Float64)).abs().mean()
    )
    result["xg_vs_goals_away_mae"] = float(
        (joined["away_xg"] - joined["away_goals"].cast(pl.Float64)).abs().mean()
    )
    result["xg_vs_goals_mae"] = (
        result["xg_vs_goals_home_mae"] + result["xg_vs_goals_away_mae"]
    ) / 2
    return result


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


# -- new loaders ---------------------------------------------------------------

SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "date": pl.Date,
    "stage": pl.Utf8,
    "group": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "city": pl.Utf8,
}

AUTOTUNE_SCHEMA: dict[str, pl.DataType] = {
    "experiment_id": pl.Utf8,
    "phase": pl.Utf8,
    "decision": pl.Utf8,
    "mean_rps": pl.Float64,
    "config": pl.Utf8,
    "per_tournament": pl.Utf8,
    "hypothesis": pl.Utf8,
}


def load_schedule(settings: Settings) -> pl.DataFrame:
    """Load the tournament match schedule (``schedule.parquet``)."""

    from polymbappe.data.tables import Table, table_path

    return _read_or_empty(table_path(Table.SCHEDULE, settings), SCHEDULE_SCHEMA)


def load_autotune_leaderboard(settings: Settings) -> pl.DataFrame:
    """Load autotuner experiment results (``autotune_leaderboard.parquet``)."""

    return _read_or_empty(
        _output_path(settings, "autotune_leaderboard.parquet"), AUTOTUNE_SCHEMA
    )


# -- new helpers ---------------------------------------------------------------


def compute_group_standings(
    match_df: pl.DataFrame, results_df: pl.DataFrame
) -> pl.DataFrame:
    """Compute actual group standings from played matches.

    Group membership comes from ``match_df`` (which carries the ``group`` column).
    Returns a frame with: ``group, team, played, won, drawn, lost, gf, ga, gd, points``,
    sorted by group then points (desc), gd (desc), gf (desc).
    """

    if match_df.is_empty() or results_df.is_empty():
        return pl.DataFrame(
            schema={
                "group": pl.Utf8, "team": pl.Utf8, "played": pl.Int64,
                "won": pl.Int64, "drawn": pl.Int64, "lost": pl.Int64,
                "gf": pl.Int64, "ga": pl.Int64, "gd": pl.Int64, "points": pl.Int64,
            }
        )

    gs = match_df.filter(pl.col("group") != "KO")
    group_map: dict[str, str] = {}
    for r in gs.iter_rows(named=True):
        group_map.setdefault(str(r["home_team"]), str(r["group"]))
        group_map.setdefault(str(r["away_team"]), str(r["group"]))

    _, finished = split_fixtures(gs, results_df)
    if finished.is_empty():
        return pl.DataFrame(
            schema={
                "group": pl.Utf8, "team": pl.Utf8, "played": pl.Int64,
                "won": pl.Int64, "drawn": pl.Int64, "lost": pl.Int64,
                "gf": pl.Int64, "ga": pl.Int64, "gd": pl.Int64, "points": pl.Int64,
            }
        )

    rows: list[dict[str, object]] = []
    team_stats: dict[str, dict[str, int]] = {}
    for r in finished.iter_rows(named=True):
        h, a = str(r["home_team"]), str(r["away_team"])
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            s = team_stats.setdefault(team, {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0})
            s["played"] += 1
            s["gf"] += gf
            s["ga"] += ga
            if gf > ga:
                s["won"] += 1
            elif gf == ga:
                s["drawn"] += 1
            else:
                s["lost"] += 1

    for team, s in team_stats.items():
        gd = s["gf"] - s["ga"]
        pts = 3 * s["won"] + s["drawn"]
        rows.append({
            "group": group_map.get(team, "?"),
            "team": team,
            "played": s["played"],
            "won": s["won"],
            "drawn": s["drawn"],
            "lost": s["lost"],
            "gf": s["gf"],
            "ga": s["ga"],
            "gd": gd,
            "points": pts,
        })

    return (
        pl.DataFrame(rows)
        .sort(["group", "points", "gd", "gf"], descending=[False, True, True, True])
    )


def predicted_group_points(match_df: pl.DataFrame) -> pl.DataFrame:
    """Compute expected group points per team from model H/D/A probabilities.

    For each fixture: home expected pts = 3*P(home) + P(draw),
    away expected pts = 3*P(away) + P(draw). Summed per team.
    Returns frame: ``group, team, predicted_points``.
    """

    if match_df.is_empty():
        return pl.DataFrame(schema={"group": pl.Utf8, "team": pl.Utf8, "predicted_points": pl.Float64})

    gs = match_df.filter(pl.col("group") != "KO")
    team_pts: dict[str, float] = {}
    team_group: dict[str, str] = {}
    for r in gs.iter_rows(named=True):
        h, a = str(r["home_team"]), str(r["away_team"])
        ph, pd, pa = float(r["model_home"]), float(r["model_draw"]), float(r["model_away"])
        team_pts[h] = team_pts.get(h, 0.0) + 3.0 * ph + pd
        team_pts[a] = team_pts.get(a, 0.0) + 3.0 * pa + pd
        team_group.setdefault(h, str(r["group"]))
        team_group.setdefault(a, str(r["group"]))

    rows = [
        {"group": team_group.get(t, "?"), "team": t, "predicted_points": round(p, 1)}
        for t, p in team_pts.items()
    ]
    return pl.DataFrame(rows).sort(["group", "predicted_points"], descending=[False, True])


def biggest_surprises(finished: pl.DataFrame, *, n: int = 5) -> pl.DataFrame:
    """Matches where the model was most wrong — sorted by P(actual_outcome) ascending."""

    if finished.is_empty() or "actual_outcome" not in finished.columns:
        return finished

    rows = []
    for r in finished.iter_rows(named=True):
        probs = {
            "home": float(r["model_home"]),
            "draw": float(r["model_draw"]),
            "away": float(r["model_away"]),
        }
        actual = str(r["actual_outcome"])
        pick = max(probs, key=probs.get)  # type: ignore[arg-type]
        rows.append({
            "Fixture": f"{r['home_team']} vs {r['away_team']}",
            "Model Pick": {"home": str(r["home_team"]), "draw": "Draw", "away": str(r["away_team"])}.get(pick, pick),
            "Pick Confidence": f"{max(probs.values()):.0%}",
            "Actual Result": {"home": str(r["home_team"]), "draw": "Draw", "away": str(r["away_team"])}.get(actual, actual),
            "P(Actual)": f"{probs[actual]:.0%}",
            "p_actual_raw": probs[actual],
            "Score": (
                f"{int(r['home_goals'])} – {int(r['away_goals'])}"
                if r.get("home_goals") is not None else "—"
            ),
        })

    return (
        pl.DataFrame(rows)
        .sort("p_actual_raw")
        .head(n)
        .drop("p_actual_raw")
    )


def classify_ko_fixtures(
    match_df: pl.DataFrame, results_df: pl.DataFrame
) -> pl.DataFrame:
    """Classify KO entries from match_predictions as R32 or R16 based on result dates.

    Returns the KO subset of match_df joined with results, adding columns:
    ``stage, home_goals, away_goals, actual_outcome, model_correct, date``.
    """

    ko = match_df.filter(pl.col("group") == "KO")
    if ko.is_empty():
        return ko

    ko = ko.with_columns(_model_pick_expr().alias("model_pick"))

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

    joined = ko.join(results_slim, on=["home_team", "away_team"], how="left")
    unmatched = joined.filter(pl.col("home_goals").is_null()).select(ko.columns)
    if not unmatched.is_empty() and not results_slim.is_empty():
        reversed_results = results_slim.rename(
            {"home_team": "away_team", "away_team": "home_team",
             "home_goals": "away_goals", "away_goals": "home_goals"}
        )
        reverse_joined = unmatched.join(
            reversed_results, on=["home_team", "away_team"], how="left"
        )
        matched_fwd = joined.filter(pl.col("home_goals").is_not_null())
        joined = pl.concat([matched_fwd, reverse_joined], how="diagonal_relaxed")

    played = pl.col("home_goals").is_not_null()
    import datetime as _dt
    r32_cutoff = _dt.date(2026, 7, 4)

    joined = joined.with_columns(
        pl.when(~played)
        .then(pl.lit("upcoming"))
        .when(pl.col("date") < r32_cutoff)
        .then(pl.lit("R32"))
        .otherwise(pl.lit("R16"))
        .alias("stage")
    )

    joined = joined.with_columns(
        pl.when(played)
        .then(
            pl.when(pl.col("home_goals") > pl.col("away_goals"))
            .then(pl.lit("home"))
            .when(pl.col("home_goals") < pl.col("away_goals"))
            .then(pl.lit("away"))
            .otherwise(pl.lit("draw"))
        )
        .alias("actual_outcome")
    ).with_columns(
        pl.when(played)
        .then(pl.col("model_pick") == pl.col("actual_outcome"))
        .alias("model_correct")
    )

    return joined


def _build_position_map(
    group_probs: pl.DataFrame, match_df: pl.DataFrame
) -> dict[tuple[str, str], str]:
    """Map ``(position, group)`` to team name using deterministic group probabilities.

    E.g. ``("1", "A") -> "Mexico"``, ``("3", "D") -> "Paraguay"``.
    """

    gs = match_df.filter(pl.col("group") != "KO")
    team_group: dict[str, str] = {}
    for r in gs.iter_rows(named=True):
        team_group.setdefault(str(r["home_team"]), str(r["group"]))
        team_group.setdefault(str(r["away_team"]), str(r["group"]))

    pos_map: dict[tuple[str, str], str] = {}
    for r in group_probs.iter_rows(named=True):
        team = str(r["team"])
        group = team_group.get(team)
        if not group:
            continue
        for pos, col in [(1, "finish_1"), (2, "finish_2"), (3, "finish_3"), (4, "finish_4")]:
            if float(r[col]) == 1.0:
                pos_map[(str(pos), group)] = team
                break
    return pos_map


def resolve_bracket(
    schedule_df: pl.DataFrame,
    ko_fixtures: pl.DataFrame,
    group_probs: pl.DataFrame | None = None,
    match_df: pl.DataFrame | None = None,
    stage_probs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Resolve placeholder codes in the KO schedule to actual team names.

    Uses group_probabilities to map simple position codes (1A, 2B) and matches
    KO fixture results against schedule slots to resolve third-place wildcards.
    """

    if schedule_df.is_empty():
        return schedule_df

    ko_sched = schedule_df.filter(pl.col("group").is_null()).sort("date")

    pos_map: dict[tuple[str, str], str] = {}
    if group_probs is not None and match_df is not None:
        pos_map = _build_position_map(group_probs, match_df)

    r16_teams: set[str] = set()
    if not ko_fixtures.is_empty() and "stage" in ko_fixtures.columns:
        r16_rows = ko_fixtures.filter(pl.col("stage") == "R16")
        for r in r16_rows.iter_rows(named=True):
            r16_teams.add(str(r["home_team"]))
            r16_teams.add(str(r["away_team"]))

    eliminated_teams: set[str] = set()
    if stage_probs is not None and not stage_probs.is_empty() and "R16" in stage_probs.columns:
        for r in stage_probs.iter_rows(named=True):
            if float(r["R16"]) == 0.0:
                eliminated_teams.add(str(r["team"]))

    # Build list of R32 fixtures from ko_fixtures for matching
    r32_fixture_pairs: list[tuple[str, str]] = []
    if not ko_fixtures.is_empty() and "stage" in ko_fixtures.columns:
        for r in ko_fixtures.filter(pl.col("stage") == "R32").iter_rows(named=True):
            r32_fixture_pairs.append((str(r["home_team"]), str(r["away_team"])))

    def _resolve_simple(code: str) -> str | None:
        if not code or len(code) < 2 or "/" in code:
            return None
        pos, group = code[0], code[1:]
        return pos_map.get((pos, group))

    def _wildcard_candidates(code: str) -> set[str]:
        if "/" not in code:
            return set()
        pos = code[0]
        groups = code[1:].split("/")
        return {pos_map[(pos, g)] for g in groups if (pos, g) in pos_map}

    # Enumerate KO schedule matches
    match_numbers: dict[int, dict[str, object]] = {}
    for i, r in enumerate(ko_sched.iter_rows(named=True)):
        mn = 73 + i
        match_numbers[mn] = {
            "stage": r["stage"], "date": r["date"], "city": r["city"],
            "home_code": str(r["home_team"]), "away_code": str(r["away_team"]),
        }

    bracket: dict[str, str | None] = {}
    used_fixtures: set[tuple[str, str]] = set()

    # Resolve R32: for each slot, resolve the fixed side then find the actual opponent
    for mn, info in sorted(match_numbers.items()):
        if info["stage"] != "Round of 32":
            continue
        code_h, code_a = str(info["home_code"]), str(info["away_code"])
        fixed_h = _resolve_simple(code_h)
        fixed_a = _resolve_simple(code_a)

        if fixed_h and fixed_a:
            # Both sides are simple position codes
            winner = _find_winner(ko_fixtures, fixed_h, fixed_a, r16_teams, eliminated_teams)
            bracket[f"W{mn}"] = winner
            if winner:
                bracket[f"L{mn}"] = fixed_a if winner == fixed_h else fixed_h
            used_fixtures.add((fixed_h, fixed_a))
        elif fixed_h and not fixed_a:
            # Home is known, away is wildcard — find which fixture has fixed_h
            candidates = _wildcard_candidates(code_a)
            for h, a in r32_fixture_pairs:
                pair = {h, a}
                if fixed_h in pair and (h, a) not in used_fixtures:
                    other = a if h == fixed_h else h
                    if other in candidates:
                        fixed_a = other
                        winner = _find_winner(ko_fixtures, h, a, r16_teams, eliminated_teams)
                        bracket[f"W{mn}"] = winner
                        if winner:
                            bracket[f"L{mn}"] = a if winner == h else h
                        used_fixtures.add((h, a))
                        break
        elif fixed_a and not fixed_h:
            candidates = _wildcard_candidates(code_h)
            for h, a in r32_fixture_pairs:
                pair = {h, a}
                if fixed_a in pair and (h, a) not in used_fixtures:
                    other = h if a == fixed_a else a
                    if other in candidates:
                        fixed_h = other
                        winner = _find_winner(ko_fixtures, h, a, r16_teams, eliminated_teams)
                        bracket[f"W{mn}"] = winner
                        if winner:
                            bracket[f"L{mn}"] = a if winner == h else h
                        used_fixtures.add((h, a))
                        break

        # Store resolved teams for this match
        match_numbers[mn]["home_resolved"] = fixed_h
        match_numbers[mn]["away_resolved"] = fixed_a

    # Resolve R16+ using bracket cascade
    def _resolve_code(code: str) -> str | None:
        if code.startswith("W") or code.startswith("L"):
            return bracket.get(code)
        return _resolve_simple(code) or code

    for mn, info in sorted(match_numbers.items()):
        if info["stage"] not in ("Round of 16",):
            continue
        h = _resolve_code(str(info["home_code"]))
        a = _resolve_code(str(info["away_code"]))
        match_numbers[mn]["home_resolved"] = h
        match_numbers[mn]["away_resolved"] = a
        if h and a:
            winner = _find_winner(ko_fixtures, h, a, set(), eliminated_teams)
            bracket[f"W{mn}"] = winner
            if winner:
                bracket[f"L{mn}"] = a if winner == h else h

    # Resolve QF/SF/Final
    for mn, info in sorted(match_numbers.items()):
        if info["stage"] in ("Round of 32", "Round of 16"):
            continue
        h = _resolve_code(str(info["home_code"]))
        a = _resolve_code(str(info["away_code"]))
        match_numbers[mn]["home_resolved"] = h
        match_numbers[mn]["away_resolved"] = a

    # Build output
    rows = []
    for mn in sorted(match_numbers.keys()):
        info = match_numbers[mn]
        res_h = info.get("home_resolved")
        res_a = info.get("away_resolved")
        status = "tbd"
        if res_h and res_a:
            status = "played" if _has_result(ko_fixtures, str(res_h), str(res_a)) else "upcoming"
        rows.append({
            "match_number": mn,
            "stage": str(info["stage"]),
            "date": info["date"],
            "city": str(info["city"]),
            "home_code": str(info["home_code"]),
            "away_code": str(info["away_code"]),
            "home_resolved": str(res_h) if res_h else None,
            "away_resolved": str(res_a) if res_a else None,
            "status": status,
        })

    return pl.DataFrame(rows)


def _find_winner(
    ko_fixtures: pl.DataFrame,
    team_a: str,
    team_b: str,
    later_stage_teams: set[str],
    eliminated_teams: set[str] | None = None,
) -> str | None:
    """Find the winner of a match between team_a and team_b from ko_fixtures results."""

    if ko_fixtures.is_empty():
        return None

    for r in ko_fixtures.iter_rows(named=True):
        h, a = str(r["home_team"]), str(r["away_team"])
        if not ({h, a} == {team_a, team_b}):
            continue
        hg, ag = r.get("home_goals"), r.get("away_goals")
        if hg is None or ag is None:
            return None
        hg, ag = int(hg), int(ag)
        if hg > ag:
            return h
        if ag > hg:
            return a
        # Draw — check later stage appearances
        if h in later_stage_teams:
            return h
        if a in later_stage_teams:
            return a
        # Draw — check if one team was eliminated (R16 prob = 0 in stage probs)
        if eliminated_teams:
            if h in eliminated_teams and a not in eliminated_teams:
                return a
            if a in eliminated_teams and h not in eliminated_teams:
                return h
        return None
    return None


def _has_result(ko_fixtures: pl.DataFrame, team_a: str, team_b: str) -> bool:
    """Check if a match between these teams has a recorded result in ko_fixtures."""

    if ko_fixtures.is_empty():
        return False
    for r in ko_fixtures.iter_rows(named=True):
        h, a = str(r["home_team"]), str(r["away_team"])
        if {h, a} == {team_a, team_b} and r.get("home_goals") is not None:
            return True
    return False


def actual_upsets(finished: pl.DataFrame, *, threshold: float = 0.35) -> pl.DataFrame:
    """Matches where the underdog won — the actual outcome had probability below threshold."""

    if finished.is_empty() or "actual_outcome" not in finished.columns:
        return pl.DataFrame(schema={"Fixture": pl.Utf8})

    rows = []
    for r in finished.iter_rows(named=True):
        probs = {
            "home": float(r["model_home"]),
            "draw": float(r["model_draw"]),
            "away": float(r["model_away"]),
        }
        actual = str(r["actual_outcome"])
        p_actual = probs[actual]
        if p_actual >= threshold:
            continue
        pick = max(probs, key=probs.get)  # type: ignore[arg-type]
        if pick == actual:
            continue

        label = {"home": str(r["home_team"]), "draw": "Draw", "away": str(r["away_team"])}
        rows.append({
            "Fixture": f"{r['home_team']} vs {r['away_team']}",
            "Score": f"{int(r['home_goals'])} – {int(r['away_goals'])}",
            "Model Pick": label.get(pick, pick),
            "Pick Confidence": f"{max(probs.values()):.0%}",
            "Actual Result": label.get(actual, actual),
            "P(Actual)": f"{p_actual:.0%}",
            "Upset Magnitude": f"{1 - p_actual:.0%}",
            "_sort": p_actual,
        })

    if not rows:
        return pl.DataFrame(schema={"Fixture": pl.Utf8})

    return pl.DataFrame(rows).sort("_sort").drop("_sort")


def dark_horses(stage_df: pl.DataFrame, *, n: int = 10) -> pl.DataFrame:
    """Teams punching above their weight — high QF/SF odds relative to champion odds."""

    if stage_df.is_empty() or "QF" not in stage_df.columns:
        return stage_df

    df = stage_df.filter(pl.col("QF") > 0)
    if df.is_empty():
        return df

    champ_floor = 0.001
    df = df.with_columns(
        (pl.col("QF") / pl.max_horizontal(pl.col("champion"), pl.lit(champ_floor)))
        .alias("overperformance")
    )
    # Filter out actual favourites (champion > 5%)
    df = df.filter(pl.col("champion") <= 0.05)
    if df.is_empty():
        return df

    return (
        df.select(["team", "R16", "QF", "SF", "FINAL", "champion", "overperformance"])
        .sort("overperformance", descending=True)
        .head(n)
    )
