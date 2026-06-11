"""Sentiment & overperformance features (spec 2.2 Group E).

Three signals of increasing speculativeness:

* **xG overperformance** (permanent, Tier A, backtestable) — ``goals_scored - xG`` over the
  last 10 matches. The quantitative "winning ugly / winning well" signal. This carries the
  load; it is mechanistically clean and available 2018+.
* **Reddit match sentiment** (additive, Tier B, forward-test only) — VADER + LLM scoring of
  r/soccer post-match threads. Known noisy (sarcasm).
* **News headline sentiment** (additive, Tier B, forward-test only) — local-LLM classified
  BBC Sport headlines, bounded by the adjuster's ±3pp cap.

The Tier B scorers degrade gracefully: if their optional dependencies (``vaderSentiment``)
or live sources are unavailable, they return a neutral 0.0 so the permanent xG signal still
drives the feature. PRAW/LLM ingestion proper lives in the live agent (spec 5).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.features.xg import build_xg_features


def build_xg_overperformance(
    matches: pl.DataFrame,
    team_xg: pl.DataFrame | None = None,
    as_of_date: date | None = None,
    window: int = 10,
) -> pl.DataFrame:
    """Rolling ``goals_scored - xG`` per team appearance (the permanent signal).

    Uses the same rolling xG builder as the core features. When real xG is unavailable the
    proxy makes overperformance ~0 (goals minus goals-proxy), which is the correct
    degenerate behaviour — no false signal.

    Returns ``(match_id, team, date, xg_overperformance)``.
    """

    xg = build_xg_features(matches, team_xg, as_of_date, window)
    from polymbappe.features.context import team_match_long

    long = team_match_long(matches, as_of_date).select(
        ["match_id", "team", "goals_for"]
    )
    # Rolling actual goals over the same window, shifted (exclude current match).
    long = long.with_columns(
        pl.col("goals_for")
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over("team")
        .alias("goals_roll")
    )
    joined = xg.join(long.select(["match_id", "team", "goals_roll"]), on=["match_id", "team"])
    return joined.with_columns(
        (pl.col("goals_roll") - pl.col("xg_for")).alias("xg_overperformance")
    ).select(["match_id", "team", "date", "xg_overperformance"])


def score_text_vader(texts: list[str]) -> float:
    """Mean VADER compound sentiment over ``texts`` in ``[-1, 1]`` (0.0 if unavailable).

    Degrades to 0.0 when ``vaderSentiment`` is not installed or no texts are given, so the
    Tier B path never raises in environments without the optional dependency.
    """

    if not texts:
        return 0.0
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts if t]
    return float(sum(scores) / len(scores)) if scores else 0.0


def build_sentiment_snapshot(
    matches: pl.DataFrame,
    team_xg: pl.DataFrame | None = None,
    reddit_scores: dict[str, float] | None = None,
    news_scores: dict[str, float] | None = None,
    as_of_date: date | None = None,
) -> pl.DataFrame:
    """Combine the permanent xG overperformance with optional Tier B sentiment.

    ``reddit_scores`` / ``news_scores`` are optional per-team additive overlays (forward
    test only); absent teams get 0.0. Returns the latest-per-team snapshot keyed by
    ``team`` with ``[team, xg_overperformance, reddit_score, news_tone]``.
    """

    overperf = build_xg_overperformance(matches, team_xg, as_of_date)
    latest = (
        overperf.sort(["team", "date"])
        .group_by("team")
        .agg(pl.col("xg_overperformance").last())
    )
    reddit_scores = reddit_scores or {}
    news_scores = news_scores or {}
    return latest.with_columns(
        pl.col("team")
        .replace_strict(reddit_scores, default=0.0, return_dtype=pl.Float64)
        .alias("reddit_score"),
        pl.col("team")
        .replace_strict(news_scores, default=0.0, return_dtype=pl.Float64)
        .alias("news_tone"),
    )
