"""Match-context and history-derived features (Tier 2-3).

All builders are point-in-time and leakage-safe: a feature for a match uses only
matches strictly earlier than it (rolling windows are shifted by one). Team-date
builders return a frame keyed by ``(match_id, team)``; match-level builders return a
frame keyed by ``match_id``. The :mod:`~polymbappe.features.pipeline` orchestrator joins
the home/away sides together.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import polars as pl

#: 2026 World Cup host nations. Name spellings must match the match-results source;
#: normalization across sources is handled upstream in the data layer.
HOSTS_2026: frozenset[str] = frozenset({"United States", "Mexico", "Canada"})


def team_match_long(matches: pl.DataFrame, as_of_date: date | None = None) -> pl.DataFrame:
    """Explode each match into two team-perspective rows.

    Returns columns ``[match_id, date, team, opponent, goals_for, goals_against, points,
    is_home]`` sorted by ``(team, date, match_id)`` so window functions over ``team`` see
    chronological order.
    """

    df = matches
    if as_of_date is not None:
        df = df.filter(pl.col("date") < as_of_date)

    home = df.select(
        pl.col("match_id"),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_goals").alias("goals_for"),
        pl.col("away_goals").alias("goals_against"),
        pl.lit(True).alias("is_home"),
    )
    away = df.select(
        pl.col("match_id"),
        pl.col("date"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_goals").alias("goals_for"),
        pl.col("home_goals").alias("goals_against"),
        pl.lit(False).alias("is_home"),
    )
    long = pl.concat([home, away], how="vertical")
    long = long.with_columns(
        pl.when(pl.col("goals_for") > pl.col("goals_against"))
        .then(3)
        .when(pl.col("goals_for") == pl.col("goals_against"))
        .then(1)
        .otherwise(0)
        .alias("points")
    )
    return long.sort(["team", "date", "match_id"])


def build_form_features(
    matches: pl.DataFrame,
    as_of_date: date | None = None,
    windows: tuple[int, ...] = (5, 10),
) -> pl.DataFrame:
    """Rolling form (avg goals for/against, avg points) over recent matches per team.

    Each window excludes the current match (shifted), so the first appearance is null.
    Returns ``(match_id, team, date, gs_<w>, ga_<w>, pts_<w> ...)``.
    """

    long = team_match_long(matches, as_of_date)
    exprs: list[pl.Expr] = []
    for w in windows:
        exprs.extend(
            [
                pl.col("goals_for")
                .shift(1)
                .rolling_mean(window_size=w, min_samples=1)
                .over("team")
                .alias(f"gs_{w}"),
                pl.col("goals_against")
                .shift(1)
                .rolling_mean(window_size=w, min_samples=1)
                .over("team")
                .alias(f"ga_{w}"),
                pl.col("points")
                .shift(1)
                .rolling_mean(window_size=w, min_samples=1)
                .over("team")
                .alias(f"pts_{w}"),
            ]
        )
    feature_cols = [f"{p}_{w}" for w in windows for p in ("gs", "ga", "pts")]
    return long.with_columns(exprs).select(["match_id", "team", "date", *feature_cols])


def build_rest_features(matches: pl.DataFrame, as_of_date: date | None = None) -> pl.DataFrame:
    """Days since each team's previous match. Null on a team's first appearance."""

    long = team_match_long(matches, as_of_date)
    return long.with_columns(
        (pl.col("date") - pl.col("date").shift(1).over("team"))
        .dt.total_days()
        .alias("rest_days")
    ).select(["match_id", "team", "date", "rest_days"])


def build_h2h_features(
    matches: pl.DataFrame,
    as_of_date: date | None = None,
    window: int = 5,
) -> pl.DataFrame:
    """Head-to-head win rate for the listed home team over recent meetings.

    For each match, looks at up to ``window`` prior meetings between the same pair (any
    venue) and computes the home team's points-share: ``(wins + 0.5 * draws) / n``. Null
    when the pair has no prior meetings. Returns ``(match_id, h2h_home_winrate,
    h2h_meetings)`` keyed by ``match_id``.
    """

    df = matches
    if as_of_date is not None:
        df = df.filter(pl.col("date") < as_of_date)
    df = df.sort(["date", "match_id"])

    # Per unordered pair: chronological list of (winner_team or None for draw).
    history: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    match_ids: list[str] = []
    winrates: list[float | None] = []
    meetings: list[int] = []

    for row in df.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        key = (home, away) if home <= away else (away, home)
        prior = history[key][-window:]

        if prior:
            score = 0.0
            for prior_winner in prior:
                if prior_winner is None:
                    score += 0.5
                elif prior_winner == home:
                    score += 1.0
            winrates.append(score / len(prior))
        else:
            winrates.append(None)
        meetings.append(len(prior))
        match_ids.append(row["match_id"])

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        winner: str | None = None if hg == ag else (home if hg > ag else away)
        history[key].append(winner)

    return pl.DataFrame(
        {"match_id": match_ids, "h2h_home_winrate": winrates, "h2h_meetings": meetings},
        schema={
            "match_id": pl.Utf8,
            "h2h_home_winrate": pl.Float64,
            "h2h_meetings": pl.Int64,
        },
    )


def build_structural_features(
    matches: pl.DataFrame,
    hosts: frozenset[str] = HOSTS_2026,
) -> pl.DataFrame:
    """Match-level structural flags: host advantage, knockout stage, neutral site.

    Returns ``(match_id, home_is_host, away_is_host, is_knockout, neutral_site)``.
    """

    host_list = list(hosts)
    return matches.select(
        pl.col("match_id"),
        pl.col("home_team").is_in(host_list).alias("home_is_host"),
        pl.col("away_team").is_in(host_list).alias("away_is_host"),
        pl.col("is_knockout"),
        pl.col("neutral_site"),
    )
