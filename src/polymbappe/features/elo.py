"""Elo and Glicko-style feature utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(slots=True)
class EloConfig:
    """Elo tuning parameters."""

    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0


class EloRatings:
    """Simple Elo implementation for match updates."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self._ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        """Get current team rating."""

        return self._ratings.get(team, self.config.base_rating)

    def expected_home_win_prob(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> float:
        """Expected score for the home side."""

        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)
        if not neutral:
            home_rating += self.config.home_advantage
        return 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        neutral: bool = False,
    ) -> None:
        """Update Elo ratings from one match result."""

        expected_home = self.expected_home_win_prob(home_team, away_team, neutral=neutral)
        if home_goals > away_goals:
            observed_home = 1.0
        elif home_goals < away_goals:
            observed_home = 0.0
        else:
            observed_home = 0.5

        delta = self.config.k_factor * (observed_home - expected_home)
        self._ratings[home_team] = self.rating(home_team) + delta
        self._ratings[away_team] = self.rating(away_team) - delta


def build_elo_features(
    matches: pl.DataFrame,
    as_of_date: date | None = None,
    config: EloConfig | None = None,
) -> pl.DataFrame:
    """Point-in-time pre-match Elo for each team appearance.

    Walks matches in chronological order, recording each team's rating *before* the
    match is played, then updating from the result. The recorded ``elo_pre`` therefore
    uses only strictly-earlier matches (no leakage). The ``neutral_site`` flag is
    threaded into each update, so no home advantage is credited at neutral venues.

    Args:
        matches: Frame with the ``matches`` schema.
        as_of_date: When set, only matches strictly before this date are used.
        config: Elo tuning parameters.

    Returns:
        Frame keyed by ``(match_id, team)`` with columns ``[match_id, team, date, elo_pre]``.
    """

    df = matches
    if as_of_date is not None:
        df = df.filter(pl.col("date") < as_of_date)
    df = df.sort(["date", "match_id"])

    elo = EloRatings(config)
    records: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        match_id, match_date = row["match_id"], row["date"]
        records.append(
            {"match_id": match_id, "team": home, "date": match_date, "elo_pre": elo.rating(home)}
        )
        records.append(
            {"match_id": match_id, "team": away, "date": match_date, "elo_pre": elo.rating(away)}
        )
        elo.update(
            home, away, int(row["home_goals"]), int(row["away_goals"]),
            neutral=bool(row.get("neutral_site", False)),
        )

    return pl.DataFrame(
        records,
        schema={"match_id": pl.Utf8, "team": pl.Utf8, "date": pl.Date, "elo_pre": pl.Float64},
    )


def build_elo_snapshots(
    matches: pl.DataFrame,
    config: EloConfig | None = None,
) -> pl.DataFrame:
    """Self-computed post-match Elo time series for every team appearance.

    Walks matches chronologically, updating ratings from each result and recording each
    team's rating *after* the match. This materializes the ``elo_snapshots`` table
    (``team, date, rating``) consumed by the dashboard's Elo-trajectory view and as a
    queryable artifact — distinct from :func:`build_elo_features`, which records the
    pre-match rating for leakage-safe modelling.
    """

    df = matches.sort(["date", "match_id"])
    elo = EloRatings(config)
    records: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        elo.update(
            home, away, int(row["home_goals"]), int(row["away_goals"]),
            neutral=bool(row.get("neutral_site", False)),
        )
        records.append({"team": home, "date": row["date"], "rating": elo.rating(home)})
        records.append({"team": away, "date": row["date"], "rating": elo.rating(away)})

    return pl.DataFrame(
        records,
        schema={"team": pl.Utf8, "date": pl.Date, "rating": pl.Float64},
    )
