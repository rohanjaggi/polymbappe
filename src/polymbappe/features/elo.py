"""Elo and Glicko-style feature utilities."""

from __future__ import annotations

from dataclasses import dataclass


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

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        """Update Elo ratings from one match result."""

        expected_home = self.expected_home_win_prob(home_team, away_team)
        if home_goals > away_goals:
            observed_home = 1.0
        elif home_goals < away_goals:
            observed_home = 0.0
        else:
            observed_home = 0.5

        delta = self.config.k_factor * (observed_home - expected_home)
        self._ratings[home_team] = self.rating(home_team) + delta
        self._ratings[away_team] = self.rating(away_team) - delta
