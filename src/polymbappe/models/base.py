"""Abstract match model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MatchModel(ABC):
    """Base interface for match outcome models."""

    @abstractmethod
    def fit(self, *args: object, **kwargs: object) -> MatchModel:
        """Fit model state from historical data."""

    @abstractmethod
    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict outcome probabilities for a match."""
