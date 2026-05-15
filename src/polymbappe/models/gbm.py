"""Gradient boosting meta-model scaffold."""

from polymbappe.models.base import MatchModel


class GBMStackedModel(MatchModel):
    """LightGBM model stacked on top of baseline outputs (stub)."""

    def fit(self, *args: object, **kwargs: object) -> "GBMStackedModel":
        raise NotImplementedError

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        raise NotImplementedError
