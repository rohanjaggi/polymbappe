"""Bayesian Dixon-Coles model scaffold."""

from polymbappe.models.base import MatchModel


class BayesianDixonColesModel(MatchModel):
    """PyMC hierarchical Dixon-Coles model (stub)."""

    def fit(self, *args: object, **kwargs: object) -> "BayesianDixonColesModel":
        raise NotImplementedError("Implement PyMC hierarchical model with random walk strengths.")

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        raise NotImplementedError
