"""Dixon-Coles bivariate Poisson baseline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import cast

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from polymbappe.models.base import MatchModel


@dataclass(frozen=True, slots=True)
class MatchObservation:
    """Input observation for Dixon-Coles training."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    days_ago: float
    competition: str
    neutral_site: bool = False


@dataclass(slots=True)
class DixonColesConfig:
    """Dixon-Coles hyperparameters."""

    xi: float = 0.0019
    friendly_weight: float = 0.3
    max_goals: int = 10


_COMPETITIVE_KEYWORDS = {
    "world cup",
    "euro",
    "qualifier",
    "nations league",
    "copa america",
}


def _is_competitive(competition: str) -> bool:
    lower = competition.lower()
    return any(keyword in lower for keyword in _COMPETITIVE_KEYWORDS)


def tau_correction(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score Dixon-Coles correction factor τ(x, y, λ, μ, ρ)."""

    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam * rho)
    if x == 1 and y == 0:
        return 1.0 + (mu * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel(MatchModel):
    """Maximum-likelihood Dixon-Coles model with time decay."""

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config = config or DixonColesConfig()
        self.team_to_index: dict[str, int] = {}
        self.index_to_team: list[str] = []
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_advantage: float = 0.0
        self.rho: float = 0.0

    def fit(self, *args: object, **kwargs: object) -> DixonColesModel:
        """Fit model parameters by minimizing weighted negative log-likelihood."""

        matches_obj = kwargs.get("matches", args[0] if args else None)
        if not isinstance(matches_obj, list) or (
            matches_obj and not isinstance(matches_obj[0], MatchObservation)
        ):
            raise TypeError("fit expects a list[MatchObservation].")
        matches = cast(list[MatchObservation], matches_obj)

        if not matches:
            raise ValueError("At least one match is required to fit the model.")

        teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
        self.team_to_index = {team: idx for idx, team in enumerate(teams)}
        self.index_to_team = teams
        n_teams = len(teams)

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            attack_free = params[: n_teams - 1]
            defense_free = params[n_teams - 1 : 2 * (n_teams - 1)]
            home_advantage = params[-2]
            rho = params[-1]
            attack = np.concatenate([attack_free, np.array([-attack_free.sum()])])
            defense = np.concatenate([defense_free, np.array([-defense_free.sum()])])
            return attack, defense, home_advantage, rho

        def objective(params: np.ndarray) -> float:
            attack, defense, home_advantage, rho = unpack(params)
            neg_log_likelihood = 0.0
            for match in matches:
                home_idx = self.team_to_index[match.home_team]
                away_idx = self.team_to_index[match.away_team]

                home_term = 0.0 if match.neutral_site else home_advantage
                lam = exp(home_term + attack[home_idx] + defense[away_idx])
                mu = exp(attack[away_idx] + defense[home_idx])

                tau = max(tau_correction(match.home_goals, match.away_goals, lam, mu, rho), 1e-12)
                base = poisson.pmf(match.home_goals, lam) * poisson.pmf(match.away_goals, mu)
                likelihood = max(base * tau, 1e-300)

                decay = exp(-self.config.xi * match.days_ago)
                match_weight = (
                    1.0 if _is_competitive(match.competition) else self.config.friendly_weight
                )
                neg_log_likelihood -= decay * match_weight * log(likelihood)

            return neg_log_likelihood

        initial = np.zeros((2 * (n_teams - 1)) + 2, dtype=float)
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * len(initial)
        bounds[-1] = (-0.25, 0.25)
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)

        if not result.success:
            raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")

        attack, defense, home_advantage, rho = unpack(result.x)
        self.attack = attack
        self.defense = defense
        self.home_advantage = float(home_advantage)
        self.rho = float(rho)
        return self

    def _expectancies(
        self, home_team: str, away_team: str, neutral_site: bool = False
    ) -> tuple[float, float]:
        if self.attack is None or self.defense is None:
            raise RuntimeError("Model must be fit before predicting.")
        home_idx = self.team_to_index[home_team]
        away_idx = self.team_to_index[away_team]
        home_term = 0.0 if neutral_site else self.home_advantage
        lam = exp(home_term + self.attack[home_idx] + self.defense[away_idx])
        mu = exp(self.attack[away_idx] + self.defense[home_idx])
        return lam, mu

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
        neutral_site: bool = False,
    ) -> np.ndarray:
        """Return P(X=x,Y=y) grid up to max_goals."""

        cap = max_goals or self.config.max_goals
        lam, mu = self._expectancies(home_team, away_team, neutral_site=neutral_site)
        home_probs = poisson.pmf(np.arange(cap + 1), lam)
        away_probs = poisson.pmf(np.arange(cap + 1), mu)
        matrix = np.outer(home_probs, away_probs)

        for x in range(min(2, cap + 1)):
            for y in range(min(2, cap + 1)):
                matrix[x, y] *= tau_correction(x, y, lam, mu, self.rho)

        matrix = np.clip(matrix, 0.0, None)
        return np.asarray(matrix / matrix.sum(), dtype=float)

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict home/draw/away probabilities."""

        matrix = self.predict_score_matrix(home_team, away_team)
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        return {"home_win": home_win, "draw": draw, "away_win": away_win}
