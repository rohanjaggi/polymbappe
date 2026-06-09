"""Dixon-Coles bivariate Poisson baseline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
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
    maxiter: int = 5000
    max_history_days: int = 4000


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

    @staticmethod
    def _build_initial_params(
        teams: list[str],
        n_teams: int,
        prev_attack: np.ndarray | None,
        prev_defense: np.ndarray | None,
        prev_team_to_index: dict[str, int],
        prev_home_advantage: float,
        prev_rho: float,
    ) -> np.ndarray:
        """Build initial parameter vector, warm-starting from previous fit if available."""
        if prev_attack is None or prev_defense is None:
            return np.zeros((2 * (n_teams - 1)) + 2, dtype=float)

        initial = np.zeros((2 * (n_teams - 1)) + 2, dtype=float)

        attack_free = np.zeros(n_teams - 1)
        defense_free = np.zeros(n_teams - 1)
        for i, team in enumerate(teams[:-1]):
            if team in prev_team_to_index:
                old_idx = prev_team_to_index[team]
                attack_free[i] = prev_attack[old_idx]
                defense_free[i] = prev_defense[old_idx]

        attack_free -= attack_free.mean()
        defense_free -= defense_free.mean()

        initial[: n_teams - 1] = attack_free
        initial[n_teams - 1 : 2 * (n_teams - 1)] = defense_free
        initial[-2] = prev_home_advantage
        initial[-1] = prev_rho
        return initial

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

        if self.config.max_history_days > 0:
            matches = [m for m in matches if m.days_ago <= self.config.max_history_days]

        teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
        n_teams = len(teams)
        prev_attack = self.attack
        prev_defense = self.defense
        prev_team_to_index = self.team_to_index
        self.team_to_index = {team: idx for idx, team in enumerate(teams)}
        self.index_to_team = teams
        n_matches = len(matches)

        # Pre-compute match arrays (done once, outside the optimizer loop).
        home_idx = np.array([self.team_to_index[m.home_team] for m in matches], dtype=np.intp)
        away_idx = np.array([self.team_to_index[m.away_team] for m in matches], dtype=np.intp)
        home_goals = np.array([m.home_goals for m in matches], dtype=np.int32)
        away_goals = np.array([m.away_goals for m in matches], dtype=np.int32)
        neutral = np.array([m.neutral_site for m in matches], dtype=bool)

        # Weights: time decay * competition weight
        days_ago = np.array([m.days_ago for m in matches], dtype=np.float64)
        comp_weight = np.array(
            [1.0 if _is_competitive(m.competition) else self.config.friendly_weight for m in matches],
            dtype=np.float64,
        )
        weights = np.exp(-self.config.xi * days_ago) * comp_weight

        # Tau correction masks (only applies when both goals <= 1).
        m00 = (home_goals == 0) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m10 = (home_goals == 1) & (away_goals == 0)
        m11 = (home_goals == 1) & (away_goals == 1)

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            attack_free = params[: n_teams - 1]
            defense_free = params[n_teams - 1 : 2 * (n_teams - 1)]
            home_advantage = params[-2]
            rho = params[-1]
            attack = np.concatenate([attack_free, np.array([-attack_free.sum()])])
            defense = np.concatenate([defense_free, np.array([-defense_free.sum()])])
            return attack, defense, home_advantage, rho

        def objective_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defense, home_advantage, rho = unpack(params)

            home_term = np.where(neutral, 0.0, home_advantage)
            lam = np.exp(home_term + attack[home_idx] + defense[away_idx])
            mu = np.exp(attack[away_idx] + defense[home_idx])

            log_lik = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)

            log_tau = np.zeros(n_matches, dtype=np.float64)
            tau_val = np.ones(n_matches, dtype=np.float64)
            tau_val[m00] = np.maximum(1.0 - lam[m00] * mu[m00] * rho, 1e-12)
            tau_val[m01] = np.maximum(1.0 + lam[m01] * rho, 1e-12)
            tau_val[m10] = np.maximum(1.0 + mu[m10] * rho, 1e-12)
            tau_val[m11] = np.maximum(1.0 - rho, 1e-12)
            log_tau[m00] = np.log(tau_val[m00])
            log_tau[m01] = np.log(tau_val[m01])
            log_tau[m10] = np.log(tau_val[m10])
            log_tau[m11] = np.log(tau_val[m11])

            nll = -float(np.sum(weights * (log_lik + log_tau)))

            # Gradient of Poisson NLL: d(-log P(k|λ))/dλ = 1 - k/λ, times dλ/dparam.
            # For home goals ~ Poisson(lam): d_nll/d_lam_i = (lam_i - home_goals_i)
            # For away goals ~ Poisson(mu):  d_nll/d_mu_i  = (mu_i  - away_goals_i)
            # (these are per-match residuals, pre-multiplied by weights below)
            dlam = weights * (lam - home_goals)
            dmu = weights * (mu - away_goals)

            # Tau correction gradients (d(-log tau)/d_param)
            dtau_dlam = np.zeros(n_matches)
            dtau_dmu = np.zeros(n_matches)
            dtau_drho = np.zeros(n_matches)
            # m00: tau = 1 - lam*mu*rho => d(-log tau)/dlam = mu*rho/tau
            dtau_dlam[m00] = weights[m00] * mu[m00] * rho / tau_val[m00]
            dtau_dmu[m00] = weights[m00] * lam[m00] * rho / tau_val[m00]
            dtau_drho[m00] = weights[m00] * lam[m00] * mu[m00] / tau_val[m00]
            # m01: tau = 1 + lam*rho => d(-log tau)/dlam = -rho/tau
            dtau_dlam[m01] = -weights[m01] * rho / tau_val[m01]
            dtau_drho[m01] = -weights[m01] * lam[m01] / tau_val[m01]
            # m10: tau = 1 + mu*rho => d(-log tau)/dmu = -rho/tau
            dtau_dmu[m10] = -weights[m10] * rho / tau_val[m10]
            dtau_drho[m10] = -weights[m10] * mu[m10] / tau_val[m10]
            # m11: tau = 1 - rho => d(-log tau)/drho = 1/tau
            dtau_drho[m11] = weights[m11] / tau_val[m11]

            # Total per-match derivatives w.r.t. lam and mu (chain rule: dlam/dparam = lam * dparam)
            d_lam_total = dlam + dtau_dlam * lam
            d_mu_total = dmu + dtau_dmu * mu

            # Accumulate into parameter gradient
            grad = np.zeros(len(params))

            # Attack parameters: lam depends on attack[home], mu depends on attack[away]
            grad_attack = np.zeros(n_teams)
            np.add.at(grad_attack, home_idx, d_lam_total)
            np.add.at(grad_attack, away_idx, d_mu_total)
            # Sum-to-zero constraint: free params are attack[0..n-2], attack[n-1] = -sum(free)
            grad[: n_teams - 1] = grad_attack[:-1] - grad_attack[-1]

            # Defense parameters: lam depends on defense[away], mu depends on defense[home]
            grad_defense = np.zeros(n_teams)
            np.add.at(grad_defense, away_idx, d_lam_total)
            np.add.at(grad_defense, home_idx, d_mu_total)
            grad[n_teams - 1 : 2 * (n_teams - 1)] = grad_defense[:-1] - grad_defense[-1]

            # Home advantage: lam depends on it (non-neutral only)
            grad[-2] = float(np.sum(d_lam_total * (~neutral)))

            # Rho
            grad[-1] = float(np.sum(dtau_drho))

            return nll, grad

        initial = self._build_initial_params(
            teams, n_teams, prev_attack, prev_defense,
            prev_team_to_index, self.home_advantage, self.rho,
        )
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * len(initial)
        bounds[-1] = (-0.25, 0.25)
        result = minimize(
            objective_and_grad,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.config.maxiter},
        )

        if not result.success and "CONVERGENCE" not in result.message:
            import structlog
            structlog.get_logger(__name__).warning(
                "dixon_coles.partial_convergence", message=result.message, nit=result.nit
            )

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
