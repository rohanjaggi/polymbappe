"""Bayesian hierarchical Dixon-Coles model (PyMC), spec section 3.2.

A hierarchical extension of the MLE Dixon-Coles baseline. Team attack/defense
strengths are partially pooled toward confederation-level means, the home advantage
carries a weakly-informative literature prior, and the Dixon-Coles ``rho`` correlation
is bounded. Inference is NUTS; the key product is a *posterior predictive distribution*
over H/D/A probabilities (not just a point estimate), giving principled credible
intervals — the property the edge pipeline (spec 3.6) relies on.

Likelihood: two Poissons with the low-score tau correction applied via ``pm.Potential``,
each observation weighted by an exponential time-decay ``exp(-xi * days_ago)`` so recent
matches dominate (the Bayesian analogue of the MLE model's weighting).

Optional time-varying strengths (spec 3.2 "random walk at per-match granularity") are
approximated with a per-period Gaussian random walk: per-match granularity over tens of
thousands of matches is intractable for NUTS, so matches are binned into periods
(default yearly) and each team's strength follows a random walk across periods. Enabled
via :attr:`BayesianConfig.time_varying`; off by default.

PyMC is an optional (``modeling``) dependency and is imported lazily so the package
imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import poisson

from polymbappe.models.base import MatchModel
from polymbappe.models.dixon_coles import MatchObservation, tau_correction

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_UNKNOWN_CONF = "UNKNOWN"


@dataclass(slots=True)
class BayesianConfig:
    """Bayesian Dixon-Coles hyperparameters and sampler settings."""

    xi: float = 0.0019
    """Exponential time-decay rate applied to each observation's log-likelihood."""
    confederation_sigma_prior: float = 0.5
    """Scale of the HalfNormal prior on between-team strength spread per confederation."""
    home_advantage_prior_mean: float = 0.25
    home_advantage_prior_sd: float = 0.1
    sigma_walk: float = 0.02
    """Random-walk step sd for time-varying strengths (only used when ``time_varying``)."""
    time_varying: bool = False
    period_days: int = 365
    """Bin width (days) for the time-varying random walk periods."""
    n_tune: int = 1000
    n_draws: int = 1000
    chains: int = 2
    cores: int = 1
    target_accept: float = 0.9
    max_goals: int = 10
    random_seed: int = 20260611


@dataclass(slots=True)
class _Posterior:
    """Flattened posterior draws (chains * draws collapsed)."""

    attack: np.ndarray  # (n_samples, n_teams)
    defense: np.ndarray  # (n_samples, n_teams)
    home_advantage: np.ndarray  # (n_samples,)
    rho: np.ndarray  # (n_samples,)


def _hda_from_rates(
    lam: float, mu: float, rho: float, max_goals: int
) -> tuple[float, float, float]:
    """Tau-corrected H/D/A probabilities for a single (lam, mu, rho)."""

    grid = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(grid, lam)
    away_pmf = poisson.pmf(grid, mu)
    matrix = np.outer(home_pmf, away_pmf)
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            matrix[x, y] *= tau_correction(x, y, lam, mu, rho)
    matrix = np.clip(matrix, 0.0, None)
    matrix /= matrix.sum()
    home = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, k=1).sum())
    return home, draw, away


class BayesianDixonColesModel(MatchModel):
    """PyMC hierarchical Dixon-Coles with posterior predictive H/D/A draws."""

    def __init__(self, config: BayesianConfig | None = None) -> None:
        self.config = config or BayesianConfig()
        self.team_to_index: dict[str, int] = {}
        self.index_to_team: list[str] = []
        self.team_confederation: dict[str, str] = {}
        self._posterior: _Posterior | None = None
        # Point estimates (posterior means) for fast scoreline matrices.
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_advantage: float = 0.0
        self.rho: float = 0.0

    # -- fitting ---------------------------------------------------------------

    def fit(self, *args: object, **kwargs: object) -> BayesianDixonColesModel:
        """Fit the hierarchical model with NUTS.

        Args (via kwargs or positional):
            matches: ``list[MatchObservation]``.
            team_confederation: optional ``dict[str, str]`` mapping team -> confederation
                code. Teams missing from the map are pooled under ``"UNKNOWN"``.
        """

        matches_obj = kwargs.get("matches", args[0] if args else None)
        if not isinstance(matches_obj, list) or (
            matches_obj and not isinstance(matches_obj[0], MatchObservation)
        ):
            raise TypeError("fit expects a list[MatchObservation].")
        matches = cast("list[MatchObservation]", matches_obj)
        if not matches:
            raise ValueError("At least one match is required to fit the model.")

        conf_map_obj = kwargs.get("team_confederation")
        conf_map: dict[str, str] = (
            dict(conf_map_obj) if isinstance(conf_map_obj, dict) else {}
        )

        import pymc as pm  # lazy: optional dependency

        teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
        self.team_to_index = {team: idx for idx, team in enumerate(teams)}
        self.index_to_team = teams
        n_teams = len(teams)

        self.team_confederation = {t: conf_map.get(t, _UNKNOWN_CONF) for t in teams}
        confs = sorted(set(self.team_confederation.values()))
        conf_to_index = {c: i for i, c in enumerate(confs)}
        team_conf_idx = np.array([conf_to_index[self.team_confederation[t]] for t in teams])

        home_idx = np.array([self.team_to_index[m.home_team] for m in matches])
        away_idx = np.array([self.team_to_index[m.away_team] for m in matches])
        home_goals = np.array([m.home_goals for m in matches])
        away_goals = np.array([m.away_goals for m in matches])
        is_home = np.array([0.0 if m.neutral_site else 1.0 for m in matches])
        weights = np.exp(-self.config.xi * np.array([m.days_ago for m in matches]))

        cfg = self.config
        coords = {"team": teams, "conf": confs}
        with pm.Model(coords=coords):
            sigma_att = pm.HalfNormal("sigma_att", sigma=cfg.confederation_sigma_prior)
            sigma_def = pm.HalfNormal("sigma_def", sigma=cfg.confederation_sigma_prior)
            mu_att = pm.Normal("mu_att", mu=0.0, sigma=cfg.confederation_sigma_prior, dims="conf")
            mu_def = pm.Normal("mu_def", mu=0.0, sigma=cfg.confederation_sigma_prior, dims="conf")

            attack = pm.Normal(
                "attack", mu=mu_att[team_conf_idx], sigma=sigma_att, dims="team"
            )
            defense = pm.Normal(
                "defense", mu=mu_def[team_conf_idx], sigma=sigma_def, dims="team"
            )
            home_adv = pm.Normal(
                "home_advantage",
                mu=cfg.home_advantage_prior_mean,
                sigma=cfg.home_advantage_prior_sd,
            )
            rho = pm.Uniform("rho", lower=-0.25, upper=0.25)

            att_h = attack[home_idx]
            att_a = attack[away_idx]
            def_h = defense[home_idx]
            def_a = defense[away_idx]

            if cfg.time_varying:
                self._add_time_varying(pm, matches, home_idx, away_idx, n_teams, cfg)
                # Time-varying deltas are added inside the helper via a Deterministic that
                # we re-read here through the model context.
                walk_h, walk_a = self._time_varying_terms
                att_h = att_h + walk_h[0]
                att_a = att_a + walk_a[0]
                def_h = def_h + walk_h[1]
                def_a = def_a + walk_a[1]

            log_lam = home_adv * is_home + att_h + def_a
            log_mu = att_a + def_h
            lam = pm.math.exp(log_lam)
            mu = pm.math.exp(log_mu)

            from pytensor.tensor import gammaln  # pm.math has no gammaln re-export

            base_logp = (
                home_goals * log_lam - lam - gammaln(home_goals + 1)
            ) + (away_goals * log_mu - mu - gammaln(away_goals + 1))
            log_tau = self._log_tau(pm, home_goals, away_goals, lam, mu, rho)
            pm.Potential("likelihood", pm.math.sum(weights * (base_logp + log_tau)))

            idata = pm.sample(
                draws=cfg.n_draws,
                tune=cfg.n_tune,
                chains=cfg.chains,
                cores=cfg.cores,
                target_accept=cfg.target_accept,
                random_seed=cfg.random_seed,
                progressbar=False,
                compute_convergence_checks=False,
            )

        post = idata.posterior
        self._posterior = _Posterior(
            attack=post["attack"].to_numpy().reshape(-1, n_teams),
            defense=post["defense"].to_numpy().reshape(-1, n_teams),
            home_advantage=post["home_advantage"].to_numpy().reshape(-1),
            rho=post["rho"].to_numpy().reshape(-1),
        )
        self.attack = self._posterior.attack.mean(axis=0)
        self.defense = self._posterior.defense.mean(axis=0)
        self.home_advantage = float(self._posterior.home_advantage.mean())
        self.rho = float(self._posterior.rho.mean())
        return self

    _time_varying_terms: tuple[tuple[object, object], tuple[object, object]]

    def _add_time_varying(
        self,
        pm: object,
        matches: list[MatchObservation],
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        n_teams: int,
        cfg: BayesianConfig,
    ) -> None:
        """Per-period Gaussian random-walk strength deltas (coarse approximation)."""

        import pymc as _pm

        days = np.array([m.days_ago for m in matches])
        # Period 0 = most recent; older matches get higher period indices.
        periods = (days // cfg.period_days).astype(int)
        n_periods = int(periods.max()) + 1
        # Random walk over periods per team (delta on attack & defense), anchored at 0
        # for the most recent period so it identifies against the static attack/defense.
        init = _pm.Normal.dist(0.0, 0.1)
        walk_att = _pm.GaussianRandomWalk(
            "walk_att",
            sigma=cfg.sigma_walk,
            init_dist=init,
            steps=n_periods - 1,
            shape=(n_teams, n_periods),
        )
        walk_def = _pm.GaussianRandomWalk(
            "walk_def",
            sigma=cfg.sigma_walk,
            init_dist=init,
            steps=n_periods - 1,
            shape=(n_teams, n_periods),
        )
        _ = (home_idx, away_idx, n_teams)
        match_period = periods
        att_h = walk_att[home_idx, match_period]
        att_a = walk_att[away_idx, match_period]
        def_h = walk_def[home_idx, match_period]
        def_a = walk_def[away_idx, match_period]
        self._time_varying_terms = ((att_h, def_h), (att_a, def_a))

    @staticmethod
    def _log_tau(
        pm: object, x: np.ndarray, y: np.ndarray, lam: object, mu: object, rho: object
    ) -> object:
        """Log of the Dixon-Coles tau correction, vectorized over matches."""

        import pymc as _pm

        is00 = (x == 0) & (y == 0)
        is01 = (x == 0) & (y == 1)
        is10 = (x == 1) & (y == 0)
        is11 = (x == 1) & (y == 1)
        tau = (
            _pm.math.switch(is00, 1.0 - lam * mu * rho, 1.0)
            * _pm.math.switch(is01, 1.0 + lam * rho, 1.0)
            * _pm.math.switch(is10, 1.0 + mu * rho, 1.0)
            * _pm.math.switch(is11, 1.0 - rho, 1.0)
        )
        return _pm.math.log(_pm.math.clip(tau, 1e-12, np.inf))

    # -- prediction ------------------------------------------------------------

    def _require_fit(self) -> _Posterior:
        if self._posterior is None or self.attack is None or self.defense is None:
            raise RuntimeError("Model must be fit before predicting.")
        return self._posterior

    def _rates(self, home_team: str, away_team: str, neutral: bool) -> tuple[float, float]:
        assert self.attack is not None and self.defense is not None
        h = self.team_to_index[home_team]
        a = self.team_to_index[away_team]
        home_term = 0.0 if neutral else self.home_advantage
        lam = float(np.exp(home_term + self.attack[h] + self.defense[a]))
        mu = float(np.exp(self.attack[a] + self.defense[h]))
        return lam, mu

    def predict_match(
        self, home_team: str, away_team: str, neutral_site: bool = False
    ) -> dict[str, float]:
        """Posterior-mean H/D/A probabilities."""

        self._require_fit()
        if home_team not in self.team_to_index or away_team not in self.team_to_index:
            return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
        lam, mu = self._rates(home_team, away_team, neutral_site)
        home, draw, away = _hda_from_rates(lam, mu, self.rho, self.config.max_goals)
        return {"home_win": home, "draw": draw, "away_win": away}

    def predict_proba_draws(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
        max_draws: int = 400,
    ) -> np.ndarray:
        """Per-posterior-draw H/D/A probabilities, shape ``(n, 3)``.

        Subsamples to ``max_draws`` draws for speed. Each row is a tau-corrected
        [home, draw, away] distribution from one posterior sample, so the spread across
        rows is the model's epistemic uncertainty.
        """

        post = self._require_fit()
        if home_team not in self.team_to_index or away_team not in self.team_to_index:
            return np.full((1, 3), 1 / 3)
        h = self.team_to_index[home_team]
        a = self.team_to_index[away_team]
        n = post.attack.shape[0]
        sel = np.arange(n) if n <= max_draws else np.linspace(0, n - 1, max_draws).astype(int)
        home_term = 0.0 if neutral_site else None
        out = np.empty((len(sel), 3))
        for i, s in enumerate(sel):
            ha = post.home_advantage[s] if home_term is None else 0.0
            lam = float(np.exp(ha + post.attack[s, h] + post.defense[s, a]))
            mu = float(np.exp(post.attack[s, a] + post.defense[s, h]))
            out[i] = _hda_from_rates(lam, mu, float(post.rho[s]), self.config.max_goals)
        return out

    def credible_interval(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
        level: float = 0.9,
    ) -> dict[str, tuple[float, float]]:
        """Per-outcome credible interval at ``level`` from the posterior draws."""

        draws = self.predict_proba_draws(home_team, away_team, neutral_site)
        lo_q = (1.0 - level) / 2.0 * 100.0
        hi_q = (1.0 + level) / 2.0 * 100.0
        lows = np.percentile(draws, lo_q, axis=0)
        highs = np.percentile(draws, hi_q, axis=0)
        return {
            "home_win": (float(lows[0]), float(highs[0])),
            "draw": (float(lows[1]), float(highs[1])),
            "away_win": (float(lows[2]), float(highs[2])),
        }
