"""Single-match simulation: scoreline sampling, extra time, penalty shootout (spec 4.1).

Group matches sample a scoreline straight from the (contextually-adjusted) Dixon-Coles
score matrix. Knockout matches must produce a winner, so a draw after 90' goes to extra
time (expected goals scaled ``30/90`` with a ``0.85`` fatigue discount) and, if still level,
a penalty shootout.

The shootout uses team-level penalty win rates shrunk toward 50% (the same Bayesian
shrinkage as manager pedigree, spec 2.5) plus a first-shooter advantage
(Apesteguia & Palacios-Huerta 2010). The first shooter is decided by coin toss, so over
many simulations the edge washes out unless one side is systematically stronger.

A knockout winner can be drawn either analytically (``knockout_home_winprob`` — closed-form
P(home advances), used by the Monte Carlo engine so format mitigations like the upset floor
apply cleanly) or by explicit nested sampling (``simulate_knockout_match``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import poisson

from polymbappe.models.dixon_coles import tau_correction

#: Extra-time scaling: 30 minutes of the 90, with a fatigue discount (spec 4.1).
ET_GOAL_SCALE: float = (30.0 / 90.0) * 0.85
DEFAULT_FIRST_SHOOTER_EDGE: float = 0.005
PENALTY_PRIOR_N: float = 10.0

DecidedBy = Literal["regulation", "extra_time", "penalties"]


@dataclass(slots=True)
class MatchOutcome:
    """Result of a simulated match."""

    home_goals: int
    away_goals: int
    home_advances: bool
    decided_by: DecidedBy


def score_matrix_from_rates(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    """Tau-corrected, normalized home-vs-away score matrix for given rates."""

    grid = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(grid, lam), poisson.pmf(grid, mu))
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            matrix[x, y] *= tau_correction(x, y, lam, mu, rho)
    matrix = np.clip(matrix, 0.0, None)
    return np.asarray(matrix / matrix.sum(), dtype=float)


def hda_marginals(matrix: np.ndarray) -> tuple[float, float, float]:
    """Home/draw/away marginal probabilities of a normalized score matrix."""

    home = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, k=1).sum())
    return home, draw, away


def reweight_matrix_to_hda(matrix: np.ndarray, target_hda: np.ndarray) -> np.ndarray:
    """Rescale a score matrix so its H/D/A marginals match ``target_hda``.

    Scales the three outcome regions (home-win lower triangle, draw diagonal, away-win
    upper triangle) by the ratio of target to current marginal, preserving the *shape* of
    the scoreline distribution within each region, then renormalizes. Used to inject a
    contextual H/D/A adjustment into scoreline sampling (spec 4.1 per-match injection).
    Regions with no current mass are left untouched.
    """

    target = np.asarray(target_hda, dtype=float)
    target = target / target.sum()
    cur_h, cur_d, cur_a = hda_marginals(matrix)
    out = matrix.copy()
    n = matrix.shape[0]
    tri_low = np.tril(np.ones((n, n), dtype=bool), k=-1)
    tri_up = np.triu(np.ones((n, n), dtype=bool), k=1)
    diag = np.eye(n, dtype=bool)
    if cur_h > 1e-12:
        out[tri_low] *= target[0] / cur_h
    if cur_d > 1e-12:
        out[diag] *= target[1] / cur_d
    if cur_a > 1e-12:
        out[tri_up] *= target[2] / cur_a
    total = out.sum()
    return np.asarray(out / total, dtype=float) if total > 0 else matrix


def sample_scoreline(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a (home_goals, away_goals) pair from a normalized score matrix."""

    flat = matrix.ravel()
    idx = rng.choice(flat.size, p=flat / flat.sum())
    return int(idx // matrix.shape[1]), int(idx % matrix.shape[1])


def _hda(matrix: np.ndarray) -> tuple[float, float, float]:
    home = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, k=1).sum())
    return home, draw, away


def shrink_penalty_rate(rate: float, n_obs: float, prior_n: float = PENALTY_PRIOR_N) -> float:
    """Shrink a team's observed shootout win rate toward 0.5 (spec 4.1)."""

    if n_obs + prior_n == 0:
        return 0.5
    return (n_obs * rate + prior_n * 0.5) / (n_obs + prior_n)


def penalty_home_winprob(
    home_rate: float,
    away_rate: float,
    first_shooter_home: bool,
    edge: float = DEFAULT_FIRST_SHOOTER_EDGE,
) -> float:
    """P(home wins the shootout) from (already-shrunk) win rates + first-shooter edge.

    Centered at 0.5 when both rates are 0.5; the rate gap shifts it, and the first shooter
    gets ``+edge``. Clamped to ``[0.02, 0.98]``.
    """

    p = 0.5 + (home_rate - away_rate) + (edge if first_shooter_home else -edge)
    return float(np.clip(p, 0.02, 0.98))


def knockout_home_winprob(
    matrix_reg: np.ndarray,
    matrix_et: np.ndarray,
    home_pen_rate: float = 0.5,
    away_pen_rate: float = 0.5,
    first_shooter_home: bool = True,
    edge: float = DEFAULT_FIRST_SHOOTER_EDGE,
) -> float:
    """Closed-form P(home advances) across regulation, extra time, and penalties."""

    rh, rd, _ = _hda(matrix_reg)
    eh, ed, _ = _hda(matrix_et)
    p_pen = penalty_home_winprob(home_pen_rate, away_pen_rate, first_shooter_home, edge)
    return rh + rd * (eh + ed * p_pen)


@dataclass(slots=True)
class KnockoutBreakdown:
    """Decomposition of a knockout tie into advance and phase-decided probabilities.

    ``p_home_advance`` + ``p_away_advance`` == 1, and ``p_decided_reg`` +
    ``p_decided_et`` + ``p_decided_pens`` == 1. The phase probabilities are how the tie
    is *decided* (regulation / extra time / shootout), independent of which side wins.
    """

    p_home_advance: float
    p_away_advance: float
    p_decided_reg: float
    p_decided_et: float
    p_decided_pens: float


def knockout_outcome_breakdown(
    matrix_reg: np.ndarray,
    matrix_et: np.ndarray,
    home_pen_rate: float = 0.5,
    away_pen_rate: float = 0.5,
    first_shooter_home: bool = True,
    edge: float = DEFAULT_FIRST_SHOOTER_EDGE,
) -> KnockoutBreakdown:
    """Closed-form advance probabilities plus the FT/ET/penalties decided-phase split.

    Exposes the intermediate phase probabilities the Monte Carlo engine already computes
    internally (see :func:`knockout_home_winprob`), so the dashboard can show *how* a tie
    is expected to be decided without re-simulating.

    ``matrix_reg`` / ``matrix_et`` are the regulation and extra-time score matrices; the
    ``home_pen_rate`` / ``away_pen_rate`` are the (already-shrunk) shootout win rates.
    """

    rd = float(np.trace(matrix_reg))  # P(draw at 90') -> extra time
    ed = float(np.trace(matrix_et))  # P(draw in ET) -> shootout
    p_home = knockout_home_winprob(
        matrix_reg, matrix_et, home_pen_rate, away_pen_rate, first_shooter_home, edge
    )
    return KnockoutBreakdown(
        p_home_advance=p_home,
        p_away_advance=1.0 - p_home,
        p_decided_reg=1.0 - rd,
        p_decided_et=rd * (1.0 - ed),
        p_decided_pens=rd * ed,
    )


def simulate_group_match(matrix: np.ndarray, rng: np.random.Generator) -> MatchOutcome:
    """Sample a group-stage scoreline (no extra time / penalties)."""

    hg, ag = sample_scoreline(matrix, rng)
    return MatchOutcome(hg, ag, home_advances=hg >= ag, decided_by="regulation")


def simulate_knockout_match(
    matrix_reg: np.ndarray,
    rates_et: tuple[float, float, float],
    rng: np.random.Generator,
    home_pen_rate: float = 0.5,
    away_pen_rate: float = 0.5,
    edge: float = DEFAULT_FIRST_SHOOTER_EDGE,
    max_goals: int = 10,
) -> MatchOutcome:
    """Sample a knockout result: regulation -> extra time -> shootout.

    ``rates_et`` is ``(lam, mu, rho)`` for full-time; extra-time rates are scaled by
    :data:`ET_GOAL_SCALE`.
    """

    hg, ag = sample_scoreline(matrix_reg, rng)
    if hg != ag:
        return MatchOutcome(hg, ag, home_advances=hg > ag, decided_by="regulation")

    lam, mu, rho = rates_et
    matrix_et = score_matrix_from_rates(lam * ET_GOAL_SCALE, mu * ET_GOAL_SCALE, rho, max_goals)
    eh, ea = sample_scoreline(matrix_et, rng)
    hg, ag = hg + eh, ag + ea
    if hg != ag:
        return MatchOutcome(hg, ag, home_advances=hg > ag, decided_by="extra_time")

    first_shooter_home = bool(rng.random() < 0.5)
    p_pen = penalty_home_winprob(home_pen_rate, away_pen_rate, first_shooter_home, edge)
    home_advances = bool(rng.random() < p_pen)
    return MatchOutcome(hg, ag, home_advances=home_advances, decided_by="penalties")
