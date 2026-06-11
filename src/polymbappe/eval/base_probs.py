"""Per-match base-model probabilities for stacking.

Produces the H/D/A probability features the meta-learner consumes: Dixon-Coles (fit on
prior matches), Elo (point-in-time, parametric three-way), market-implied (from the
``market_odds`` table when available), and — when ``use_bayesian`` is set — the Bayesian
hierarchical Dixon-Coles posterior means plus per-outcome credible intervals. All are
leakage-safe: DC and the Bayesian model train only on history before the tournament, Elo
uses pre-match ratings, market odds are external.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from polymbappe.features.elo import EloConfig, build_elo_features
from polymbappe.features.pipeline import result_label
from polymbappe.models.bayesian_dc import BayesianConfig, BayesianDixonColesModel
from polymbappe.models.dixon_coles import DixonColesConfig, DixonColesModel, MatchObservation


@dataclass(slots=True)
class BaseProbConfig:
    """Configuration for base-probability construction."""

    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    draw_max: float = 0.28
    use_bayesian: bool = False
    """Fit the Bayesian hierarchical DC model and emit ``bay_*`` + ``ci_*`` columns.

    Off by default: NUTS sampling is expensive (~hours per tournament), so this is never
    enabled inside the autotuner's per-trial TPE loop. Enabled by ``models.train`` (via the
    ``--bayesian`` flag) and the standalone ``compare_bayesian_ab`` A/B harness.
    """
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    credible_level: float = 0.9
    """Credibility level for the ``ci_*`` interval bounds (spec 3.6 edge test)."""


def matches_to_observations(matches: pl.DataFrame, reference_date: date) -> list[MatchObservation]:
    """Convert a matches frame into Dixon-Coles observations (days_ago vs reference)."""

    obs: list[MatchObservation] = []
    for row in matches.iter_rows(named=True):
        days_ago = float(max((reference_date - row["date"]).days, 0))
        obs.append(
            MatchObservation(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                days_ago=days_ago,
                competition=row["competition"],
                neutral_site=bool(row["neutral_site"]),
            )
        )
    return obs


def elo_probabilities(expected_home: np.ndarray, draw_max: float = 0.28) -> np.ndarray:
    """Map Elo expected home score in [0, 1] to [home, draw, away] probabilities.

    Draw probability peaks for even matchups and decays as the matchup becomes lopsided;
    the remaining mass splits around the expected score. Returns an (n, 3) array whose
    rows sum to 1.
    """

    e = np.clip(expected_home, 1e-6, 1.0 - 1e-6)
    draw = np.clip(draw_max * (1.0 - 2.0 * np.abs(e - 0.5)), 0.02, draw_max)
    home = np.clip(e - draw / 2.0, 1e-6, None)
    away = np.clip((1.0 - e) - draw / 2.0, 1e-6, None)
    stacked = np.stack([home, draw, away], axis=1)
    return np.asarray(stacked / stacked.sum(axis=1, keepdims=True), dtype=float)


def _dc_match_probs(model: DixonColesModel, home: str, away: str, neutral: bool) -> tuple[
    float, float, float
]:
    if home not in model.team_to_index or away not in model.team_to_index:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    matrix = model.predict_score_matrix(home, away, neutral_site=neutral)
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    return home_win, draw, away_win


def compute_tournament_base_probs(
    history: pl.DataFrame,
    fixtures: pl.DataFrame,
    *,
    tournament: str,
    config: BaseProbConfig | None = None,
    market_odds: pl.DataFrame | None = None,
    dc_model: DixonColesModel | None = None,
    bayesian_model: BayesianDixonColesModel | None = None,
) -> pl.DataFrame:
    """Base H/D/A probabilities for every fixture in one tournament.

    Args:
        history: Matches strictly before the tournament (Dixon-Coles training data).
        fixtures: The tournament's matches (with results, for labels).
        tournament: Tournament name (carried through for grouping).
        config: Base-probability configuration.
        market_odds: Optional ``market_odds`` table; ``mkt_*`` columns added when present.
        dc_model: Optional pre-existing model instance for warm-starting.
        bayesian_model: Optional pre-fit Bayesian DC model (reused without refitting when
            its posterior is already present); only consulted when ``config.use_bayesian``.

    Returns:
        One row per fixture with ``match_id``, ``tournament``, ``label``, ``dc_*``,
        ``elo_*``, optionally ``mkt_*``, and — when ``config.use_bayesian`` — ``bay_*``
        posterior-mean probabilities plus ``ci_*_low``/``ci_*_high`` credible bounds.
    """

    config = config or BaseProbConfig()
    reference_date = fixtures["date"].min()
    assert isinstance(reference_date, date)

    history_obs = matches_to_observations(history, reference_date)
    model = dc_model or DixonColesModel(config.dixon_coles)
    model.fit(matches=history_obs)

    # The Bayesian model trains on the identical leakage-safe ``history`` observations. A
    # pre-fit instance is reused as-is (``attack`` is set only after a fit); otherwise it is
    # fit here. Fitting is gated behind ``use_bayesian`` because NUTS is expensive.
    bayes: BayesianDixonColesModel | None = None
    if config.use_bayesian:
        bayes = bayesian_model or BayesianDixonColesModel(config.bayesian)
        if bayes.attack is None:
            bayes.fit(matches=history_obs)

    timeline = pl.concat([history, fixtures], how="vertical").unique(
        subset=["match_id"], keep="first"
    )
    elo_feats = build_elo_features(timeline, config=config.elo)
    elo_lookup = {
        (row["match_id"], row["team"]): row["elo_pre"]
        for row in elo_feats.iter_rows(named=True)
    }
    home_adv = config.elo.home_advantage

    rows: list[dict[str, object]] = []
    expected_home: list[float] = []
    for fx in fixtures.iter_rows(named=True):
        home, away, mid = fx["home_team"], fx["away_team"], fx["match_id"]
        neutral = bool(fx["neutral_site"])
        dc_h, dc_d, dc_a = _dc_match_probs(model, home, away, neutral)

        elo_home = float(elo_lookup.get((mid, home), config.elo.base_rating))
        elo_away = float(elo_lookup.get((mid, away), config.elo.base_rating))
        adv = 0.0 if neutral else home_adv
        expected_home.append(1.0 / (1.0 + 10.0 ** (-((elo_home - elo_away) + adv) / 400.0)))

        row: dict[str, object] = {
            "match_id": mid,
            "tournament": tournament,
            "label": result_label(int(fx["home_goals"]), int(fx["away_goals"])),
            "dc_home": dc_h,
            "dc_draw": dc_d,
            "dc_away": dc_a,
        }
        if bayes is not None:
            bp = bayes.predict_match(home, away, neutral)
            ci = bayes.credible_interval(home, away, neutral, level=config.credible_level)
            row.update(
                {
                    "bay_home": bp["home_win"],
                    "bay_draw": bp["draw"],
                    "bay_away": bp["away_win"],
                    "ci_home_low": ci["home_win"][0],
                    "ci_home_high": ci["home_win"][1],
                    "ci_draw_low": ci["draw"][0],
                    "ci_draw_high": ci["draw"][1],
                    "ci_away_low": ci["away_win"][0],
                    "ci_away_high": ci["away_win"][1],
                }
            )
        rows.append(row)

    elo_probs = elo_probabilities(np.array(expected_home), config.draw_max)
    df = pl.DataFrame(rows).with_columns(
        pl.Series("elo_home", elo_probs[:, 0]),
        pl.Series("elo_draw", elo_probs[:, 1]),
        pl.Series("elo_away", elo_probs[:, 2]),
    )

    if market_odds is not None:
        odds = market_odds.select(
            "match_id",
            pl.col("home_win_prob").alias("mkt_home"),
            pl.col("draw_prob").alias("mkt_draw"),
            pl.col("away_win_prob").alias("mkt_away"),
        )
        df = df.join(odds, on="match_id", how="left")

    return df
