"""Backtest objective for the autotuner (spec section 8.1).

Maps a flat config dict (namespaced ``group.param`` keys from the search space, or
structural overrides from Phase 1) onto the model configs and runs the
leave-one-tournament-out backtest, returning per-tournament and mean RPS. This is the
fitness function both phases optimize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from polymbappe.eval.backtest import (
    DEFAULT_TOURNAMENTS,
    Tournament,
    run_leave_one_tournament_out,
)
from polymbappe.eval.base_probs import BaseProbConfig
from polymbappe.features.elo import EloConfig
from polymbappe.models.dixon_coles import DixonColesConfig
from polymbappe.models.meta import MetaConfig


def _get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First present value among the given namespaced keys."""

    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def config_to_configs(config: dict[str, Any]) -> tuple[BaseProbConfig, MetaConfig]:
    """Translate a flat tuner config into :class:`BaseProbConfig` / :class:`MetaConfig`."""

    dc = DixonColesConfig(
        xi=float(_get(config, "dixon_coles.xi", default=0.0019)),
        friendly_weight=float(_get(config, "dixon_coles.friendly_weight", default=0.3)),
        max_goals=int(_get(config, "dixon_coles.max_goals", default=10)),
    )
    elo = EloConfig(k_factor=float(_get(config, "features.elo_k_factor", default=20.0)))
    base = BaseProbConfig(
        dixon_coles=dc,
        elo=elo,
        draw_max=float(_get(config, "features.draw_max", default=0.28)),
    )
    meta = MetaConfig(C=float(_get(config, "ensemble.meta_C", default=1.0)))
    return base, meta


@dataclass(slots=True)
class ExperimentMetrics:
    """Result of one backtest evaluation."""

    mean_rps: float
    per_tournament: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)


def config_to_metrics(
    config: dict[str, Any],
    matches: pl.DataFrame,
    *,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    market_odds: pl.DataFrame | None = None,
) -> ExperimentMetrics:
    """Evaluate one config via the leave-one-tournament-out backtest."""

    base, meta = config_to_configs(config)
    result = run_leave_one_tournament_out(
        matches, tournaments, base_config=base, meta_config=meta, market_odds=market_odds
    )
    return ExperimentMetrics(
        mean_rps=result.mean_rps,
        per_tournament={k: v["rps"] for k, v in result.per_tournament.items()},
        feature_columns=result.feature_columns,
    )


@dataclass(slots=True)
class BacktestObjective:
    """Callable objective bound to a fixed dataset (for Optuna / structural search)."""

    matches: pl.DataFrame
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS
    market_odds: pl.DataFrame | None = None

    def __call__(self, config: dict[str, Any]) -> ExperimentMetrics:
        return config_to_metrics(
            config,
            self.matches,
            tournaments=self.tournaments,
            market_odds=self.market_odds,
        )
