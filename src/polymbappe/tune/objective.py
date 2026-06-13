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

from polymbappe.context.adjuster import ContextualAdjusterConfig
from polymbappe.eval.backtest import (
    DEFAULT_TOURNAMENTS,
    Tournament,
    run_leave_one_tournament_out,
)
from polymbappe.eval.base_probs import BaseProbConfig
from polymbappe.features.elo import EloConfig
from polymbappe.models.dixon_coles import DixonColesConfig
from polymbappe.models.ensemble import EnsembleConfig
from polymbappe.models.gbm import GBMConfig
from polymbappe.models.meta import MetaConfig


def _get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First present value among the given namespaced keys."""

    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


@dataclass(slots=True)
class BacktestConfigs:
    """Fully-resolved model configs the backtest objective consumes.

    Bundles the base-probability config, the stacking-ensemble config (carrying the
    meta-learner and optional GBM hyperparameters), and the optional contextual-adjuster
    config, so every group of the autotuner search space maps onto a live knob.
    """

    base: BaseProbConfig
    ensemble: EnsembleConfig
    contextual: ContextualAdjusterConfig


def config_to_configs(config: dict[str, Any]) -> BacktestConfigs:
    """Translate a flat tuner config into the resolved backtest configs.

    Defaults reproduce the historical baseline (Dixon-Coles + Elo stacked by a logistic
    meta-learner, no GBM, no contextual layer), so an empty config measures the same
    baseline as before while every sampled group now feeds a real knob.
    """

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

    meta = MetaConfig(
        C=float(_get(config, "ensemble.meta_C", default=1.0)),
        learner=str(_get(config, "ensemble.meta_learner", default="logistic")),
    )
    gbm = GBMConfig(
        num_leaves=int(_get(config, "gbm.num_leaves", default=31)),
        learning_rate=float(_get(config, "gbm.learning_rate", default=0.05)),
        n_estimators=int(_get(config, "gbm.n_estimators", default=300)),
        min_child_samples=int(_get(config, "gbm.min_child_samples", default=20)),
    )
    ensemble = EnsembleConfig(
        use_gbm=bool(_get(config, "gbm.enable", default=False)),
        meta=meta,
        gbm=gbm,
    )

    contextual = ContextualAdjusterConfig(
        enable_contextual_layer=bool(
            _get(config, "contextual.enable_contextual_layer", default=False)
        ),
        num_leaves=int(_get(config, "contextual.context_num_leaves", default=15)),
        n_estimators=int(_get(config, "contextual.context_n_estimators", default=100)),
        toggle_xg_overperformance=bool(
            _get(config, "contextual.toggle_xg_overperformance", default=True)
        ),
        toggle_draw_pressure=bool(_get(config, "contextual.toggle_draw_pressure", default=True)),
        toggle_cohesion=bool(_get(config, "contextual.toggle_cohesion", default=True)),
        toggle_manager=bool(_get(config, "contextual.toggle_manager", default=True)),
    )

    return BacktestConfigs(base=base, ensemble=ensemble, contextual=contextual)


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
    squad_valuations: pl.DataFrame | None = None,
) -> ExperimentMetrics:
    """Evaluate one config via the leave-one-tournament-out backtest.

    ``squad_valuations`` (when supplied) stacks the Tier-1 ``squad_value_ratio`` into the GBM;
    ``features.toggle_squad_value`` (default True) lets the tuner switch it in/out so its RPS
    contribution is measurable. The feature only has a path when the GBM is enabled.
    """

    configs = config_to_configs(config)
    use_squad = bool(_get(config, "features.toggle_squad_value", default=True))
    result = run_leave_one_tournament_out(
        matches,
        tournaments,
        base_config=configs.base,
        ensemble_config=configs.ensemble,
        contextual_config=configs.contextual,
        market_odds=market_odds,
        squad_valuations=squad_valuations if use_squad else None,
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
    squad_valuations: pl.DataFrame | None = None

    def __call__(self, config: dict[str, Any]) -> ExperimentMetrics:
        return config_to_metrics(
            config,
            self.matches,
            tournaments=self.tournaments,
            market_odds=self.market_odds,
            squad_valuations=self.squad_valuations,
        )
