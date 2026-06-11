"""Phase 2: Optuna TPE numeric tuning (spec section 8.3).

Once Phase 1 locks the structure, Optuna's TPE sampler optimizes the numeric search space
(:mod:`.search_space`) to minimize mean RPS. Every trial is recorded to the leaderboard,
and the best trial is returned with its metrics for the acceptance gate.

Optuna is an optional (``modeling``) dependency, imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polymbappe.tune.leaderboard import Leaderboard
from polymbappe.tune.objective import BacktestObjective, ExperimentMetrics
from polymbappe.tune.search_space import SearchSpace


@dataclass(slots=True)
class OptunaResult:
    """Best config/metrics from a Phase-2 study."""

    best_config: dict[str, Any]
    best_metrics: ExperimentMetrics
    n_trials: int


def run_optuna(
    objective: BacktestObjective,
    search_space: SearchSpace,
    *,
    n_trials: int = 100,
    seed: int = 20260611,
    leaderboard: Leaderboard | None = None,
    record_phase: str = "phase2",
) -> OptunaResult:
    """Run a TPE study minimizing mean RPS over the numeric search space."""

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    state: dict[str, Any] = {"best": None, "best_config": {}, "count": 0}

    def _objective(trial: Any) -> float:
        config = search_space.sample(trial)
        metrics = objective(config)
        state["count"] += 1
        if leaderboard is not None:
            leaderboard.record(
                experiment_id=f"{record_phase}-{trial.number}",
                phase=record_phase,
                decision="trial",
                metrics=metrics,
                config=config,
            )
        if state["best"] is None or metrics.mean_rps < state["best"].mean_rps:
            state["best"] = metrics
            state["best_config"] = config
        return metrics.mean_rps

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(_objective, n_trials=n_trials)
    return OptunaResult(
        best_config=state["best_config"],
        best_metrics=state["best"],
        n_trials=state["count"],
    )
