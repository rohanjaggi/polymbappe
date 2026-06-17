"""Autotuner orchestration and CLI entrypoint (spec sections 8.1, 8.4).

Runs Phase 1 (structural search) then Phase 2 (Optuna TPE), gating every candidate through
the acceptance criteria and logging to the leaderboard. The best accepted config can be
serialized to ``configs/best_config.yaml`` via ``--apply-best``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import structlog
import yaml

from polymbappe.config import Settings
from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS, Tournament
from polymbappe.tune.leaderboard import AcceptanceGate, Leaderboard
from polymbappe.tune.llm_search import default_structural_experiments, propose_structural_experiment
from polymbappe.tune.objective import BacktestObjective, ExperimentMetrics, config_to_metrics
from polymbappe.tune.optuna_tuner import run_optuna
from polymbappe.tune.search_space import load_search_space

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class AutotuneResult:
    """Outcome of an autotune run."""

    best_config: dict[str, Any]
    best_metrics: ExperimentMetrics
    baseline_metrics: ExperimentMetrics
    history: list[dict[str, Any]] = field(default_factory=list)


def parse_budget_to_trials(budget: str, trials_per_hour: int = 60) -> int:
    """Map a ``"2h"`` / ``"30m"`` budget string to an approximate Phase-2 trial count."""

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([hm])\s*", budget.lower())
    if not match:
        return trials_per_hour
    value, unit = float(match.group(1)), match.group(2)
    hours = value if unit == "h" else value / 60.0
    return max(1, int(hours * trials_per_hour))


def autotune(
    matches: pl.DataFrame,
    *,
    market_odds: pl.DataFrame | None = None,
    squad_valuations: pl.DataFrame | None = None,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    n_structural: int | None = None,
    n_trials: int = 30,
    gate: AcceptanceGate | None = None,
    leaderboard: Leaderboard | None = None,
    llm_model: str | None = None,
) -> AutotuneResult:
    """Run the two-phase autoresearch loop and return the best accepted config."""

    gate = gate or AcceptanceGate()
    leaderboard = leaderboard or Leaderboard()
    objective = BacktestObjective(
        matches=matches,
        tournaments=tournaments,
        market_odds=market_odds,
        squad_valuations=squad_valuations,
    )

    baseline = config_to_metrics(
        {},
        matches,
        tournaments=tournaments,
        market_odds=market_odds,
        squad_valuations=squad_valuations,
    )
    best_metrics = baseline
    best_config: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    # -- Phase 1: structural search --
    experiments = default_structural_experiments()
    limit = n_structural if n_structural is not None else len(experiments)
    propose_kwargs = {"model": llm_model} if llm_model else {}
    for _ in range(limit):
        exp = propose_structural_experiment(history, **propose_kwargs)
        odds = None if exp.exclude_market else market_odds
        metrics = config_to_metrics(
            exp.config,
            matches,
            tournaments=tournaments,
            market_odds=odds,
            squad_valuations=squad_valuations,
        )
        decision = gate.decide(metrics, best_metrics)
        leaderboard.record(exp.name, "phase1", decision, metrics, exp.config, exp.hypothesis)
        history.append({"name": exp.name, "mean_rps": metrics.mean_rps, "decision": decision})
        if decision == "accept":
            best_metrics, best_config = metrics, exp.config
        # An experiment that lands exactly on the baseline RPS changed no live knob; flag it
        # so a no-op is visible rather than hiding behind a bare "inconclusive".
        no_op = abs(metrics.mean_rps - baseline.mean_rps) < 1e-9
        logger.info(
            "autotune.phase1",
            name=exp.name,
            rps=round(metrics.mean_rps, 4),
            decision=decision,
            no_op=no_op,
        )

    # -- Phase 2: numeric TPE within the locked structure --
    if n_trials > 0:
        space = load_search_space()
        result = run_optuna(objective, space, n_trials=n_trials, leaderboard=leaderboard)
        decision = gate.decide(result.best_metrics, best_metrics)
        leaderboard.record(
            "phase2-best", "phase2", decision, result.best_metrics, result.best_config
        )
        history.append(
            {"name": "phase2-best", "mean_rps": result.best_metrics.mean_rps, "decision": decision}
        )
        if decision == "accept":
            best_metrics, best_config = result.best_metrics, result.best_config

    return AutotuneResult(
        best_config=best_config,
        best_metrics=best_metrics,
        baseline_metrics=baseline,
        history=history,
    )


def apply_best_config(result: AutotuneResult, settings: Settings | None = None) -> None:
    """Serialize the best config to ``configs/best_config.yaml`` (spec 8.5)."""

    settings = settings or Settings()
    path = settings.configs_dir / "best_config.yaml"
    payload = {
        "best_config": result.best_config,
        "meta": {
            "mean_rps": result.best_metrics.mean_rps,
            "baseline_mean_rps": result.baseline_metrics.mean_rps,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True)
    logger.info("autotune.applied_best", path=str(path), mean_rps=result.best_metrics.mean_rps)


def run_autotune(
    budget: str = "2h",
    metric: str = "rps",
    resume: bool = False,
    leaderboard: bool = False,
    apply_best: bool = False,
) -> None:
    """CLI entrypoint for ``polymbappe autotune``."""

    _ = (metric, resume)
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    board = Leaderboard(settings)

    if leaderboard:
        print(board.load().sort("mean_rps"))
        return

    matches = read_table(Table.MATCHES, settings)
    market_odds = (
        read_table(Table.MARKET_ODDS, settings)
        if table_exists(Table.MARKET_ODDS, settings)
        else None
    )
    squad_valuations = (
        read_table(Table.SQUAD_VALUATIONS, settings)
        if table_exists(Table.SQUAD_VALUATIONS, settings)
        else None
    )
    n_trials = parse_budget_to_trials(budget)
    result = autotune(
        matches,
        market_odds=market_odds,
        squad_valuations=squad_valuations,
        n_trials=n_trials,
        leaderboard=board,
        llm_model=settings.autotune_llm_model,
    )
    print(
        f"baseline RPS={result.baseline_metrics.mean_rps:.4f} -> "
        f"best RPS={result.best_metrics.mean_rps:.4f}"
    )
    if apply_best:
        apply_best_config(result, settings)
