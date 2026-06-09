"""Acceptance gate and leaderboard persistence (spec sections 8.1, 8.5).

The acceptance gate is the autotuner's quality bar: a candidate config is accepted only
if it improves mean RPS by more than ``min_delta`` (0.003) *and* beats the current best on
at least ``min_tournaments`` (3) of the individual tournament backtests. Marginal results
are logged as inconclusive; clearly worse ones are rejected.

The leaderboard is a Parquet table of every evaluated experiment with its config diff and
metrics, mirroring ``data/outputs/autotune_leaderboard.parquet``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from polymbappe.config import Settings
from polymbappe.tune.objective import ExperimentMetrics

LEADERBOARD_FILE = "autotune_leaderboard.parquet"


@dataclass(slots=True)
class AcceptanceGate:
    """Thresholds for accepting a candidate over the current best (spec 8.1)."""

    min_delta: float = 0.003
    min_tournaments: int = 3

    def decide(self, candidate: ExperimentMetrics, best: ExperimentMetrics | None) -> str:
        """Return ``"accept"``, ``"inconclusive"``, or ``"reject"``."""

        if best is None:
            return "accept"
        delta = best.mean_rps - candidate.mean_rps  # positive = candidate is better
        wins = sum(
            1
            for name, rps in candidate.per_tournament.items()
            if name in best.per_tournament and rps < best.per_tournament[name]
        )
        if delta > self.min_delta and wins >= self.min_tournaments:
            return "accept"
        if delta < -self.min_delta:
            return "reject"
        return "inconclusive"


def accept_config(
    candidate: ExperimentMetrics,
    best: ExperimentMetrics | None,
    gate: AcceptanceGate | None = None,
) -> bool:
    """Convenience boolean: whether ``candidate`` should replace ``best``."""

    return (gate or AcceptanceGate()).decide(candidate, best) == "accept"


@dataclass(slots=True)
class Leaderboard:
    """Append-only experiment leaderboard backed by Parquet."""

    settings: Settings | None = None

    @property
    def path(self) -> Path:
        settings = self.settings or Settings()
        return settings.outputs_data_dir / LEADERBOARD_FILE

    def load(self) -> pl.DataFrame:
        if self.path.exists():
            return pl.read_parquet(self.path)
        return pl.DataFrame(
            schema={
                "experiment_id": pl.Utf8,
                "phase": pl.Utf8,
                "decision": pl.Utf8,
                "mean_rps": pl.Float64,
                "config": pl.Utf8,
                "per_tournament": pl.Utf8,
                "hypothesis": pl.Utf8,
            }
        )

    def record(
        self,
        experiment_id: str,
        phase: str,
        decision: str,
        metrics: ExperimentMetrics,
        config: dict[str, Any],
        hypothesis: str = "",
    ) -> None:
        """Append one experiment row."""

        row = pl.DataFrame(
            {
                "experiment_id": [experiment_id],
                "phase": [phase],
                "decision": [decision],
                "mean_rps": [metrics.mean_rps],
                "config": [json.dumps(config, sort_keys=True)],
                "per_tournament": [json.dumps(metrics.per_tournament, sort_keys=True)],
                "hypothesis": [hypothesis],
            }
        )
        combined = pl.concat([self.load(), row], how="vertical")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self.path)

    def best(self) -> pl.DataFrame:
        """Return the single best (lowest mean RPS) accepted experiment, if any."""

        df = self.load().filter(pl.col("decision") == "accept")
        if df.is_empty():
            return df
        return df.sort("mean_rps").head(1)
