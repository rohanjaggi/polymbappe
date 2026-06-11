"""Manager tournament-pedigree features with Bayesian shrinkage (spec 2.2 Group C, 2.5).

Four signals derived from a manager's tournament match record:

* **Knockout win rate** — share of knockout matches won (shrunk).
* **Deepest run (weighted recency)** — furthest stage reached, exponentially decayed
  across tournaments (``lambda=0.3`` per tournament back from the most recent).
* **Knockout conversion rate** — wins / knockout matches (group stage excluded; shrunk).
* **Tenure matches** — competitive matches managed for the current national team.

Thin tournament records are pulled toward the global mean via Bayesian shrinkage
(spec 2.5)::

    effective_rate = (n * observed_rate + prior_n * prior_rate) / (n + prior_n)

so a manager with one tournament does not get an extreme rate on a handful of matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

#: Ordinal value per knockout stage reached, used for the "deepest run" signal.
STAGE_DEPTH: dict[str, int] = {
    "group": 0,
    "R32": 1,
    "R16": 2,
    "QF": 3,
    "SF": 4,
    "THIRD": 4,
    "final": 5,
    "FINAL": 5,
    "winner": 6,
    "champion": 6,
}


@dataclass(slots=True)
class ManagerConfig:
    """Manager-pedigree shrinkage and recency settings."""

    prior_n: float = 4.0
    """Pseudo-count strength of the global-mean prior (spec 2.5)."""
    recency_lambda: float = 0.3
    """Exponential decay per tournament for the deepest-run signal."""


def shrink(observed_rate: float, n: float, prior_rate: float, prior_n: float) -> float:
    """Bayesian-shrunk rate toward ``prior_rate`` (spec 2.5)."""

    if n + prior_n == 0:
        return prior_rate
    return (n * observed_rate + prior_n * prior_rate) / (n + prior_n)


def stage_depth(stage: str) -> int:
    """Map a stage label to its ordinal depth (unknown stages -> 0)."""

    return STAGE_DEPTH.get(stage, STAGE_DEPTH.get(stage.lower(), 0))


def build_manager_features(
    records: pl.DataFrame,
    config: ManagerConfig | None = None,
) -> pl.DataFrame:
    """Per-manager pedigree features with Bayesian shrinkage.

    Args:
        records: Frame with columns ``[manager, team, tournament, stage_reached,
            knockout_matches, knockout_wins]`` plus an optional integer ``tournament_order``
            (larger = more recent) used for recency weighting; when absent, row order
            within each manager is used.
        config: Shrinkage and recency settings.

    Returns:
        Frame keyed by ``manager`` with ``[manager, team, knockout_win_rate,
        deepest_run_weighted, knockout_conversion_rate, tenure_matches]``. Rates are
        shrunk toward the global mean.
    """

    config = config or ManagerConfig()
    required = {
        "manager", "team", "tournament", "stage_reached", "knockout_matches", "knockout_wins",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"manager records missing columns: {sorted(missing)}")

    has_order = "tournament_order" in records.columns

    # Global prior: pooled knockout win rate across all managers.
    total_ko = int(records["knockout_matches"].sum())
    total_wins = int(records["knockout_wins"].sum())
    prior_rate = (total_wins / total_ko) if total_ko > 0 else 0.5

    rows: list[dict[str, object]] = []
    for (manager,), group in records.group_by(["manager"]):
        team = group["team"].to_list()[-1]
        ko_matches = int(group["knockout_matches"].sum())
        ko_wins = int(group["knockout_wins"].sum())
        observed = (ko_wins / ko_matches) if ko_matches > 0 else prior_rate
        shrunk_rate = shrink(observed, ko_matches, prior_rate, config.prior_n)

        ordered = (
            group.sort("tournament_order", descending=True)
            if has_order
            else group.reverse()
        )
        weighted_depth = 0.0
        for i, rec in enumerate(ordered.iter_rows(named=True)):
            weight = (1.0 - config.recency_lambda) ** i
            weighted_depth += weight * stage_depth(str(rec["stage_reached"]))

        rows.append(
            {
                "manager": manager,
                "team": team,
                "knockout_win_rate": shrunk_rate,
                "deepest_run_weighted": weighted_depth,
                "knockout_conversion_rate": shrink(
                    observed, ko_matches, prior_rate, config.prior_n
                ),
                "tenure_matches": ko_matches,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "manager": pl.Utf8,
            "team": pl.Utf8,
            "knockout_win_rate": pl.Float64,
            "deepest_run_weighted": pl.Float64,
            "knockout_conversion_rate": pl.Float64,
            "tenure_matches": pl.Int64,
        },
    )
