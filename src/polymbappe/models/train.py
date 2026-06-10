"""Model training orchestration.

Fits the base models, the stacked meta-learner, and (optionally) the contextual
adjuster, persisting fitted artifacts for simulation and backtesting.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def train_models(model: str | None = None) -> None:
    """Fit forecasting models.

    Args:
        model: Optional single model to fit (e.g. ``"bayesian"``). When ``None``,
            fit the full stack: base models, meta-learner, contextual adjuster.
    """

    logger.info("train.start", model=model)
    raise NotImplementedError("Implement base-model + meta-learner training orchestration.")
