"""Prediction report generation.

Assembles tournament probability outputs and edge reports into the artifacts
under ``data/outputs/``.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def generate_report(tournament: int = 2026) -> None:
    """Generate the tournament prediction report.

    Args:
        tournament: Tournament year to report on.
    """

    logger.info("report.start", tournament=tournament)
    raise NotImplementedError("Implement report assembly from simulation outputs.")
