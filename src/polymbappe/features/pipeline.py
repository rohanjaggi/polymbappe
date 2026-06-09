"""Feature pipeline orchestration.

Joins the individual team-date feature builders into the final training matrix,
enforcing no leakage via ``as_of_date``.
"""

from __future__ import annotations

from datetime import date

import structlog

logger = structlog.get_logger(__name__)


def build_feature_matrix(as_of: date | None = None, contextual: bool = False) -> None:
    """Build the feature matrix as of a given date.

    Args:
        as_of: Only use data available strictly before this date (leakage guard).
        contextual: When ``True``, build the contextual feature table instead of
            the Tier 1-3 core matrix.
    """

    logger.info("features.start", as_of=as_of, contextual=contextual)
    raise NotImplementedError("Implement feature builder orchestration and join.")
