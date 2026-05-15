"""Data ingestion orchestration."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def ingest_all_sources() -> None:
    """Ingest all configured upstream datasets into raw storage."""

    logger.info("ingest.start")
    raise NotImplementedError("Implement external data pulls and normalization.")
