"""Data ingestion orchestration.

Wires source fetch -> normalize -> store for each upstream dataset. Each source is
ingested independently and failures are isolated: a missing optional source or a
network error logs a warning and is skipped rather than aborting the whole run.

Sources prefer a local raw file under ``data/raw`` when present (reproducible,
offline-friendly), falling back to a network fetch otherwise.
"""

from __future__ import annotations

import io

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.data import sources
from polymbappe.data.normalize import normalize_kaggle_results
from polymbappe.data.store import write_table
from polymbappe.data.tables import Table

logger = structlog.get_logger(__name__)


def ingest_results(
    settings: Settings | None = None, *, live: bool = False, url: str | None = None
) -> int:
    """Ingest international match results into the ``matches`` table.

    Prefers ``data/raw/results.csv`` if present, else downloads the public mirror.
    ``live=True`` appends/dedupes onto the existing table; otherwise it overwrites.

    Returns the number of normalized match rows written.
    """

    settings = settings or Settings()
    local = settings.raw_data_dir / "results.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()), null_values=["NA"])
        logger.info("ingest.results.local", path=str(local), rows=raw.height)
    else:
        raw = sources.fetch_results_csv(url or sources.KAGGLE_RESULTS_RAW_URL)
        logger.info("ingest.results.fetched", rows=raw.height)

    normalized = normalize_kaggle_results(raw)
    write_table(
        Table.MATCHES, normalized, mode="append" if live else "overwrite", settings=settings
    )
    logger.info("ingest.results.stored", rows=normalized.height, live=live)
    return normalized.height


def ingest_all_sources(live: bool = False, settings: Settings | None = None) -> dict[str, int]:
    """Ingest all configured upstream datasets into normalized storage.

    Args:
        live: Incremental mode — append latest results/odds rather than overwrite.
        settings: Optional settings override.

    Returns:
        Mapping of source name to rows ingested. Sources that were skipped or failed
        are recorded with a count of ``-1``.
    """

    settings = settings or Settings()
    logger.info("ingest.start", live=live)
    report: dict[str, int] = {}

    for name, fn in (("results", ingest_results),):
        try:
            report[name] = fn(settings, live=live)
        except Exception as exc:  # noqa: BLE001 — isolate per-source failure
            logger.warning("ingest.source_failed", source=name, error=str(exc))
            report[name] = -1

    # Elo, market odds, Transfermarkt squad value, and FBref xG ingestion are wired in
    # later phases (they are not on the minimum-viable-model critical path). Their
    # fetchers/normalizers already exist in `sources`, `normalize`, and `polymarket`.

    logger.info("ingest.done", report=report)
    return report
