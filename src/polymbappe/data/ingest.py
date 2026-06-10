"""Data ingestion orchestration.

Wires source fetch -> normalize -> store for each upstream dataset. Each source is
ingested independently and failures are isolated: a missing optional source or a
network error logs a warning and is skipped rather than aborting the whole run.

Sources prefer a local raw file under ``data/raw`` when present (reproducible,
offline-friendly), falling back to a network fetch (or, for Elo, a self-computation from
the already-ingested ``matches`` table) otherwise.
"""

from __future__ import annotations

import io

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.data import sources
from polymbappe.data.normalize import normalize_kaggle_results, normalize_odds_frame
from polymbappe.data.store import read_table, table_exists, write_table
from polymbappe.data.tables import TABLE_COLUMNS, Table

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


def ingest_elo(settings: Settings | None = None) -> int:
    """Materialize the ``elo_snapshots`` table by self-computing Elo from ``matches``.

    Requires the ``matches`` table (run :func:`ingest_results` first). Walks results
    chronologically and records each team's post-match rating, producing the time series
    the dashboard's Elo-trajectory view reads. No network access.

    Returns the number of snapshot rows written.
    """

    from polymbappe.features.elo import build_elo_snapshots

    settings = settings or Settings()
    if not table_exists(Table.MATCHES, settings):
        raise FileNotFoundError("Elo ingestion needs the matches table; run ingest_results first.")

    matches = read_table(Table.MATCHES, settings)
    snapshots = build_elo_snapshots(matches)
    write_table(Table.ELO_SNAPSHOTS, snapshots, mode="overwrite", settings=settings)
    logger.info("ingest.elo.stored", rows=snapshots.height)
    return snapshots.height


def ingest_market_odds(
    settings: Settings | None = None, *, live: bool = False, source: str = "football-data"
) -> int:
    """Ingest bookmaker/market odds into the ``market_odds`` table.

    Reads a normalized decimal-odds CSV at ``data/raw/odds.csv`` with columns
    ``date, home_team, away_team, home_odds, draw_odds, away_odds`` (optional
    ``timestamp``). The per-match ``match_id`` is rebuilt with the same
    ``date__home__away`` convention as the matches table so odds join cleanly; team
    spellings must therefore match the results source. Overround is removed by
    :func:`~polymbappe.data.normalize.normalize_odds_frame`.

    Skips (returns 0) when the file is absent — market odds are optional for the MVM path.
    """

    settings = settings or Settings()
    local = settings.raw_data_dir / "odds.csv"
    if not local.exists():
        logger.info("ingest.odds.skip", reason="no data/raw/odds.csv")
        return 0

    raw = pl.read_csv(io.BytesIO(local.read_bytes()))
    required = {"date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"odds.csv missing columns: {sorted(missing)}")

    raw = raw.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("date")
    ).with_columns(
        pl.format("{}__{}__{}", pl.col("date"), pl.col("home_team"), pl.col("away_team"))
        .alias("match_id")
    )

    normalized = normalize_odds_frame(
        raw,
        source=source,
        home_col="home_odds",
        draw_col="draw_odds",
        away_col="away_odds",
        timestamp_col="timestamp" if "timestamp" in raw.columns else None,
    )
    write_table(
        Table.MARKET_ODDS, normalized, mode="append" if live else "overwrite", settings=settings
    )
    logger.info("ingest.odds.stored", rows=normalized.height, live=live)
    return normalized.height


def ingest_team_xg(settings: Settings | None = None) -> int:
    """Ingest team-level xG into the ``team_xg`` table from ``data/raw/team_xg.csv``.

    Expects columns ``team, date, xg, xga`` (FBref team-match xG, 2018+). Live FBref
    scraping via :func:`~polymbappe.data.sources.get_fbref_matches` is available but heavy
    and network-dependent, so the default path is the reproducible local file. Skips
    (returns 0) when the file is absent.
    """

    settings = settings or Settings()
    local = settings.raw_data_dir / "team_xg.csv"
    if not local.exists():
        logger.info("ingest.xg.skip", reason="no data/raw/team_xg.csv")
        return 0

    raw = pl.read_csv(io.BytesIO(local.read_bytes()))
    required = {"team", "date", "xg", "xga"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"team_xg.csv missing columns: {sorted(missing)}")

    normalized = raw.select(
        pl.col("team").cast(pl.Utf8),
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False),
        pl.col("xg").cast(pl.Float64),
        pl.col("xga").cast(pl.Float64),
    ).select(TABLE_COLUMNS[Table.TEAM_XG])
    write_table(Table.TEAM_XG, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.xg.stored", rows=normalized.height)
    return normalized.height


def ingest_all_sources(live: bool = False, settings: Settings | None = None) -> dict[str, int]:
    """Ingest all configured upstream datasets into normalized storage.

    Runs results first (the others depend on it or are optional overlays), then Elo
    (self-computed from results), market odds, and team xG. Each source is isolated:
    a failure or a skipped optional source is recorded rather than aborting the run.

    Args:
        live: Incremental mode — append latest results/odds rather than overwrite.
        settings: Optional settings override.

    Returns:
        Mapping of source name to rows ingested. Sources that failed are recorded as
        ``-1``; optional sources with no input return ``0``.
    """

    settings = settings or Settings()
    logger.info("ingest.start", live=live)
    report: dict[str, int] = {}

    # (name, callable, passes-live-flag). Order matters: Elo reads the matches table.
    steps: tuple[tuple[str, object, bool], ...] = (
        ("results", ingest_results, True),
        ("elo", ingest_elo, False),
        ("market_odds", ingest_market_odds, True),
        ("team_xg", ingest_team_xg, False),
    )
    for name, fn, takes_live in steps:
        try:
            report[name] = fn(settings, live=live) if takes_live else fn(settings)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 — isolate per-source failure
            logger.warning("ingest.source_failed", source=name, error=str(exc))
            report[name] = -1

    # Transfermarkt squad valuations and FBref live scraping remain manual/optional:
    # their fetchers exist in `sources` but require auth/heavy network and are off the
    # minimum-viable-model critical path.

    logger.info("ingest.done", report=report)
    return report
