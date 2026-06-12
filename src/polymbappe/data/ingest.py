"""Data ingestion orchestration.

Wires source fetch -> normalize -> store for each upstream dataset. Each source is
ingested independently and failures are isolated: a missing optional source or a
network error logs a warning and is skipped rather than aborting the whole run.

Sources prefer a local raw file under ``data/raw`` when present (reproducible,
offline-friendly), falling back to a network fetch otherwise. Elo prefers published
EloRatings.net ratings (local ``elo.html`` or an opt-in fetch) and self-computes from the
already-ingested ``matches`` table only when no published source is available.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from datetime import date

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.data import sources
from polymbappe.data.aliases import normalize_team_expr
from polymbappe.data.normalize import (
    normalize_geonames_cities,
    normalize_kaggle_results,
    normalize_odds_frame,
    normalize_openfootball_schedule,
    normalize_openfootball_stadiums,
    normalize_player_attributes,
    statsbomb_team_match_ppda,
    statsbomb_team_match_xg,
)
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


def ingest_elo(
    settings: Settings | None = None,
    *,
    url: str | None = None,
    as_of: date | None = None,
) -> int:
    """Materialize the ``elo_snapshots`` table, preferring published EloRatings.net ratings.

    Resolution order (first that yields data wins):

    1. **Published ratings** — local raw TSVs (``elo_world.tsv`` + ``elo_teams.tsv``), else a
       local ``data/raw/elo.html`` page, else an opt-in network fetch of EloRatings.net's
       ``World.tsv`` + ``en.teams.tsv`` (see :func:`_published_elo`). TSVs are joined on the
       2-letter team code and parsed via
       :func:`~polymbappe.data.normalize.parse_eloratings_tsv` (HTML via
       :func:`~polymbappe.data.normalize.parse_eloratings`); team names are canonicalized and
       rows stamped with ``as_of`` (today by default).
    2. **Self-computed** — walk the ``matches`` table chronologically, recording each
       team's post-match rating. Requires the ``matches`` table (run :func:`ingest_results`
       first) and no network access.

    The dashboard reads only the latest rating per team, so either a single published
    snapshot or the self-computed time series serves it. ``url`` forces a specific
    ``World.tsv`` fetch URL; ``as_of`` overrides the published snapshot date (useful for
    reproducibility).

    Returns the number of snapshot rows written.
    """

    settings = settings or Settings()

    published = _published_elo(settings, url=url, as_of=as_of)
    if published is not None:
        write_table(Table.ELO_SNAPSHOTS, published, mode="overwrite", settings=settings)
        logger.info("ingest.elo.published", rows=published.height)
        return published.height

    from polymbappe.features.elo import build_elo_snapshots

    if not table_exists(Table.MATCHES, settings):
        raise FileNotFoundError("Elo ingestion needs the matches table; run ingest_results first.")

    matches = read_table(Table.MATCHES, settings)
    snapshots = build_elo_snapshots(matches)
    write_table(Table.ELO_SNAPSHOTS, snapshots, mode="overwrite", settings=settings)
    logger.info("ingest.elo.self_computed", rows=snapshots.height)
    return snapshots.height


def _elo_url(settings: Settings) -> str | None:
    """Opt-in network source for published Elo: the ``World.tsv`` URL in ``data/raw/elo_url.txt``.

    Returns the first non-comment line as the ``World.tsv`` fetch URL, or
    :data:`~polymbappe.data.sources.ELORATINGS_WORLD_TSV_URL` if the file exists but names
    none (so merely creating the file enables the default EloRatings.net feed). An absent
    file returns ``None`` — keeping network fetches opt-in so the default path stays offline,
    reproducible, and test-safe.
    """

    url_file = settings.raw_data_dir / "elo_url.txt"
    if not url_file.exists():
        return None
    for line in url_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return sources.ELORATINGS_WORLD_TSV_URL


def _published_elo(
    settings: Settings, *, url: str | None, as_of: date | None
) -> pl.DataFrame | None:
    """Published EloRatings.net snapshot from local raw files or an opt-in network fetch.

    Resolution order (first that yields rows wins); every path stamps rows with ``as_of``
    (today by default), canonicalizes team names via :func:`normalize_team_expr`, dedupes per
    ``(team, date)``, and returns the ``elo_snapshots`` schema:

    1. **Local TSV** — ``data/raw/elo_world.tsv`` + ``data/raw/elo_teams.tsv`` (the site's
       ``World.tsv`` / ``en.teams.tsv``), parsed via :func:`parse_eloratings_tsv`.
    2. **Local HTML** — ``data/raw/elo.html`` (any saved/rendered ranking table), parsed via
       :func:`parse_eloratings`.
    3. **Network TSV (opt-in)** — fetches ``World.tsv`` + ``en.teams.tsv`` when ``url`` is
       given or ``data/raw/elo_url.txt`` exists (see :func:`_elo_url`). This is the live
       published source: the ranking page itself is a JS shell that serves its ratings from
       these TSVs (keyed by 2-letter team code), so the HTML parser only ever sees an empty
       table.

    Returns ``None`` when no published source is configured, a fetch fails, or the parse
    yields no rows — the caller then self-computes from ``matches``.
    """

    from polymbappe.data.normalize import parse_eloratings, parse_eloratings_tsv

    as_of = as_of or date.today()
    local_world = settings.raw_data_dir / "elo_world.tsv"
    local_teams = settings.raw_data_dir / "elo_teams.tsv"
    local_html = settings.raw_data_dir / "elo.html"

    if local_world.exists() and local_teams.exists():
        parsed = parse_eloratings_tsv(
            local_world.read_text(), local_teams.read_text(), as_of=as_of
        )
        logger.info("ingest.elo.local_tsv", path=str(local_world))
    elif local_html.exists():
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(local_html.read_bytes(), "html.parser")
        parsed = parse_eloratings(soup, as_of=as_of)
        logger.info("ingest.elo.local", path=str(local_html))
    else:
        url = url or _elo_url(settings)
        if url is None:
            return None
        try:
            world_tsv, teams_tsv = sources.fetch_eloratings_tsv(world_url=url)
        except Exception as exc:  # noqa: BLE001 - network isolated; fall back to self-compute
            logger.warning("ingest.elo.fetch_failed", url=url, error=str(exc))
            return None
        parsed = parse_eloratings_tsv(world_tsv, teams_tsv, as_of=as_of)
        logger.info("ingest.elo.fetched", url=url)

    if parsed.is_empty():
        logger.warning(
            "ingest.elo.published_empty",
            hint="EloRatings.net returned no parseable ratings; self-computing instead",
        )
        return None

    return (
        parsed.with_columns(normalize_team_expr("team").alias("team"))
        .unique(subset=["team", "date"], keep="first")
        .select(TABLE_COLUMNS[Table.ELO_SNAPSHOTS])
    )


_MARKET_ODDS_COLUMNS = TABLE_COLUMNS[Table.MARKET_ODDS]


def _local_odds(settings: Settings) -> pl.DataFrame | None:
    """Normalized decimal-odds CSV at ``data/raw/odds.csv`` (manual / fallback source)."""

    local = settings.raw_data_dir / "odds.csv"
    if not local.exists():
        return None
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
    return normalize_odds_frame(
        raw, source="local-csv", home_col="home_odds", draw_col="draw_odds",
        away_col="away_odds", timestamp_col="timestamp" if "timestamp" in raw.columns else None,
    )


def _footballdata_odds(settings: Settings) -> pl.DataFrame | None:
    """Football-Data.co.uk bookmaker odds from local CSVs and/or configured URLs.

    Local CSVs: ``data/raw/football_data/*.csv``. Remote: one URL per line in
    ``data/raw/football_data_urls.txt``. Each is parsed by
    :func:`~polymbappe.data.normalize.normalize_footballdata_odds`; network failures are
    isolated per URL.
    """

    from polymbappe.data.normalize import normalize_footballdata_odds

    frames: list[pl.DataFrame] = []
    local_dir = settings.raw_data_dir / "football_data"
    if local_dir.is_dir():
        for csv_path in sorted(local_dir.glob("*.csv")):
            raw = pl.read_csv(io.BytesIO(csv_path.read_bytes()), ignore_errors=True)
            frames.append(normalize_footballdata_odds(raw))
    url_file = settings.raw_data_dir / "football_data_urls.txt"
    if url_file.exists():
        for url in url_file.read_text().splitlines():
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            try:
                raw = sources.fetch_football_data_csv(url)
                frames.append(normalize_footballdata_odds(raw))
            except Exception as exc:  # noqa: BLE001 - isolate per-URL fetch/parse failure
                logger.warning("ingest.odds.footballdata_url_failed", url=url, error=str(exc))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed")


def _polymarket_odds(settings: Settings) -> pl.DataFrame | None:
    """Polymarket three-way match odds, aligned to the 2026 fixtures in match_predictions.

    Requires ``data/outputs/match_predictions.parquet`` (run ``simulate`` once) to orient
    each market to a fixture's home/away and key it as ``2026__home__away``. The query slug
    is read from ``data/raw/polymarket_query.txt`` when present. Network/auth failures are
    isolated and return ``None``.
    """

    from polymbappe.polymarket import adapter

    preds_path = settings.outputs_data_dir / "match_predictions.parquet"
    if not preds_path.exists():
        logger.info("ingest.odds.polymarket_skip", reason="no match_predictions fixtures")
        return None
    fixtures = pl.read_parquet(preds_path).select("match_id", "home_team", "away_team")
    query_file = settings.raw_data_dir / "polymarket_query.txt"
    query = query_file.read_text().strip() if query_file.exists() else None
    try:
        long_prices = adapter.fetch_polymarket_prices(query=query)
    except Exception as exc:  # noqa: BLE001 - network/auth isolated
        logger.warning("ingest.odds.polymarket_failed", error=str(exc))
        return None
    three_way = adapter.normalize_polymarket_three_way(long_prices)
    unmatched = adapter.unmatched_market_teams(three_way, fixtures)
    if unmatched:
        logger.warning(
            "ingest.odds.polymarket_unmatched",
            teams=unmatched,
            hint="add these spellings to configs/team_aliases.yaml so their odds join",
        )
    aligned = adapter.align_polymarket_to_fixtures(three_way, fixtures)
    return aligned if aligned.height > 0 else None


def ingest_market_odds(settings: Settings | None = None, *, live: bool = False) -> int:
    """Ingest market odds into ``market_odds`` from all available sources.

    Gathers from (1) a local normalized ``odds.csv``, (2) Football-Data.co.uk CSVs/URLs,
    and (3) Polymarket (aligned to the 2026 fixtures). Each source is isolated; sources
    with no input are skipped. Frames are concatenated and de-duplicated on
    ``(match_id, source)``. Match ids follow ``date__home__away`` (Football-Data / local)
    or ``2026__home__away`` (Polymarket) so odds join the matches / predictions tables.

    Returns the number of odds rows written (0 if no source produced any).
    """

    settings = settings or Settings()
    gathered: list[pl.DataFrame] = []
    for name, fetch in (
        ("local", _local_odds),
        ("football-data", _footballdata_odds),
        ("polymarket", _polymarket_odds),
    ):
        try:
            frame = fetch(settings)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failure
            logger.warning("ingest.odds.source_failed", source=name, error=str(exc))
            frame = None
        if frame is not None and frame.height > 0:
            gathered.append(frame.select(_MARKET_ODDS_COLUMNS))
            logger.info("ingest.odds.source", source=name, rows=frame.height)

    if not gathered:
        logger.info("ingest.odds.skip", reason="no odds source produced data")
        return 0

    combined = pl.concat(gathered, how="vertical_relaxed").unique(subset=["match_id", "source"])
    write_table(
        Table.MARKET_ODDS, combined, mode="append" if live else "overwrite", settings=settings
    )
    logger.info("ingest.odds.stored", rows=combined.height, live=live)
    return combined.height


def _statsbomb_team_match_rows(
    settings: Settings,
    row_builder: Callable[..., list[dict[str, object]]],
    *,
    competitions: tuple[tuple[int, int], ...] | None = None,
) -> list[dict[str, object]]:
    """Walk StatsBomb open data → per-team-match rows via ``row_builder``.

    For each configured ``(competition, season)`` fetches the match list, then each match's
    event stream, and applies ``row_builder(events, home_team=, away_team=, match_date=)``.
    Network/parse failures are isolated per competition and per match (logged, skipped). The
    on-disk HTTP cache means a second pass (e.g. PPDA after xG) re-reads events from disk with
    no network.
    """

    competitions = competitions or sources.STATSBOMB_COMPETITIONS
    rows: list[dict[str, object]] = []
    for competition_id, season_id in competitions:
        try:
            matches = sources.fetch_statsbomb_matches(
                competition_id, season_id, settings=settings
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-competition fetch failure
            logger.warning(
                "ingest.statsbomb.matches_failed",
                competition=competition_id,
                season=season_id,
                error=str(exc),
            )
            continue
        for match in matches:
            match_id = match.get("match_id")
            home = match.get("home_team", {}).get("home_team_name")
            away = match.get("away_team", {}).get("away_team_name")
            match_date = match.get("match_date")
            if not (match_id and home and away and match_date):
                continue
            try:
                events = sources.fetch_statsbomb_events(match_id, settings=settings)
            except Exception as exc:  # noqa: BLE001 — isolate per-match fetch failure
                logger.warning("ingest.statsbomb.events_failed", match_id=match_id, error=str(exc))
                continue
            rows.extend(
                row_builder(events, home_team=home, away_team=away, match_date=match_date)
            )
    return rows


def ingest_team_xg(settings: Settings | None = None, *, live: bool = False) -> int:
    """Ingest team-level xG into the ``team_xg`` table (``[team, date, xg, xga]``).

    Prefers a reproducible local ``data/raw/team_xg.csv`` (columns ``team, date, xg, xga``).
    When that file is absent and ``live=True``, derives real xG from StatsBomb Open Data —
    summing ``shot.statsbomb_xg`` per team across each covered international match (World Cup
    2018/2022, Euro 2020/2024, Copa América 2024; the public xG that FBref re-publishes). The
    fetch is pinned to a commit for reproducibility but heavy (~260 event files), so offline
    runs (no ``live``) skip cleanly. Note: StatsBomb open data is released *after* tournaments,
    so this is the historical/backtest source, not a live 2026 feed.
    """

    settings = settings or Settings()
    local = settings.raw_data_dir / "team_xg.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        required = {"team", "date", "xg", "xga"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"team_xg.csv missing columns: {sorted(missing)}")
        normalized = raw.select(
            normalize_team_expr("team").alias("team"),
            pl.col("date").cast(pl.Utf8).str.to_date(strict=False),
            pl.col("xg").cast(pl.Float64),
            pl.col("xga").cast(pl.Float64),
        ).select(TABLE_COLUMNS[Table.TEAM_XG])
        write_table(Table.TEAM_XG, normalized, mode="overwrite", settings=settings)
        logger.info("ingest.xg.stored", rows=normalized.height, source="local")
        return normalized.height

    if not live:
        logger.info(
            "ingest.xg.skip",
            reason="no data/raw/team_xg.csv (pass live=True to pull StatsBomb open data)",
        )
        return 0

    rows = _statsbomb_team_match_rows(settings, statsbomb_team_match_xg)
    if not rows:
        logger.info("ingest.xg.skip", reason="StatsBomb open data produced no xG rows")
        return 0

    frame = (
        pl.DataFrame(
            rows,
            schema={"team": pl.Utf8, "date": pl.Utf8, "xg": pl.Float64, "xga": pl.Float64},
        )
        .with_columns(
            normalize_team_expr("team").alias("team"),
            pl.col("date").str.to_date(strict=False),
        )
        .unique(subset=["team", "date"], keep="first")
        .select(TABLE_COLUMNS[Table.TEAM_XG])
    )
    write_table(Table.TEAM_XG, frame, mode="overwrite", settings=settings)
    logger.info("ingest.xg.stored", rows=frame.height, source="statsbomb")
    return frame.height


def ingest_ppda(settings: Settings | None = None, *, live: bool = False) -> int:
    """Ingest team-level PPDA into the ``team_ppda`` table (``[team, date, ppda]``).

    PPDA — passes allowed per defensive action; lower = a more aggressive high press. Prefers
    a reproducible local ``data/raw/team_ppda.csv`` (columns ``team, date, ppda``). When that
    file is absent and ``live=True``, computes true zonal PPDA from StatsBomb Open Data event
    streams (see :func:`~polymbappe.data.normalize.statsbomb_team_match_ppda`) over the same
    covered tournaments as :func:`ingest_team_xg`. Team names are canonicalized so they join
    the ``matches`` table; populating this table lights up the otherwise-null ``ppda_diff``
    contextual feature. Offline runs (no ``live``) skip cleanly.
    """

    settings = settings or Settings()
    local = settings.raw_data_dir / "team_ppda.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        required = {"team", "date", "ppda"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"team_ppda.csv missing columns: {sorted(missing)}")
        normalized = raw.select(
            normalize_team_expr("team").alias("team"),
            pl.col("date").cast(pl.Utf8).str.to_date(strict=False),
            pl.col("ppda").cast(pl.Float64),
        ).select(TABLE_COLUMNS[Table.TEAM_PPDA])
        write_table(Table.TEAM_PPDA, normalized, mode="overwrite", settings=settings)
        logger.info("ingest.ppda.stored", rows=normalized.height, source="local")
        return normalized.height

    if not live:
        logger.info(
            "ingest.ppda.skip",
            reason="no data/raw/team_ppda.csv (pass live=True to pull StatsBomb open data)",
        )
        return 0

    rows = _statsbomb_team_match_rows(settings, statsbomb_team_match_ppda)
    if not rows:
        logger.info("ingest.ppda.skip", reason="StatsBomb open data produced no PPDA rows")
        return 0

    frame = (
        pl.DataFrame(rows, schema={"team": pl.Utf8, "date": pl.Utf8, "ppda": pl.Float64})
        .with_columns(
            normalize_team_expr("team").alias("team"),
            pl.col("date").str.to_date(strict=False),
        )
        .unique(subset=["team", "date"], keep="first")
        .select(TABLE_COLUMNS[Table.TEAM_PPDA])
    )
    write_table(Table.TEAM_PPDA, frame, mode="overwrite", settings=settings)
    logger.info("ingest.ppda.stored", rows=frame.height, source="statsbomb")
    return frame.height


def ingest_venues(settings: Settings | None = None) -> int:
    """Ingest tournament venue coordinates into the ``venues`` table.

    Prefers ``data/raw/venues.csv`` (reproducible, offline) whose columns equal
    ``TABLE_COLUMNS[Table.VENUES]`` (``venue, city, country, latitude, longitude``); else
    fetches the openfootball 2026 stadiums feed
    (:func:`~polymbappe.data.sources.fetch_openfootball_stadiums`) and parses each venue's
    coordinates. ``city`` is the openfootball host-city string kept verbatim so it joins the
    ``schedule`` table's ``city`` and the travel-feature coordinate lookup
    (:func:`~polymbappe.context.fatigue.coord_lookup_from_venues`). This replaces the static
    16-city coordinate table that the travel feature previously hard-coded.

    Skips (returns 0) when the local file is absent AND the feed yields nothing.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.VENUES])
    local = settings.raw_data_dir / "venues.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"venues.csv missing columns: {sorted(missing)}")
        normalized = raw.select(
            pl.col("venue").cast(pl.Utf8),
            pl.col("city").cast(pl.Utf8),
            pl.col("country").cast(pl.Utf8),
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
        ).select(TABLE_COLUMNS[Table.VENUES])
        logger.info("ingest.venues.local", path=str(local), rows=normalized.height)
    else:
        stadiums = sources.fetch_openfootball_stadiums(settings=settings)
        normalized = normalize_openfootball_stadiums(stadiums)
        if normalized.is_empty():
            logger.info(
                "ingest.venues.skip", reason="no data/raw/venues.csv and openfootball empty"
            )
            return 0
        logger.info("ingest.venues.fetched", rows=normalized.height)

    write_table(Table.VENUES, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.venues.stored", rows=normalized.height)
    return normalized.height


def ingest_city_coords(
    settings: Settings | None = None, *, dataset: str = "cities15000"
) -> int:
    """Ingest the GeoNames city gazetteer into the ``city_coords`` table.

    Prefers ``data/raw/city_coords.csv`` (reproducible, offline) whose columns equal
    ``TABLE_COLUMNS[Table.CITY_COORDS]``; else downloads the GeoNames ``{dataset}`` dump
    (:func:`~polymbappe.data.sources.fetch_geonames_cities`) and normalizes it to one
    lower-cased ``(city, country)`` alias row per coordinate. This is the geocoding backbone
    that resolves each historical match's ``city`` to coordinates for the travel-distance
    backfill (:func:`~polymbappe.context.fatigue.build_city_coord_lookup`).

    Skips (returns 0) when the local file is absent AND the download yields nothing.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.CITY_COORDS])
    local = settings.raw_data_dir / "city_coords.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"city_coords.csv missing columns: {sorted(missing)}")
        normalized = raw.select(
            pl.col("city").cast(pl.Utf8).str.to_lowercase(),
            pl.col("country").cast(pl.Utf8),
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
            pl.col("population").cast(pl.Int64),
        ).select(TABLE_COLUMNS[Table.CITY_COORDS])
        logger.info("ingest.city_coords.local", path=str(local), rows=normalized.height)
    else:
        raw = sources.fetch_geonames_cities(dataset, settings=settings)
        normalized = normalize_geonames_cities(raw)
        if normalized.is_empty():
            logger.info(
                "ingest.city_coords.skip",
                reason="no data/raw/city_coords.csv and GeoNames empty",
            )
            return 0
        logger.info("ingest.city_coords.fetched", rows=normalized.height, dataset=dataset)

    write_table(Table.CITY_COORDS, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.city_coords.stored", rows=normalized.height)
    return normalized.height


def ingest_schedule(settings: Settings | None = None) -> int:
    """Ingest the tournament match schedule into the ``schedule`` table.

    Prefers ``data/raw/schedule.csv`` (reproducible, offline) whose columns equal
    ``TABLE_COLUMNS[Table.SCHEDULE]``; else fetches the openfootball 2026 fixtures feed
    (:func:`~polymbappe.data.sources.fetch_openfootball_schedule`) and normalizes it via
    :func:`~polymbappe.data.normalize.normalize_openfootball_schedule`. Either way ``team``
    columns are canonicalized via :func:`normalize_team_expr` and ``match_id`` is rebuilt on
    the shared ``date__home__away`` convention so group-stage fixtures join the matches / Elo
    tables; ``city`` joins the ``venues`` table. Feeds the travel-distance feature
    (:func:`~polymbappe.context.fatigue.schedule_to_appearances`).

    Skips (returns 0) when the local file is absent AND the feed yields nothing.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.SCHEDULE])
    local = settings.raw_data_dir / "schedule.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"schedule.csv missing columns: {sorted(missing)}")
        normalized = raw.with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("date"),
            pl.col("stage").cast(pl.Utf8),
            pl.col("group").cast(pl.Utf8),
            normalize_team_expr("home_team").alias("home_team"),
            normalize_team_expr("away_team").alias("away_team"),
            pl.col("city").cast(pl.Utf8),
        ).with_columns(
            pl.format("{}__{}__{}", pl.col("date"), pl.col("home_team"), pl.col("away_team"))
            .alias("match_id")
        ).select(TABLE_COLUMNS[Table.SCHEDULE])
        logger.info("ingest.schedule.local", path=str(local), rows=normalized.height)
    else:
        matches = sources.fetch_openfootball_schedule(settings=settings)
        normalized = normalize_openfootball_schedule(matches)
        if normalized.is_empty():
            logger.info(
                "ingest.schedule.skip",
                reason="no data/raw/schedule.csv and openfootball empty",
            )
            return 0
        logger.info("ingest.schedule.fetched", rows=normalized.height)

    write_table(Table.SCHEDULE, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.schedule.stored", rows=normalized.height)
    return normalized.height


def ingest_squads(settings: Settings | None = None) -> int:
    """Ingest per-player squad call-ups into the ``squads`` table (cohesion inputs).

    Prefers ``data/raw/squads.csv`` (reproducible, offline); else scrapes each manifest team
    Transfermarkt-first (:func:`~polymbappe.data.sources.fetch_transfermarkt_squad`) with a
    Wikipedia fallback (:func:`~polymbappe.data.sources.fetch_wikipedia_squad`) when
    Transfermarkt is unavailable. Either way expects/produces columns
    ``team, tournament, player, club, age``. ``team`` is
    canonicalized via :func:`normalize_team_expr`; ``club`` is trimmed; ``tournament`` must
    already equal a ``Tournament.name`` (e.g. ``"WC2018"``) and is passed through as-is;
    ``age`` is cast to ``Float64``. Market value is NOT ingested here (cohesion needs only
    player/club/age — see :func:`ingest_squad_valuations` for the ``squad_valuations`` table).

    Skips (returns 0) when the local file is absent AND the scraper yields nothing.
    """

    settings = settings or Settings()
    required = {"team", "tournament", "player", "club", "age"}
    local = settings.raw_data_dir / "squads.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        logger.info("ingest.squads.local", path=str(local), rows=raw.height)
    else:
        rows = _scrape_squads(settings)
        if not rows:
            logger.info("ingest.squads.skip", reason="no data/raw/squads.csv and scraper empty")
            return 0
        raw = pl.DataFrame(rows)
        logger.info("ingest.squads.scraped", rows=raw.height)

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"squads source missing columns: {sorted(missing)}")

    normalized = raw.select(
        normalize_team_expr("team").alias("team"),
        pl.col("tournament").cast(pl.Utf8),
        pl.col("player").cast(pl.Utf8),
        pl.col("club").cast(pl.Utf8).str.strip_chars(),
        pl.col("age").cast(pl.Float64),
    ).select(TABLE_COLUMNS[Table.SQUADS])
    write_table(Table.SQUADS, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.squads.stored", rows=normalized.height)
    return normalized.height


def _scrape_squads(settings: Settings) -> list[dict[str, object]]:
    """Scrape each ``(tournament, team, ...)`` in the manifest, Transfermarkt-first.

    Reads optional ``data/raw/squads_manifest.csv`` (columns ``tournament, team`` plus
    optional ``tm_id, saison_id, url, wiki_page``) and fetches each team's squad,
    Transfermarkt-first with a **Wikipedia fallback** (see :func:`_scrape_one_squad`).
    Returns the concatenated raw rows (``[]`` when no manifest or nothing fetched). Keeping
    the manifest out of code lets the local-CSV path stay the default and tests stay offline.
    """

    manifest_path = settings.raw_data_dir / "squads_manifest.csv"
    if not manifest_path.exists():
        return []
    manifest = pl.read_csv(io.BytesIO(manifest_path.read_bytes()))
    rows: list[dict[str, object]] = []
    for entry in manifest.iter_rows(named=True):
        rows.extend(_scrape_one_squad(entry, settings))
    return rows


def _scrape_one_squad(
    entry: dict[str, object], settings: Settings
) -> list[dict[str, object]]:
    """Fetch one team's squad: Transfermarkt first, Wikipedia fallback if it yields nothing.

    Transfermarkt is the primary source (richer, per-club). When it is unavailable — blocked,
    no ``tm_id``/``url`` in the manifest entry, or a layout change makes it return no rows —
    the Wikipedia "<tournament> squads" page is scraped instead
    (:func:`~polymbappe.data.sources.fetch_wikipedia_squad`). Both fetchers already isolate
    their own failures and return ``[]``, so this only chooses between them.
    """

    tournament = str(entry["tournament"])
    team = str(entry["team"])

    tm_kwargs: dict[str, object] = {"settings": settings}
    for key in ("tm_id", "saison_id", "url"):
        if key in entry and entry[key] is not None:
            tm_kwargs[key] = entry[key]
    rows = sources.fetch_transfermarkt_squad(tournament, team, **tm_kwargs)
    if rows:
        logger.info("ingest.squads.source", team=team, source="transfermarkt", rows=len(rows))
        return rows

    wiki_page = entry.get("wiki_page") if "wiki_page" in entry else None
    rows = sources.fetch_wikipedia_squad(
        tournament, team, settings=settings,
        page=str(wiki_page) if wiki_page is not None else None,
    )
    logger.info(
        "ingest.squads.source", team=team,
        source="wikipedia" if rows else "none", rows=len(rows),
    )
    return rows


def ingest_squad_valuations(settings: Settings | None = None) -> int:
    """Ingest per-team Transfermarkt squad valuations into the ``squad_valuations`` table.

    Prefers ``data/raw/squad_valuations.csv`` (reproducible, offline) whose columns equal
    ``TABLE_COLUMNS[Table.SQUAD_VALUATIONS]`` (``team, tournament, total_value, median_value,
    player_count``); else scrapes each manifest team's Transfermarkt squad page
    (:func:`~polymbappe.data.sources.fetch_transfermarkt_squad_valuation`) and aggregates the
    per-player market values into one row per ``(team, tournament)``. ``team`` is canonicalized
    via :func:`normalize_team_expr`; ``tournament`` must already equal a ``Tournament.name``
    (e.g. ``"WC2018"``) and is passed through as-is. Powers the squad-value features
    (:func:`~polymbappe.features.squad.build_squad_features`).

    Skips (returns 0) when the local file is absent AND the scraper yields nothing.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.SQUAD_VALUATIONS])
    local = settings.raw_data_dir / "squad_valuations.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        logger.info("ingest.squad_valuations.local", path=str(local), rows=raw.height)
    else:
        rows = _scrape_squad_valuations(settings)
        if not rows:
            logger.info(
                "ingest.squad_valuations.skip",
                reason="no data/raw/squad_valuations.csv and scraper empty",
            )
            return 0
        raw = pl.DataFrame(rows)
        logger.info("ingest.squad_valuations.scraped", rows=raw.height)

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"squad_valuations source missing columns: {sorted(missing)}")

    normalized = raw.select(
        normalize_team_expr("team").alias("team"),
        pl.col("tournament").cast(pl.Utf8),
        pl.col("total_value").cast(pl.Float64),
        pl.col("median_value").cast(pl.Float64),
        pl.col("player_count").cast(pl.Int64),
    ).select(TABLE_COLUMNS[Table.SQUAD_VALUATIONS])
    write_table(Table.SQUAD_VALUATIONS, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.squad_valuations.stored", rows=normalized.height)
    return normalized.height


def _scrape_squad_valuations(settings: Settings) -> list[dict[str, object]]:
    """Scrape Transfermarkt market values for each manifest team, aggregated per team.

    Reuses the squads manifest (``data/raw/squads_manifest.csv``: columns ``tournament,
    team`` plus optional ``tm_id, saison_id, url``) — the same Transfermarkt pages that feed
    the per-player ``squads`` scrape. Returns one aggregate row per ``(team, tournament)``
    (``[]`` when no manifest or nothing fetched). Unlike squads there is no Wikipedia fallback,
    since Wikipedia squad pages carry no market values; a team Transfermarkt can't serve is
    simply absent. Keeping the manifest out of code lets the local-CSV path stay the default
    and tests stay offline.
    """

    manifest_path = settings.raw_data_dir / "squads_manifest.csv"
    if not manifest_path.exists():
        return []
    manifest = pl.read_csv(io.BytesIO(manifest_path.read_bytes()))
    player_rows: list[dict[str, object]] = []
    for entry in manifest.iter_rows(named=True):
        tournament = str(entry["tournament"])
        team = str(entry["team"])
        tm_kwargs: dict[str, object] = {"settings": settings}
        for key in ("tm_id", "saison_id", "url"):
            if key in entry and entry[key] is not None:
                tm_kwargs[key] = entry[key]
        rows = sources.fetch_transfermarkt_squad_valuation(tournament, team, **tm_kwargs)
        logger.info(
            "ingest.squad_valuations.source", team=team,
            source="transfermarkt" if rows else "none", rows=len(rows),
        )
        player_rows.extend(rows)
    return _aggregate_squad_valuations(player_rows)


def _aggregate_squad_valuations(
    player_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-player ``market_value`` rows into per-``(team, tournament)`` valuations.

    Pure, network-free, unit-testable. ``total_value`` / ``median_value`` are taken over the
    non-null market values (a player with no listed value still counts toward
    ``player_count`` but not the totals); a group with no valued players yields ``0.0``.
    """

    if not player_rows:
        return []
    frame = pl.DataFrame(player_rows)
    out: list[dict[str, object]] = []
    for (team, tournament), group in frame.group_by(["team", "tournament"]):
        values = group["market_value"].drop_nulls()
        out.append(
            {
                "team": team,
                "tournament": tournament,
                "total_value": float(values.sum()) if values.len() > 0 else 0.0,
                "median_value": float(values.median()) if values.len() > 0 else 0.0,
                "player_count": int(group.height),
            }
        )
    return out


def ingest_player_attributes(settings: Settings | None = None) -> int:
    """Ingest EA FC / FM player attributes into the ``player_attributes`` table.

    Prefers ``data/raw/player_attributes.csv`` (reproducible, offline) whose columns equal
    ``TABLE_COLUMNS[Table.PLAYER_ATTRIBUTES]`` (``team, player, overall``); else fetches the
    Kaggle dataset named in ``data/raw/player_attributes_kaggle.txt`` (first line: the Kaggle
    slug, optional second line ``file=<name>.csv``) via
    :func:`~polymbappe.data.sources.fetch_kaggle_player_attributes` and reconciles its columns
    with :func:`~polymbappe.data.normalize.normalize_player_attributes`. ``team`` is the
    player's national team, canonicalized via :func:`normalize_team_expr` so it joins the squad
    tables; ``overall`` is cast to ``Int64``. Powers the agent's player-importance tiers
    (:func:`~polymbappe.features.players.build_player_tiers`), not the prediction model
    (unified spec, "Player attribute data strategy").

    Skips (returns 0) when the local file is absent AND no Kaggle config / fetch yields rows.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.PLAYER_ATTRIBUTES])
    local = settings.raw_data_dir / "player_attributes.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        logger.info("ingest.player_attributes.local", path=str(local), rows=raw.height)
    else:
        fetched = _fetch_player_attributes(settings)
        if fetched is None or fetched.height == 0:
            logger.info(
                "ingest.player_attributes.skip",
                reason="no data/raw/player_attributes.csv and Kaggle fetch empty",
            )
            return 0
        raw = fetched
        logger.info("ingest.player_attributes.fetched", rows=raw.height)

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"player_attributes source missing columns: {sorted(missing)}")

    normalized = raw.select(
        normalize_team_expr("team").alias("team"),
        pl.col("player").cast(pl.Utf8),
        pl.col("overall").cast(pl.Int64),
    ).select(TABLE_COLUMNS[Table.PLAYER_ATTRIBUTES])
    write_table(Table.PLAYER_ATTRIBUTES, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.player_attributes.stored", rows=normalized.height)
    return normalized.height


def _fetch_player_attributes(settings: Settings) -> pl.DataFrame | None:
    """Fetch + reconcile EA FC / FM attributes from the Kaggle dataset in the config file.

    Reads optional ``data/raw/player_attributes_kaggle.txt`` (first non-comment line: the
    Kaggle dataset slug; optional ``file=<name>.csv`` line selects the CSV inside it). Returns
    ``None`` when the config is absent — so the local-CSV path stays the default and tests stay
    offline — or when the fetch/reconcile yields nothing. The heavy, auth-bound ``kagglehub``
    import lives in the source fetcher and is only triggered on this path.
    """

    config_path = settings.raw_data_dir / "player_attributes_kaggle.txt"
    if not config_path.exists():
        return None
    dataset: str | None = None
    file: str | None = None
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("file="):
            file = line[len("file=") :].strip()
        elif dataset is None:
            dataset = line
    if dataset is None:
        return None

    raw = sources.fetch_kaggle_player_attributes(dataset, file=file)
    reconciled = normalize_player_attributes(raw)
    logger.info("ingest.player_attributes.source", dataset=dataset, rows=reconciled.height)
    return reconciled if reconciled.height > 0 else None


def derive_manager_records(
    tenure_rows: list[dict[str, object]] | pl.DataFrame,
    matches: pl.DataFrame,
    tournaments=None,
) -> pl.DataFrame:
    """Derive ``manager_records`` rows from manager tenure windows + the matches table.

    Pure, network-free, unit-testable. For each tenure row (``manager, team, start_year``,
    optional ``end_year``) and each tournament whose window the tenure covers, count the
    manager's team's knockout matches/wins in that tournament from ``matches`` and label the
    deepest knockout stage reached.

    Args:
        tenure_rows: Manager tenure windows (from the Wikipedia scraper).
        matches: The ingested ``matches`` table (``home_team, away_team, home_goals,
            away_goals, competition, is_knockout, date``).
        tournaments: Iterable of backtest ``Tournament`` objects (defaults to
            ``DEFAULT_TOURNAMENTS``) providing each tournament's name + date window. Their
            order in the sequence supplies ``tournament_order``.

    Returns:
        Frame with ``TABLE_COLUMNS[Table.MANAGER_RECORDS]`` columns. ``stage_reached`` uses
        the manager builder's stage vocabulary (group/R16/QF/SF/final/winner).
    """

    from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS

    tours = list(tournaments) if tournaments is not None else list(DEFAULT_TOURNAMENTS)
    order_by_name = {t.name: i for i, t in enumerate(tours)}

    if isinstance(tenure_rows, pl.DataFrame):
        tenures = tenure_rows.to_dicts()
    else:
        tenures = list(tenure_rows)

    if matches.height == 0:
        return _empty_manager_records()

    matches = matches.with_columns(pl.col("date").cast(pl.Date, strict=False))

    out: list[dict[str, object]] = []
    for tenure in tenures:
        manager = str(tenure["manager"])
        team = str(tenure["team"])
        start_year = int(tenure["start_year"])
        end_year = tenure.get("end_year")
        end_year = int(end_year) if end_year is not None else 9999
        for tour in tours:
            if not (start_year <= tour.start.year <= end_year):
                continue
            window = matches.filter(
                (pl.col("date") >= tour.start) & (pl.col("date") <= tour.end)
            )
            team_ko = window.filter(
                (pl.col("is_knockout"))
                & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
            )
            ko_matches = team_ko.height
            if ko_matches == 0 and window.filter(
                (pl.col("home_team") == team) | (pl.col("away_team") == team)
            ).is_empty():
                continue  # team did not play this tournament under this manager
            ko_wins = 0
            for m in team_ko.iter_rows(named=True):
                home = m["home_team"] == team
                gf = m["home_goals"] if home else m["away_goals"]
                ga = m["away_goals"] if home else m["home_goals"]
                if gf is not None and ga is not None and gf > ga:
                    ko_wins += 1
            stage = _deepest_stage(team_ko, team)
            out.append(
                {
                    "manager": manager,
                    "team": team,
                    "tournament": tour.name,
                    "stage_reached": stage,
                    "knockout_matches": ko_matches,
                    "knockout_wins": ko_wins,
                    "tournament_order": order_by_name.get(tour.name, 0),
                }
            )

    if not out:
        return _empty_manager_records()
    return pl.DataFrame(out).select(TABLE_COLUMNS[Table.MANAGER_RECORDS])


#: Number of knockout matches a team plays to reach a given depth, mapped to the manager
#: builder's stage vocabulary. A team with N knockout matches reached at least this stage.
_KO_COUNT_TO_STAGE: dict[int, str] = {0: "group", 1: "R16", 2: "QF", 3: "SF", 4: "final"}


def _deepest_stage(team_ko: pl.DataFrame, team: str) -> str:
    """Label the deepest knockout stage a team reached from its knockout-match count + result.

    Uses the manager builder's stage vocabulary (``STAGE_DEPTH`` keys). A team that won its
    final knockout match (and played a full 4-round bracket) is labelled ``winner``.
    """

    ko_count = team_ko.height
    if ko_count == 0:
        return "group"
    base = _KO_COUNT_TO_STAGE.get(ko_count, "final")
    if ko_count >= 4:
        last = team_ko.sort("date").row(-1, named=True)
        home = last["home_team"] == team
        gf = last["home_goals"] if home else last["away_goals"]
        ga = last["away_goals"] if home else last["home_goals"]
        if gf is not None and ga is not None and gf > ga:
            return "winner"
    return base


def _empty_manager_records() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "manager": pl.Utf8,
            "team": pl.Utf8,
            "tournament": pl.Utf8,
            "stage_reached": pl.Utf8,
            "knockout_matches": pl.Int64,
            "knockout_wins": pl.Int64,
            "tournament_order": pl.Int64,
        }
    ).select(TABLE_COLUMNS[Table.MANAGER_RECORDS])


def ingest_manager_records(settings: Settings | None = None) -> int:
    """Ingest manager tournament pedigree into the ``manager_records`` table.

    Prefers ``data/raw/manager_records.csv`` (reproducible, offline); else runs the
    Wikipedia tenure scraper (:func:`~polymbappe.data.sources.fetch_wikipedia_manager_history`)
    and derives knockout stats from the ingested ``matches`` table via
    :func:`derive_manager_records`. Output columns =
    ``TABLE_COLUMNS[Table.MANAGER_RECORDS]``: ``team`` is canonicalized via
    :func:`normalize_team_expr`; int counts (``knockout_matches``, ``knockout_wins``,
    ``tournament_order``) are cast; ``stage_reached`` must be from the manager builder's
    stage vocabulary; ``tournament`` must equal a ``Tournament.name``.

    Skips (returns 0) when the local file is absent AND the scraper yields nothing.
    """

    settings = settings or Settings()
    required = set(TABLE_COLUMNS[Table.MANAGER_RECORDS])
    local = settings.raw_data_dir / "manager_records.csv"
    if local.exists():
        raw = pl.read_csv(io.BytesIO(local.read_bytes()))
        logger.info("ingest.manager_records.local", path=str(local), rows=raw.height)
    else:
        raw = _scrape_manager_records(settings)
        if raw is None or raw.height == 0:
            logger.info(
                "ingest.manager_records.skip",
                reason="no data/raw/manager_records.csv and scraper empty",
            )
            return 0
        logger.info("ingest.manager_records.scraped", rows=raw.height)

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"manager_records source missing columns: {sorted(missing)}")

    normalized = raw.select(
        pl.col("manager").cast(pl.Utf8),
        normalize_team_expr("team").alias("team"),
        pl.col("tournament").cast(pl.Utf8),
        pl.col("stage_reached").cast(pl.Utf8),
        pl.col("knockout_matches").cast(pl.Int64),
        pl.col("knockout_wins").cast(pl.Int64),
        pl.col("tournament_order").cast(pl.Int64),
    ).select(TABLE_COLUMNS[Table.MANAGER_RECORDS])
    write_table(Table.MANAGER_RECORDS, normalized, mode="overwrite", settings=settings)
    logger.info("ingest.manager_records.stored", rows=normalized.height)
    return normalized.height


def _scrape_manager_records(settings: Settings) -> pl.DataFrame | None:
    """Scrape manager tenure windows + derive records against the ``matches`` table.

    Reads optional ``data/raw/managers.csv`` (column ``manager`` per row), fetches each
    manager's Wikipedia tenure history, and calls :func:`derive_manager_records` against the
    ingested ``matches`` table. Returns ``None`` when no manifest, no matches table, or
    nothing derived — so the local-CSV path stays the default and tests stay offline.
    """

    manifest_path = settings.raw_data_dir / "managers.csv"
    if not manifest_path.exists() or not table_exists(Table.MATCHES, settings):
        return None
    manifest = pl.read_csv(io.BytesIO(manifest_path.read_bytes()))
    tenure_rows: list[dict[str, object]] = []
    for entry in manifest.iter_rows(named=True):
        tenure_rows.extend(
            sources.fetch_wikipedia_manager_history(str(entry["manager"]), settings=settings)
        )
    if not tenure_rows:
        return None
    matches = read_table(Table.MATCHES, settings)
    derived = derive_manager_records(tenure_rows, matches)
    return derived if derived.height > 0 else None


def ingest_all_sources(live: bool = False, settings: Settings | None = None) -> dict[str, int]:
    """Ingest all configured upstream datasets into normalized storage.

    Runs results first (the others depend on it or are optional overlays), then Elo
    (self-computed from results), market odds, team xG and team PPDA (local CSV, or StatsBomb
    Open Data when ``live``), squads (Transfermarkt → cohesion inputs), and manager records
    (Wikipedia tenure + match-join derivation → manager pedigree). Each source is isolated: a
    failure or a skipped optional source is recorded rather than aborting the run.

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
        ("team_xg", ingest_team_xg, True),
        ("team_ppda", ingest_ppda, True),
        ("venues", ingest_venues, False),
        ("schedule", ingest_schedule, False),
        ("city_coords", ingest_city_coords, False),
        ("squads", ingest_squads, False),
        ("squad_valuations", ingest_squad_valuations, False),
        ("player_attributes", ingest_player_attributes, False),
        ("manager_records", ingest_manager_records, False),
    )
    for name, fn, takes_live in steps:
        try:
            report[name] = fn(settings, live=live) if takes_live else fn(settings)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 — isolate per-source failure
            logger.warning("ingest.source_failed", source=name, error=str(exc))
            report[name] = -1

    # team_xg/team_ppda prefer a local CSV and, under `live`, derive real xG/PPDA from
    # StatsBomb Open Data (event-level, the source FBref re-publishes; soccerdata exposes
    # neither for the international comps). That pull is heavy (~260 event files) so it only
    # runs in live mode; offline runs without a local CSV record `0`. StatsBomb open data is
    # historical (released after a tournament) — the live 2026 feed is a separate TODO (see
    # `sources.fetch_statsbomb_events` and the live-xg note beneath it).
    #
    # venues/schedule/city_coords prefer a local CSV and fall back to a cached fetch
    # (openfootball for venues/schedule; GeoNames for the city gazetteer), so they are safe to
    # run by default: an absent local file + empty fetch records `0` rather than failing.

    logger.info("ingest.done", report=report)
    return report
