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
from datetime import date

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.data import sources
from polymbappe.data.aliases import normalize_team_expr
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


def ingest_elo(
    settings: Settings | None = None,
    *,
    url: str | None = None,
    as_of: date | None = None,
) -> int:
    """Materialize the ``elo_snapshots`` table, preferring published EloRatings.net ratings.

    Resolution order (first that yields data wins):

    1. **Published ratings** — a local ``data/raw/elo.html`` page, else an opt-in network
       fetch (see :func:`_published_elo`). Parsed via
       :func:`~polymbappe.data.normalize.parse_eloratings`, team names canonicalized, and
       stamped with ``as_of`` (today by default).
    2. **Self-computed** — walk the ``matches`` table chronologically, recording each
       team's post-match rating. Requires the ``matches`` table (run :func:`ingest_results`
       first) and no network access.

    The dashboard reads only the latest rating per team, so either a single published
    snapshot or the self-computed time series serves it. ``url`` forces a specific fetch
    source; ``as_of`` overrides the published snapshot date (useful for reproducibility).

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
    """Opt-in network source for published Elo: the URL in ``data/raw/elo_url.txt``.

    Returns the first non-comment line as the fetch URL, or
    :data:`~polymbappe.data.sources.ELORATINGS_WORLD_URL` if the file exists but names none
    (so merely creating the file enables the default EloRatings.net page). An absent file
    returns ``None`` — keeping network fetches opt-in so the default path stays offline,
    reproducible, and test-safe.
    """

    url_file = settings.raw_data_dir / "elo_url.txt"
    if not url_file.exists():
        return None
    for line in url_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return sources.ELORATINGS_WORLD_URL


def _published_elo(
    settings: Settings, *, url: str | None, as_of: date | None
) -> pl.DataFrame | None:
    """Published EloRatings.net snapshot from a local raw HTML file or a configured URL.

    Prefers ``data/raw/elo.html`` (reproducible, offline). Otherwise fetches ``url`` if
    given, else the opt-in URL from :func:`_elo_url`. The parsed ratings are stamped with
    ``as_of`` (today by default), team names canonicalized via :func:`normalize_team_expr`,
    deduped per ``(team, date)``, and returned in the ``elo_snapshots`` schema.

    Returns ``None`` when no published source is configured, a fetch fails, or the page
    yields no rows (EloRatings.net populates its table via JavaScript, so a plain scrape can
    come back empty) — the caller then self-computes from ``matches``.
    """

    from polymbappe.data.normalize import parse_eloratings

    local = settings.raw_data_dir / "elo.html"
    if local.exists():
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(local.read_bytes(), "html.parser")
        logger.info("ingest.elo.local", path=str(local))
    else:
        url = url or _elo_url(settings)
        if url is None:
            return None
        try:
            soup = sources.fetch_eloratings_html(url)
        except Exception as exc:  # noqa: BLE001 - network isolated; fall back to self-compute
            logger.warning("ingest.elo.fetch_failed", url=url, error=str(exc))
            return None
        logger.info("ingest.elo.fetched", url=url)

    parsed = parse_eloratings(soup, as_of=as_of or date.today())
    if parsed.is_empty():
        logger.warning(
            "ingest.elo.published_empty",
            hint="EloRatings.net table may be JS-populated; self-computing instead",
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
    (self-computed from results), market odds, team xG, squads (Transfermarkt → cohesion
    inputs), and manager records (Wikipedia tenure + match-join derivation → manager
    pedigree). Each source is isolated: a failure or a skipped optional source is recorded
    rather than aborting the run.

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
        ("squads", ingest_squads, False),
        ("squad_valuations", ingest_squad_valuations, False),
        ("manager_records", ingest_manager_records, False),
    )
    for name, fn, takes_live in steps:
        try:
            report[name] = fn(settings, live=live) if takes_live else fn(settings)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 — isolate per-source failure
            logger.warning("ingest.source_failed", source=name, error=str(exc))
            report[name] = -1

    # FBref *live* xG scraping remains manual/optional: its fetcher exists in `sources` but is
    # heavy/network-dependent and off the minimum-viable-model critical path (the default
    # team_xg path reads a local CSV). Every source above — results, odds, squads, squad
    # valuations, manager records — prefers a local CSV and falls back to a cached/manifest
    # scraper, so they are safe to run by default: an absent local file + empty scraper
    # records `0` rather than failing.

    logger.info("ingest.done", report=report)
    return report
