"""External data source adapters.

Thin network/IO wrappers. The brittle parse/normalize logic lives in
:mod:`polymbappe.data.normalize` (pure, unit-tested); functions here only fetch raw
bytes/HTML/dataframes so they stay correct-by-construction and free of business logic.
"""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl
import requests
import structlog
from bs4 import BeautifulSoup

from polymbappe.config import Settings

logger = structlog.get_logger(__name__)

#: Default raw CSV mirror of martj42/international_results (GitHub raw, no Kaggle auth).
KAGGLE_RESULTS_RAW_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

_DEFAULT_HEADERS = {"User-Agent": "polymbappe/0.1 (+https://github.com/)"}

#: Realistic desktop-browser headers for anti-bot-sensitive sources (Transfermarkt,
#: Wikipedia). The Phase B scrapers fetch through :func:`cached_get`, which sends these.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
}

#: Sub-directory of ``data/raw`` used for the shared on-disk HTTP request cache.
_HTTP_CACHE_DIRNAME = ".http_cache"

#: host -> monotonic timestamp of last request, for per-host self-throttling.
_LAST_REQUEST: dict[str, float] = {}


def _get(url: str, timeout: float) -> requests.Response:
    response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response


def http_cache_dir(settings: Settings | None = None) -> Path:
    """Resolve the shared on-disk HTTP cache directory (``data/raw/.http_cache``)."""

    settings = settings or Settings()
    return settings.raw_data_dir / _HTTP_CACHE_DIRNAME


def _cache_key(url: str) -> str:
    """Stable cache filename for ``url`` (full URL incl. query) via sha256."""

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{digest}.bin"


def _throttle(host: str, min_interval: float) -> None:
    """Block until at least ``min_interval`` seconds have elapsed since this host's
    last request. ``min_interval <= 0`` is a no-op (and never sleeps — test-safe)."""

    if min_interval <= 0:
        _LAST_REQUEST[host] = time.monotonic()
        return
    last = _LAST_REQUEST.get(host)
    now = time.monotonic()
    if last is not None:
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
    _LAST_REQUEST[host] = time.monotonic()


def cached_get(
    url: str,
    *,
    settings: Settings | None = None,
    timeout: float = 20.0,
    min_interval: float = 1.0,
    force_refresh: bool = False,
    _fetcher: Callable[..., requests.Response] = requests.get,
) -> bytes:
    """Browser-like GET with an on-disk cache + per-host self-throttle.

    Returns the response **body as bytes** (decode with ``.decode()`` for text).

    On a cache hit (a prior call for the same ``url``) the cached bytes are returned
    with **no network call and no throttle wait**. On a miss the request is fetched
    via ``_fetcher`` (raising for HTTP status), written to the cache, and returned.

    Parameters
    ----------
    settings:
        Resolves the cache dir (``settings.raw_data_dir / ".http_cache"``).
    min_interval:
        Minimum seconds between requests to the same host (default ``1.0``; pass a
        higher value such as ``2.5`` for Transfermarkt, or ``0`` in tests to skip the
        sleep). Only applied on cache misses.
    force_refresh:
        Bypass any cached entry and re-fetch.
    _fetcher:
        Injection point for the underlying fetch (defaults to ``requests.get``); a
        test can pass a stub to assert cache hits avoid re-fetching.
    """

    cache_dir = http_cache_dir(settings)
    cache_path = cache_dir / _cache_key(url)

    if not force_refresh and cache_path.exists():
        return cache_path.read_bytes()

    host = urlsplit(url).netloc
    _throttle(host, min_interval)

    response = _fetcher(url, headers=_BROWSER_HEADERS, timeout=timeout)
    response.raise_for_status()
    content = response.content

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def fetch_eloratings_html(url: str, timeout: float = 20.0) -> BeautifulSoup:
    """Fetch and parse an Elo ratings page."""

    return BeautifulSoup(_get(url, timeout).text, "html.parser")


def load_kaggle_results_csv(csv_bytes: bytes) -> pl.DataFrame:
    """Load Kaggle international results CSV bytes into a Polars DataFrame."""

    return pl.read_csv(io.BytesIO(csv_bytes), null_values=["NA"])


def fetch_results_csv(url: str = KAGGLE_RESULTS_RAW_URL, timeout: float = 60.0) -> pl.DataFrame:
    """Download the international results CSV and load it into Polars."""

    return load_kaggle_results_csv(_get(url, timeout).content)


def fetch_football_data_csv(url: str, timeout: float = 60.0) -> pl.DataFrame:
    """Download a Football-Data.co.uk CSV of bookmaker odds into Polars."""

    return pl.read_csv(io.BytesIO(_get(url, timeout).content), ignore_errors=True)


def fetch_kaggle_player_attributes(dataset: str, *, file: str | None = None) -> pl.DataFrame:
    """Download an EA FC / FM player-attributes CSV from a Kaggle dataset into Polars.

    Used for agent player-importance tiering only (not as model features) — see the
    unified spec ("Player attribute data strategy"). ``dataset`` is a Kaggle slug such as
    ``"stefanoleone992/ea-sports-fc-24-complete-player-dataset"``; ``file`` selects which
    CSV inside it (e.g. ``"male_players.csv"``), defaulting to the first ``*.csv`` found.

    Requires the optional ``kagglehub`` package and a Kaggle API token (``~/.kaggle/
    kaggle.json``). ``kagglehub`` is imported lazily so the dependency is needed only when
    this network path is actually taken; the local-CSV path in
    :func:`~polymbappe.data.ingest.ingest_player_attributes` never imports it. The raw
    columns are reconciled to the table schema by
    :func:`~polymbappe.data.normalize.normalize_player_attributes`.
    """

    import kagglehub  # lazy: optional, network/auth-bound dependency

    path = Path(kagglehub.dataset_download(dataset))
    if file is not None:
        csv_path = path / file
    else:
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"no CSV found in Kaggle dataset {dataset!r} at {path}")
        csv_path = csvs[0]
    return pl.read_csv(csv_path, infer_schema_length=10_000, ignore_errors=True)


#: Transfermarkt squad/kader page template. ``{slug}`` is the team's URL slug and
#: ``{tm_id}`` its numeric club id; appended ``saison_id`` selects the season. Layout is
#: anti-bot-sensitive, so fetches go through :func:`cached_get` (browser headers + cache).
TRANSFERMARKT_SQUAD_URL = (
    "https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}/saison_id/{saison_id}"
)


def fetch_transfermarkt_squad(
    tournament: str,
    team: str,
    *,
    settings: Settings | None = None,
    url: str | None = None,
    tm_id: str | None = None,
    saison_id: str = "",
    min_interval: float = 2.5,
    timeout: float = 20.0,
) -> list[dict[str, object]]:
    """Fetch one national team's Transfermarkt squad (kader) page → raw player rows.

    Returns a list of ``{"player", "club", "age", "team", "tournament"}`` dicts (one per
    called-up player). Market-value parsing is intentionally OUT OF SCOPE — cohesion only
    needs ``player``/``club``/``age``.

    The page is fetched through :func:`cached_get` (browser headers, on-disk cache,
    ``min_interval`` throttle defaulting to 2.5s for Transfermarkt). Brittle selectors are
    isolated: any parse/layout failure logs and returns ``[]`` rather than raising, so an
    upstream redesign degrades to "no rows" instead of breaking ingestion.

    Args:
        tournament: ``Tournament.name`` this snapshot belongs to (passed through onto rows).
        team: Source team name (canonicalization happens at ingest time).
        url: Explicit squad URL; overrides the ``slug``/``tm_id`` template.
        tm_id / saison_id: Components of the default :data:`TRANSFERMARKT_SQUAD_URL`.
        min_interval: Per-host throttle seconds (pass ``0`` in tests).
    """

    if url is None:
        if tm_id is None:
            logger.warning(
                "sources.transfermarkt.no_url", team=team, reason="no url or tm_id provided"
            )
            return []
        slug = team.lower().replace(" ", "-")
        url = TRANSFERMARKT_SQUAD_URL.format(slug=slug, tm_id=tm_id, saison_id=saison_id)

    try:
        html = cached_get(
            url, settings=settings, timeout=timeout, min_interval=min_interval
        ).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network failure isolated, degrade to empty
        logger.warning("sources.transfermarkt.fetch_failed", team=team, url=url, error=str(exc))
        return []

    try:
        return _parse_transfermarkt_squad(html, team=team, tournament=tournament)
    except Exception as exc:  # noqa: BLE001 - brittle selectors isolated
        logger.warning("sources.transfermarkt.parse_failed", team=team, error=str(exc))
        return []


def _parse_transfermarkt_squad(
    html: str, *, team: str, tournament: str
) -> list[dict[str, object]]:
    """Parse a Transfermarkt squad table into ``player``/``club``/``age`` rows.

    Selector-isolated helper for :func:`fetch_transfermarkt_squad`. The kader table rows
    (``table.items > tbody > tr``) carry the player name in the ``inline-table`` hauptlink
    cell, the club in the row's club-logo ``img`` alt/title, and the age inside a
    parenthesized ``(DD/MM/YYYY (age))`` birth-date cell.
    """

    import re

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        return []

    rows: list[dict[str, object]] = []
    for tr in table.select("tbody > tr"):
        name_cell = tr.select_one("td.hauptlink a") or tr.select_one(".inline-table a")
        if name_cell is None:
            continue
        player = name_cell.get_text(strip=True)
        if not player:
            continue

        club: str | None = None
        club_img = tr.select_one("td img.tiny_wappen") or tr.select_one(
            "td a img[class*='wappen']"
        )
        if club_img is not None:
            club = (club_img.get("alt") or club_img.get("title") or "").strip() or None

        age: float | None = None
        for td in tr.select("td.zentriert"):
            text = td.get_text(strip=True)
            match = re.search(r"\((\d{1,2})\)", text)
            if match:
                age = float(match.group(1))
                break

        rows.append(
            {
                "player": player,
                "club": club,
                "age": age,
                "team": team,
                "tournament": tournament,
            }
        )
    return rows


def fetch_transfermarkt_squad_valuation(
    tournament: str,
    team: str,
    *,
    settings: Settings | None = None,
    url: str | None = None,
    tm_id: str | None = None,
    saison_id: str = "",
    min_interval: float = 2.5,
    timeout: float = 20.0,
) -> list[dict[str, object]]:
    """Fetch one national team's Transfermarkt squad page → raw per-player market values.

    Returns a list of ``{"player", "market_value", "team", "tournament"}`` dicts (one per
    called-up player) where ``market_value`` is the player's Transfermarkt value in euros
    (``None`` when the page lists no value, e.g. ``"-"``). The caller aggregates these into
    the per-team ``squad_valuations`` table (total / median / count).

    This reads the **same** kader page as :func:`fetch_transfermarkt_squad` (which drops the
    value column because cohesion only needs player/club/age); both go through
    :func:`cached_get` so a single fetch serves both. Brittle selectors are isolated: any
    parse/layout failure logs and returns ``[]`` rather than raising, so an upstream redesign
    degrades to "no rows" instead of breaking ingestion.

    Args:
        tournament: ``Tournament.name`` this snapshot belongs to (passed through onto rows).
        team: Source team name (canonicalization happens at ingest time).
        url: Explicit squad URL; overrides the ``slug``/``tm_id`` template.
        tm_id / saison_id: Components of the default :data:`TRANSFERMARKT_SQUAD_URL`.
        min_interval: Per-host throttle seconds (pass ``0`` in tests).
    """

    if url is None:
        if tm_id is None:
            logger.warning(
                "sources.transfermarkt.no_url", team=team, reason="no url or tm_id provided"
            )
            return []
        slug = team.lower().replace(" ", "-")
        url = TRANSFERMARKT_SQUAD_URL.format(slug=slug, tm_id=tm_id, saison_id=saison_id)

    try:
        html = cached_get(
            url, settings=settings, timeout=timeout, min_interval=min_interval
        ).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network failure isolated, degrade to empty
        logger.warning("sources.transfermarkt.fetch_failed", team=team, url=url, error=str(exc))
        return []

    try:
        return _parse_transfermarkt_valuations(html, team=team, tournament=tournament)
    except Exception as exc:  # noqa: BLE001 - brittle selectors isolated
        logger.warning("sources.transfermarkt.parse_failed", team=team, error=str(exc))
        return []


def _parse_transfermarkt_valuations(
    html: str, *, team: str, tournament: str
) -> list[dict[str, object]]:
    """Parse a Transfermarkt squad table into ``player``/``market_value`` rows.

    Selector-isolated helper for :func:`fetch_transfermarkt_squad_valuation`. Each kader row
    (``table.items > tbody > tr``) carries the player name in the ``hauptlink`` cell and the
    market value in the trailing right-aligned ``td.rechts.hauptlink`` cell (e.g. ``€80.00m``).
    """

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        return []

    rows: list[dict[str, object]] = []
    for tr in table.select("tbody > tr"):
        name_cell = tr.select_one("td.hauptlink a") or tr.select_one(".inline-table a")
        if name_cell is None:
            continue
        player = name_cell.get_text(strip=True)
        if not player:
            continue

        value_cell = tr.select_one("td.rechts.hauptlink")
        market_value = (
            _parse_market_value(value_cell.get_text(strip=True))
            if value_cell is not None
            else None
        )

        rows.append(
            {
                "player": player,
                "market_value": market_value,
                "team": team,
                "tournament": tournament,
            }
        )
    return rows


def _parse_market_value(text: str) -> float | None:
    """Parse a Transfermarkt market-value string (``€80.00m``, ``€500k``, ``-``) into euros.

    Returns ``None`` for empty / placeholder (``"-"``) cells or anything unparseable, so a
    missing value never crashes aggregation. Handles the English-site suffixes ``k`` / ``m`` /
    ``bn``.
    """

    text = text.strip().replace("€", "").replace(",", "")
    if not text or text == "-":
        return None
    multiplier = 1.0
    if text.endswith("bn"):
        multiplier, text = 1_000_000_000.0, text[:-2]
    elif text.endswith("m"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text.endswith("k"):
        multiplier, text = 1_000.0, text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


#: Wikipedia "<tournament> squads" article per internal ``Tournament.name``. These pages
#: list every nation's call-up (player, club, date-of-birth/age) and are the **fallback**
#: squad source when Transfermarkt is unavailable (blocked / no ``tm_id``). Extend as new
#: tournaments are added; an entry can also be overridden per-team via the squads manifest.
WIKIPEDIA_SQUADS_PAGES: dict[str, str] = {
    "WC2010": "2010 FIFA World Cup squads",
    "WC2014": "2014 FIFA World Cup squads",
    "WC2018": "2018 FIFA World Cup squads",
    "WC2022": "2022 FIFA World Cup squads",
    "WC2026": "2026 FIFA World Cup squads",
    "EU2016": "UEFA Euro 2016 squads",
    "EU2020": "UEFA Euro 2020 squads",
    "EU2024": "UEFA Euro 2024 squads",
    "CA2016": "Copa América Centenario squads",
    "CA2019": "2019 Copa América squads",
    "CA2021": "2021 Copa América squads",
    "CA2024": "2024 Copa América squads",
}

#: MediaWiki API endpoint used to pull a manager's article wikitext for tenure parsing.
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_squad(
    tournament: str,
    team: str,
    *,
    settings: Settings | None = None,
    page: str | None = None,
    api_url: str = WIKIPEDIA_API_URL,
    min_interval: float = 1.0,
    timeout: float = 20.0,
) -> list[dict[str, object]]:
    """Fetch one team's call-up from the Wikipedia "<tournament> squads" page.

    Fallback squad source for :func:`fetch_transfermarkt_squad` — same output shape
    (``{"player", "club", "age", "team", "tournament"}`` rows). The rendered article HTML
    (MediaWiki ``action=parse``) is fetched through :func:`cached_get`, then the section
    whose heading matches ``team`` and its squad table are parsed. Any fetch/parse failure
    logs and returns ``[]`` (graceful degrade), so a layout change or a missing page degrades
    to "no rows" rather than raising.

    Args:
        tournament: ``Tournament.name`` (e.g. ``"WC2022"``); selects the page via
            :data:`WIKIPEDIA_SQUADS_PAGES` unless ``page`` is given.
        team: Section heading to match (English Wikipedia name; canonicalized at ingest).
        page: Explicit article title override (e.g. from the squads manifest).
        min_interval: Per-host throttle seconds (pass ``0`` in tests).
    """

    from urllib.parse import urlencode

    page = page or WIKIPEDIA_SQUADS_PAGES.get(tournament)
    if not page:
        logger.warning(
            "sources.wikipedia_squad.no_page", team=team, tournament=tournament,
            reason="no page override and tournament not in WIKIPEDIA_SQUADS_PAGES",
        )
        return []

    params = {
        "action": "parse",
        "page": page,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    url = f"{api_url}?{urlencode(params)}"
    try:
        body = cached_get(
            url, settings=settings, timeout=timeout, min_interval=min_interval
        ).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network isolated, degrade to empty
        logger.warning("sources.wikipedia_squad.fetch_failed", team=team, page=page, error=str(exc))
        return []

    try:
        import json

        html = json.loads(body).get("parse", {}).get("text", "")
        if isinstance(html, dict):  # formatversion=1 shape: {"*": "<html>"}
            html = html.get("*", "")
        return _parse_wikipedia_squad(str(html), team=team, tournament=tournament)
    except Exception as exc:  # noqa: BLE001 - brittle parse isolated
        logger.warning("sources.wikipedia_squad.parse_failed", team=team, page=page, error=str(exc))
        return []


def _norm_heading(text: str) -> str:
    """Lowercase + collapse whitespace, for loose squad-section heading matching."""

    return " ".join(text.split()).strip().lower()


def _cell_text(cells: list, idx: int | None) -> str:
    """Meaningful text of ``cells[idx]``, or ``""`` when out of range.

    Wikipedia squad cells often lead with a flag-icon anchor whose text is empty (e.g. the
    Club cell is ``<flag link><club link>``), so the LAST non-empty, non-reference anchor is
    taken; cells with no usable anchor fall back to their full stripped text.
    """

    if idx is None or idx >= len(cells):
        return ""
    cell = cells[idx]
    texts = [
        t
        for a in cell.find_all("a")
        if (t := a.get_text(strip=True)) and not t.startswith("[")
    ]
    return texts[-1] if texts else cell.get_text(" ", strip=True)


def _parse_wikipedia_squad(
    html: str, *, team: str, tournament: str
) -> list[dict[str, object]]:
    """Parse one team's squad table out of a Wikipedia "squads" page's rendered HTML.

    Selector-isolated helper for :func:`fetch_wikipedia_squad`. Finds the heading whose text
    matches ``team``, takes the following squad table, locates the ``Player`` / ``Club`` /
    date-of-birth columns from the header row, and reads each player row. ``age`` comes from
    the ``(aged NN)`` annotation in the date-of-birth cell (``None`` when absent).
    """

    import re

    soup = BeautifulSoup(html, "html.parser")

    target = _norm_heading(team)
    heading = next(
        (
            h
            for h in soup.find_all(["h2", "h3", "h4"])
            if _norm_heading(h.get_text()) == target
        ),
        None,
    )
    if heading is None:
        return []
    table = heading.find_next("table")
    if table is None:
        return []

    header_cells = table.find("tr")
    if header_cells is None:
        return []
    headers = [c.get_text(" ", strip=True).lower() for c in header_cells.find_all(["th", "td"])]

    def _col(*needles: str) -> int | None:
        for i, head in enumerate(headers):
            if any(n in head for n in needles):
                return i
        return None

    player_idx = _col("player")
    club_idx = _col("club")
    dob_idx = _col("date of birth", "birth")

    rows: list[dict[str, object]] = []
    for tr in header_cells.find_next_siblings("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) != len(headers):
            continue  # skip sub-rows / spanning rows that don't align to the header

        player = _cell_text(cells, player_idx)
        if not player:
            continue
        club = _cell_text(cells, club_idx) or None

        age: float | None = None
        if dob_idx is not None and dob_idx < len(cells):
            match = re.search(r"aged\s*(\d{1,2})", cells[dob_idx].get_text(" ", strip=True))
            if match:
                age = float(match.group(1))

        rows.append(
            {"player": player, "club": club, "age": age, "team": team, "tournament": tournament}
        )
    return rows


def fetch_wikipedia_manager_history(
    manager: str,
    *,
    settings: Settings | None = None,
    api_url: str = WIKIPEDIA_API_URL,
    min_interval: float = 1.0,
    timeout: float = 20.0,
) -> list[dict[str, object]]:
    """Fetch a manager's national-team tenure rows from Wikipedia (MediaWiki API).

    Returns raw ``{"manager", "team", "start_year", "end_year"}`` tenure rows. There is NO
    canonical knockout-record field on Wikipedia, so this fetch only yields the manager's
    national-team **tenure windows**; the derivation of
    ``knockout_matches``/``knockout_wins``/``stage_reached`` from those windows joined
    against the ingested ``matches`` table happens in the ingest layer
    (:func:`~polymbappe.data.ingest.derive_manager_records`), not here.

    The infobox "Managerial career → National team (start–end)" rows are parsed from the
    article wikitext. Any fetch/parse failure logs and returns ``[]`` (graceful degrade).
    """

    from urllib.parse import urlencode

    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": manager,
        "redirects": "1",
    }
    url = f"{api_url}?{urlencode(params)}"
    try:
        body = cached_get(
            url, settings=settings, timeout=timeout, min_interval=min_interval
        ).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network isolated
        logger.warning("sources.wikipedia.fetch_failed", manager=manager, error=str(exc))
        return []

    try:
        return _parse_wikipedia_manager_history(body, manager=manager)
    except Exception as exc:  # noqa: BLE001 - brittle parse isolated
        logger.warning("sources.wikipedia.parse_failed", manager=manager, error=str(exc))
        return []


def _parse_wikipedia_manager_history(
    body: str, *, manager: str
) -> list[dict[str, object]]:
    """Parse national-team manager tenure windows from a MediaWiki revisions JSON blob.

    Selector-isolated helper for :func:`fetch_wikipedia_manager_history`. Scans the infobox
    "Managerclubs"/"Manageryears" wikitext for national-team rows and extracts the
    ``start_year``/``end_year`` of each tenure (an open-ended tenure has ``end_year=None``).
    """

    import json
    import re

    payload = json.loads(body)
    pages = payload.get("query", {}).get("pages", {})
    wikitext = ""
    for page in pages.values():
        revisions = page.get("revisions") or []
        if revisions:
            slots = revisions[0].get("slots", {})
            wikitext = slots.get("main", {}).get("*", "") or revisions[0].get("*", "")
            break
    if not wikitext:
        return []

    rows: list[dict[str, object]] = []
    # Infobox managerial-career rows pair a years field with a club/team field, e.g.
    # | manageryears3 = 2018–2022 | managerclubs3 = [[England national football team|England]]
    # The club value must capture the WHOLE wikilink (its display name follows a ``|``), so it
    # runs to end-of-line rather than stopping at the first pipe.
    years = dict(re.findall(r"manageryears(\d*)\s*=\s*([^\n|]+)", wikitext))
    clubs = dict(re.findall(r"managerclubs(\d*)\s*=\s*([^\n]+)", wikitext))
    for key, raw_years in years.items():
        raw_team = clubs.get(key)
        if not raw_team:
            continue
        team = _clean_national_team(_wikilink_text(raw_team))
        span = re.search(r"(\d{4})\s*[–\-]\s*(\d{4})?", raw_years)
        if span is None:
            continue
        start_year = int(span.group(1))
        end_year = int(span.group(2)) if span.group(2) else None
        rows.append(
            {
                "manager": manager,
                "team": team,
                "start_year": start_year,
                "end_year": end_year,
            }
        )
    return rows


def _wikilink_text(raw: str) -> str:
    """Extract display text from a ``[[Target|Display]]`` (or ``[[Target]]``) wikilink."""

    import re

    cleaned = raw.strip()
    match = re.search(r"\[\[([^\]]+)\]\]", cleaned)
    if match:
        inner = match.group(1)
        return inner.split("|")[-1].strip() if "|" in inner else inner.strip()
    return cleaned


def _clean_national_team(name: str) -> str:
    """Reduce a Wikipedia team link to the bare nation, e.g. ``France national football
    team`` → ``France``, so it joins the ``matches`` table (canonicalized again at ingest).
    An unpiped wikilink yields the long article title; a piped one already gives the nation.
    """

    import re

    cleaned = re.sub(
        r"\s+national\s+(football|soccer|association football)?\s*team$",
        "",
        name.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def get_fbref_matches(
    leagues: str | list[str],
    seasons: str | int | list[str | int],
    stat_type: str = "schedule",
) -> pl.DataFrame:
    """Fetch FBref match-level data via the ``soccerdata`` package.

    Returns a Polars frame of the requested ``stat_type`` (default ``"schedule"``, which
    includes per-match xG where FBref provides it, i.e. 2018+). Team-level xG feature
    construction from this frame is handled downstream in the feature layer.
    """

    import soccerdata as sd  # local import: heavy, network-backed, optional at import time

    fbref = sd.FBref(leagues=leagues, seasons=seasons)
    if stat_type == "schedule":
        pandas_df = fbref.read_schedule()
    else:
        pandas_df = fbref.read_team_match_stats(stat_type=stat_type)
    return pl.from_pandas(pandas_df.reset_index())
