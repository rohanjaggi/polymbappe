"""External data source adapters.

Thin network/IO wrappers. The brittle parse/normalize logic lives in
:mod:`polymbappe.data.normalize` (pure, unit-tested); functions here only fetch raw
bytes/HTML/dataframes so they stay correct-by-construction and free of business logic.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import polars as pl
import requests
from bs4 import BeautifulSoup

from polymbappe.config import Settings

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
