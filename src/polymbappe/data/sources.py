"""External data source adapters.

Thin network/IO wrappers. The brittle parse/normalize logic lives in
:mod:`polymbappe.data.normalize` (pure, unit-tested); functions here only fetch raw
bytes/HTML/dataframes so they stay correct-by-construction and free of business logic.
"""

from __future__ import annotations

import io

import polars as pl
import requests
from bs4 import BeautifulSoup

#: Default raw CSV mirror of martj42/international_results (GitHub raw, no Kaggle auth).
KAGGLE_RESULTS_RAW_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

_DEFAULT_HEADERS = {"User-Agent": "polymbappe/0.1 (+https://github.com/)"}


def _get(url: str, timeout: float) -> requests.Response:
    response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response


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
