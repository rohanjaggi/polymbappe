"""External data source adapters."""

from __future__ import annotations

import io

import polars as pl
import requests
from bs4 import BeautifulSoup


def fetch_eloratings_html(url: str, timeout: float = 20.0) -> BeautifulSoup:
    """Fetch and parse an Elo ratings page."""

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def load_kaggle_results_csv(csv_bytes: bytes) -> pl.DataFrame:
    """Load Kaggle international results CSV bytes into a Polars DataFrame."""

    return pl.read_csv(io.BytesIO(csv_bytes))


def get_fbref_matches() -> pl.DataFrame:
    """Fetch FBref data via soccerdata (stub)."""

    raise NotImplementedError("Integrate soccerdata FBref ingestion.")
