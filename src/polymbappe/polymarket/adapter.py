"""Polymarket market price ingestion (gamma API).

The HTTP fetch is a thin wrapper; the JSON parsing is a pure, unit-tested function.
Polymarket data is used for live edge detection only (not backtesting — historical
order-book data is not available pre-2024), so this is off the MVM critical path.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import requests

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

_HEADERS = {"User-Agent": "polymbappe/0.1 (+https://github.com/)"}


def parse_market_outcomes(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ``(outcome, price)`` rows from one gamma-API market object.

    ``outcomes`` and ``outcomePrices`` arrive as JSON-encoded string arrays. Returns one
    row per outcome with the implied probability (gamma prices are already 0-1). Returns
    an empty list when the market is malformed or the two arrays disagree in length.
    """

    outcomes_raw = raw.get("outcomes")
    prices_raw = raw.get("outcomePrices")
    if outcomes_raw is None or prices_raw is None:
        return []

    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    if len(outcomes) != len(prices):
        return []

    market_id = str(raw.get("id", ""))
    question = str(raw.get("question", ""))
    rows: list[dict[str, Any]] = []
    for outcome, price in zip(outcomes, prices, strict=True):
        rows.append(
            {
                "market_id": market_id,
                "question": question,
                "outcome": str(outcome),
                "price": float(price),
            }
        )
    return rows


def fetch_polymarket_markets(
    *, query: str | None = None, limit: int = 100, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Fetch active, open market objects from the Polymarket gamma API."""

    params: dict[str, Any] = {"limit": limit, "active": "true", "closed": "false"}
    if query is not None:
        params["slug"] = query
    response = requests.get(GAMMA_MARKETS_URL, params=params, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    markets = payload.get("data", payload) if isinstance(payload, dict) else payload
    return list(markets)


def fetch_polymarket_prices(
    *, query: str | None = None, limit: int = 100, timeout: float = 30.0
) -> pl.DataFrame:
    """Fetch and normalize current Polymarket prices into a long (outcome-level) frame."""

    rows: list[dict[str, Any]] = []
    for market in fetch_polymarket_markets(query=query, limit=limit, timeout=timeout):
        rows.extend(parse_market_outcomes(market))
    return pl.DataFrame(
        rows,
        schema={
            "market_id": pl.Utf8,
            "question": pl.Utf8,
            "outcome": pl.Utf8,
            "price": pl.Float64,
        },
    )
