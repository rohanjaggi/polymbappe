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

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
GAMMA_MARKETS_URL = f"{GAMMA_BASE_URL}/markets"
GAMMA_EVENTS_URL = f"{GAMMA_BASE_URL}/events"
CLOB_PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"

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


def _gamma_data(payload: Any) -> list[dict[str, Any]]:
    """Unwrap a gamma-API JSON payload to its list of objects (handles ``{"data": [...]}``)."""

    objects = payload.get("data", payload) if isinstance(payload, dict) else payload
    return list(objects)


def fetch_polymarket_events(
    tag_slug: str, *, limit: int = 100, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Page through active, open events grouped under a gamma **tag** slug.

    The gamma ``/events`` endpoint caps each page at ``limit`` rows, so this walks
    ``offset`` until a short page is returned.
    """

    events: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "tag_slug": tag_slug,
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        response = requests.get(GAMMA_EVENTS_URL, params=params, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
        page = _gamma_data(response.json())
        events.extend(page)
        if len(page) < limit:
            return events
        offset += limit


def fetch_polymarket_markets(
    *, query: str | None = None, limit: int = 100, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Fetch active, open market objects from the Polymarket gamma API.

    ``query`` is a gamma **event tag** slug (e.g. ``world-cup``), *not* an individual
    market slug. Per-match three-way (home/draw/away) markets are listed as sub-markets
    of the events grouped under a tournament's tag, so they are reached via ``/events``
    and flattened out of each event's ``markets`` list. Passing a tag/event slug to
    ``/markets?slug=`` — which filters by an individual *market* slug — silently returns
    nothing; that mismatch was the original bug. With ``query=None`` the bare ``/markets``
    listing is returned unchanged.
    """

    if query is None:
        params: dict[str, Any] = {"limit": limit, "active": "true", "closed": "false"}
        response = requests.get(GAMMA_MARKETS_URL, params=params, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
        return _gamma_data(response.json())

    markets: list[dict[str, Any]] = []
    for event in fetch_polymarket_events(query, limit=limit, timeout=timeout):
        markets.extend(event.get("markets") or [])
    return markets


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


#: Live 2026 World Cup *futures* markets (gamma event slugs) -> the simulation output
#: column they price. Per-match H/D/A markets do not exist until fixtures are scheduled;
#: these futures are what is tradeable pre-tournament. ``normalize`` de-vigs mutually
#: exclusive markets (one champion / one group winner) but not the reach-stage markets
#: (a team can clear several stages, so their Yes prices are independent).
WORLD_CUP_FUTURES: dict[str, dict[str, Any]] = {
    "world-cup-winner": {"output": "stage_probabilities", "column": "champion", "normalize": True},
    "world-cup-nation-to-reach-final": {
        "output": "stage_probabilities", "column": "FINAL", "normalize": False
    },
    "world-cup-nation-to-reach-semifinals": {
        "output": "stage_probabilities", "column": "SF", "normalize": False
    },
    "world-cup-nation-to-reach-quarterfinals": {
        "output": "stage_probabilities", "column": "QF", "normalize": False
    },
    "world-cup-nation-to-reach-round-of-16": {
        "output": "stage_probabilities", "column": "R16", "normalize": False
    },
    "world-cup-team-to-advance-to-knockout-stages": {
        "output": "stage_probabilities", "column": "R32", "normalize": False
    },
}

#: Outcome labels that are placeholders rather than real teams.
_NON_TEAM_LABELS = {"other", "team am", "field", "any other team"}


def fetch_polymarket_event(slug: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch one gamma-API event (with its per-team sub-markets) by slug."""

    response = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"slug": slug},
        headers=_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    events = payload if isinstance(payload, list) else payload.get("data", [payload])
    if not events:
        raise ValueError(f"No Polymarket event for slug {slug!r}.")
    return dict(events[0])


def parse_team_yes_prices(event: dict[str, Any], *, normalize: bool = False) -> pl.DataFrame:
    """Per-team implied probabilities from a grouped Yes/No futures event.

    Each sub-market is a team's Yes/No market (``groupItemTitle`` = team, prices the
    [Yes, No] pair). Returns ``[team, market_prob]`` with team names canonicalized via the
    alias table. Placeholder outcomes (``Other``/``Field``) and price-less markets are
    dropped. ``normalize=True`` rescales to sum 1 (de-vig) for mutually-exclusive markets.
    """

    from polymbappe.data.aliases import normalize_team_name

    rows: list[dict[str, Any]] = []
    for market in event.get("markets", []):
        team_raw = market.get("groupItemTitle") or ""
        outcomes_raw = market.get("outcomes")
        prices_raw = market.get("outcomePrices")
        if not team_raw or outcomes_raw is None or prices_raw is None:
            continue
        if team_raw.strip().lower() in _NON_TEAM_LABELS:
            continue
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        lowered = [str(o).strip().lower() for o in outcomes]
        if "yes" not in lowered or len(prices) != len(outcomes):
            continue
        rows.append(
            {
                "team": normalize_team_name(team_raw),
                "market_prob": float(prices[lowered.index("yes")]),
            }
        )
    frame = pl.DataFrame(rows, schema={"team": pl.Utf8, "market_prob": pl.Float64})
    if normalize and frame.height > 0:
        total = float(frame["market_prob"].sum())
        if total > 0:
            frame = frame.with_columns((pl.col("market_prob") / total).alias("market_prob"))
    return frame


def parse_team_yes_tokens(event: dict[str, Any]) -> pl.DataFrame:
    """Per-team **Yes** CLOB token ids from a grouped Yes/No futures event.

    Mirrors :func:`parse_team_yes_prices` but returns ``[team, token_id]`` — the token that
    :func:`fetch_polymarket_price_history` needs. ``clobTokenIds`` arrives as a
    JSON-encoded list aligned with the market's ``outcomes`` order; sub-markets without a
    Yes outcome, a token list of the wrong length, or a placeholder team are dropped.
    """

    from polymbappe.data.aliases import normalize_team_name

    rows: list[dict[str, Any]] = []
    for market in event.get("markets", []):
        team_raw = market.get("groupItemTitle") or ""
        outcomes_raw = market.get("outcomes")
        tokens_raw = market.get("clobTokenIds")
        if not team_raw or outcomes_raw is None or tokens_raw is None:
            continue
        if team_raw.strip().lower() in _NON_TEAM_LABELS:
            continue
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
        lowered = [str(o).strip().lower() for o in outcomes]
        if "yes" not in lowered or len(tokens) != len(outcomes):
            continue
        rows.append(
            {
                "team": normalize_team_name(team_raw),
                "token_id": str(tokens[lowered.index("yes")]),
            }
        )
    return pl.DataFrame(rows, schema={"team": pl.Utf8, "token_id": pl.Utf8})


def fetch_polymarket_price_history(
    token_id: str, *, interval: str = "max", fidelity: int = 1440, timeout: float = 30.0
) -> pl.DataFrame:
    """Price history for one CLOB token: ``[timestamp, price]`` (UTC, oldest first).

    ``fidelity`` is the sample spacing in minutes (1440 = daily). Any HTTP or payload-shape
    failure returns a typed empty frame — history for long-resolved markets is not
    guaranteed to stay served, and callers degrade to an "unavailable" note.
    """

    empty = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64})
    try:
        response = requests.get(
            CLOB_PRICES_HISTORY_URL,
            params={"market": token_id, "interval": interval, "fidelity": fidelity},
            headers=_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        points = response.json().get("history", [])
    except Exception:  # noqa: BLE001 - degrade to empty; caller logs context
        return empty
    rows = [
        {"t": int(p["t"]), "price": float(p["p"])}
        for p in points
        if isinstance(p, dict) and "t" in p and "p" in p
    ]
    if not rows:
        return empty
    timestamp = pl.from_epoch("t", time_unit="s").dt.replace_time_zone("UTC")
    return (
        pl.DataFrame(rows)
        .with_columns(timestamp.alias("timestamp"))
        .sort("timestamp")
        .select("timestamp", "price")
    )


def normalize_polymarket_three_way(
    long_prices: pl.DataFrame, draw_label: str = "Draw"
) -> pl.DataFrame:
    """Collapse long outcome prices into per-market three-way (team/draw/team) rows.

    A match market has exactly three outcomes: two team names and a draw. Prices are
    treated as implied probabilities and renormalized to remove the overround. Markets that
    aren't a clean three-way (no draw, wrong outcome count) are skipped.

    Returns ``[market_id, question, team_a, team_b, prob_a, prob_draw, prob_b]`` (team_a /
    team_b in their listed order; orientation to home/away happens in
    :func:`align_polymarket_to_fixtures`).
    """

    rows: list[dict[str, Any]] = []
    for (market_id,), group in long_prices.group_by(["market_id"]):
        outcomes = group["outcome"].to_list()
        prices = [float(p) for p in group["price"].to_list()]
        draw_idx = [i for i, o in enumerate(outcomes) if o.strip().lower() == draw_label.lower()]
        team_idx = [i for i in range(len(outcomes)) if i not in draw_idx]
        if len(draw_idx) != 1 or len(team_idx) != 2:
            continue
        total = sum(prices)
        if total <= 0:
            continue
        ia, ib = team_idx
        idr = draw_idx[0]
        rows.append(
            {
                "market_id": str(market_id),
                "question": group["question"].to_list()[0],
                "team_a": outcomes[ia],
                "team_b": outcomes[ib],
                "prob_a": prices[ia] / total,
                "prob_draw": prices[idr] / total,
                "prob_b": prices[ib] / total,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "market_id": pl.Utf8,
            "question": pl.Utf8,
            "team_a": pl.Utf8,
            "team_b": pl.Utf8,
            "prob_a": pl.Float64,
            "prob_draw": pl.Float64,
            "prob_b": pl.Float64,
        },
    )


def unmatched_market_teams(three_way: pl.DataFrame, fixtures: pl.DataFrame) -> list[str]:
    """Canonical market team names that match no fixture — i.e. need an alias entry.

    Compares the (normalized) teams appearing in Polymarket markets against the teams in
    the known fixtures. Any market team absent from the fixtures is reported so a spelling
    can be added to ``configs/team_aliases.yaml``. Returns a sorted, de-duplicated list.
    """

    from polymbappe.data.aliases import normalize_team_name

    fixture_teams: set[str] = set()
    for r in fixtures.iter_rows(named=True):
        fixture_teams.add(normalize_team_name(r["home_team"]))
        fixture_teams.add(normalize_team_name(r["away_team"]))
    market_teams: set[str] = set()
    for m in three_way.iter_rows(named=True):
        market_teams.add(normalize_team_name(m["team_a"]))
        market_teams.add(normalize_team_name(m["team_b"]))
    return sorted(market_teams - fixture_teams)


def align_polymarket_to_fixtures(
    three_way: pl.DataFrame, fixtures: pl.DataFrame, *, source: str = "polymarket"
) -> pl.DataFrame:
    """Map three-way market rows onto known fixtures, oriented to home/away.

    ``fixtures`` provides ``[match_id, home_team, away_team]``. A market is matched to a
    fixture by its unordered team pair, then its probabilities are oriented to the fixture's
    home/away so the result joins ``match_predictions`` by ``match_id``. Team spellings must
    match across sources (normalization is handled upstream). Returns the ``market_odds``
    schema.
    """

    from polymbappe.data.aliases import normalize_team_name

    lookup: dict[frozenset[str], dict[str, str]] = {
        frozenset(
            {normalize_team_name(r["home_team"]), normalize_team_name(r["away_team"])}
        ): {
            "match_id": r["match_id"],
            "home_team": normalize_team_name(r["home_team"]),
            "away_team": normalize_team_name(r["away_team"]),
        }
        for r in fixtures.iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for m in three_way.iter_rows(named=True):
        team_a = normalize_team_name(m["team_a"])
        team_b = normalize_team_name(m["team_b"])
        fixture = lookup.get(frozenset({team_a, team_b}))
        if fixture is None:
            continue
        a_is_home = team_a == fixture["home_team"]
        home_p = m["prob_a"] if a_is_home else m["prob_b"]
        away_p = m["prob_b"] if a_is_home else m["prob_a"]
        rows.append(
            {
                "match_id": fixture["match_id"],
                "source": source,
                "home_win_prob": float(home_p),
                "draw_prob": float(m["prob_draw"]),
                "away_win_prob": float(away_p),
                "timestamp": None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "match_id": pl.Utf8,
            "source": pl.Utf8,
            "home_win_prob": pl.Float64,
            "draw_prob": pl.Float64,
            "away_win_prob": pl.Float64,
            "timestamp": pl.Datetime,
        },
    )
