"""Fatigue & schedule features (spec 2.2 Group D).

Three signals:

* **Rest days** between a team's consecutive matches (``<4`` days flags fatigue). The
  point-in-time rest-day builder already lives in :mod:`polymbappe.features.context`; this
  module adds the fatigue flag and the tournament-schedule travel/load signals.
* **Travel distance** — haversine distance between a team's consecutive match venues. The
  fixtures come from the ingested ``schedule`` table and the per-stadium coordinates from the
  ingested ``venues`` table (both openfootball, ingested in :mod:`polymbappe.data.ingest`);
  :func:`build_travel_features_from_tables` ties them together. A static 16-host-city table
  (:data:`HOST_CITY_COORDS`) remains only as an offline fallback.
* **Club season minutes load** — total minutes played by the expected XI in the preceding
  club season ("tired from a 60-game season").
"""

from __future__ import annotations

import re
import unicodedata
from math import asin, cos, radians, sin, sqrt

import polars as pl


def _asciifold(value: str) -> str:
    """Lower-case and strip accents (``"São Paulo"`` -> ``"sao paulo"``) for city matching.

    The GeoNames gazetteer keys are the ASCII ``asciiname`` (plus Latin alternatenames), so a
    match's accented ``city`` must be folded the same way to resolve. NFKD decomposition drops
    combining marks without any extra dependency.
    """

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()

#: Offline **fallback** coordinates for the 16 FIFA 2026 host cities, keyed by bare city name.
#: The source of truth is now the ingested ``venues`` table (openfootball, real per-stadium
#: coordinates) consumed via :func:`coord_lookup_from_venues`; this static table is only used
#: when no ``coords`` mapping is supplied, so :func:`venue_distance` /
#: :func:`build_travel_features` keep working without ingestion (and existing tests stay
#: offline).
HOST_CITY_COORDS: dict[str, tuple[float, float]] = {
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.0909, -71.2643),  # Foxborough
    "Dallas": (32.7473, -97.0945),  # Arlington
    "Houston": (29.6847, -95.4107),
    "Kansas City": (39.0489, -94.4839),
    "Los Angeles": (33.9535, -118.3392),  # Inglewood
    "Miami": (25.9580, -80.2389),
    "New York": (40.8136, -74.0744),  # East Rutherford
    "Philadelphia": (39.9008, -75.1675),
    "San Francisco": (37.4030, -121.9700),  # Santa Clara
    "Seattle": (47.5952, -122.3316),
    "Guadalajara": (20.6818, -103.4626),
    "Mexico City": (19.3029, -99.1505),
    "Monterrey": (25.6692, -100.2444),
    "Toronto": (43.6332, -79.4185),
    "Vancouver": (49.2768, -123.1120),
}

FATIGUE_REST_THRESHOLD_DAYS: int = 4


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""

    lat1, lon1 = radians(a[0]), radians(a[1])
    lat2, lon2 = radians(b[0]), radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2.0 * 6371.0088 * asin(sqrt(h))


def _bare_city(name: str) -> str:
    """Strip a trailing ``" (district)"`` qualifier (``"Boston (Foxborough)"`` -> ``"Boston"``)."""

    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _lookup_coord(
    city: str, coords: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """Resolve a city's coordinates, falling back to its bare (de-parenthesized) name."""

    hit = coords.get(city)
    if hit is None:
        hit = coords.get(_bare_city(city))
    return hit


def coord_lookup_from_venues(venues: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Build a ``city -> (lat, lon)`` lookup from the ingested ``venues`` table.

    Each venue is registered under both its full host-city string (e.g.
    ``"Boston (Foxborough)"``, which the schedule's ``city`` uses verbatim) and its bare city
    name (``"Boston"``), so a schedule referencing either spelling resolves. This replaces the
    static :data:`HOST_CITY_COORDS` table as the travel feature's coordinate source.

    Requires ``city``, ``latitude``, ``longitude`` columns (the ``venues`` schema).
    """

    required = {"city", "latitude", "longitude"}
    missing = required - set(venues.columns)
    if missing:
        raise ValueError(f"venues frame missing columns: {sorted(missing)}")

    coords: dict[str, tuple[float, float]] = {}
    for rec in venues.iter_rows(named=True):
        city, lat, lon = rec["city"], rec["latitude"], rec["longitude"]
        if city is None or lat is None or lon is None:
            continue
        point = (float(lat), float(lon))
        coords[str(city)] = point
        coords.setdefault(_bare_city(str(city)), point)
    return coords


def build_city_coord_lookup(city_coords: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Build a lower-cased ``city -> (lat, lon)`` lookup from the GeoNames gazetteer table.

    The gazetteer (`Table.CITY_COORDS`) carries one row per ``(city, country)`` name alias;
    a bare city string can map to several countries (many "Springfield"s), so the
    **highest-population** entry wins each city name — which resolves tournament host cities
    correctly (Saint Petersburg RU over FL, Manchester UK over NH) without needing the
    match's country. Keys are lower-cased to match `build_match_travel_features`, which
    case-folds the match ``city`` before lookup.

    Requires ``city``, ``latitude``, ``longitude``, ``population`` columns.
    """

    required = {"city", "latitude", "longitude", "population"}
    missing = required - set(city_coords.columns)
    if missing:
        raise ValueError(f"city_coords frame missing columns: {sorted(missing)}")

    best = city_coords.sort("population", descending=True).unique(
        subset=["city"], keep="first"
    )
    coords: dict[str, tuple[float, float]] = {}
    for rec in best.iter_rows(named=True):
        city, lat, lon = rec["city"], rec["latitude"], rec["longitude"]
        if city is None or lat is None or lon is None:
            continue
        coords[_asciifold(str(city))] = (float(lat), float(lon))
    return coords


def build_match_travel_features(
    matches: pl.DataFrame, coords: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Per-(match, team) travel distance over a set of matches, geocoded by ``city``.

    Each match's ``city`` is the venue; per team, travel is the haversine from its previous
    match's city to this one (first appearance = 0.0). Use it to **backfill** historical
    ``travel_km`` over any match frame carrying ``[match_id, date, home_team, away_team,
    city]`` (e.g. one tournament's fixtures) with a `build_city_coord_lookup` gazetteer.

    ``city`` is case-folded to match the lower-cased ``coords`` keys; matches whose city does
    not resolve contribute 0.0 (handled by :func:`venue_distance`). Returns
    ``[match_id, team, travel_km]``.
    """

    if "city" not in matches.columns:
        raise ValueError("matches frame needs a 'city' column for travel backfill.")
    folded = matches.with_columns(
        pl.col("city")
        .cast(pl.Utf8)
        .map_elements(lambda c: _asciifold(c) if c is not None else None, return_dtype=pl.Utf8)
        .alias("city")
    )
    appearances = schedule_to_appearances(folded)
    return build_travel_features(appearances, coords)


def venue_distance(
    home_city: str,
    away_city: str,
    coords: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Haversine distance between two host cities (None if either is unknown).

    ``coords`` defaults to the offline :data:`HOST_CITY_COORDS` fallback; pass
    :func:`coord_lookup_from_venues` output to use the ingested per-stadium coordinates. Each
    city is matched exactly, then by its bare (de-parenthesized) name.
    """

    coords = coords if coords is not None else HOST_CITY_COORDS
    a = _lookup_coord(home_city, coords)
    b = _lookup_coord(away_city, coords)
    if a is None or b is None:
        return None
    return haversine_km(a, b)


def schedule_to_appearances(schedule: pl.DataFrame) -> pl.DataFrame:
    """Explode a ``schedule`` table into one ``[team, date, match_id, venue]`` row per team.

    Each fixture contributes two appearances (home + away), with ``venue`` taken from the
    fixture's ``city`` (the openfootball host-city string), producing exactly the frame
    :func:`build_travel_features` consumes. Group-stage rows carry real nations; knockout
    rows carry bracket placeholders (e.g. ``"2A"``) whose travel is only meaningful once the
    bracket is resolved, so callers wanting deterministic travel can pre-filter to ``group``
    being non-null.

    Requires ``match_id``, ``date``, ``home_team``, ``away_team``, ``city`` columns.
    """

    required = {"match_id", "date", "home_team", "away_team", "city"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule frame missing columns: {sorted(missing)}")

    home = schedule.select(
        pl.col("home_team").alias("team"), "date", "match_id", pl.col("city").alias("venue")
    )
    away = schedule.select(
        pl.col("away_team").alias("team"), "date", "match_id", pl.col("city").alias("venue")
    )
    return pl.concat([home, away], how="vertical")


def build_travel_features(
    schedule: pl.DataFrame,
    coords: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Per-appearance travel distance from a team's previous venue.

    Args:
        schedule: Frame with columns ``[team, date, match_id, venue]`` (one row per team
            appearance; ``venue`` a key resolvable in ``coords``). Build it from the ingested
            ``schedule`` table via :func:`schedule_to_appearances`.
        coords: ``city -> (lat, lon)`` lookup; defaults to the offline
            :data:`HOST_CITY_COORDS` fallback. Pass :func:`coord_lookup_from_venues` output to
            use the ingested per-stadium coordinates.

    Returns:
        Frame ``[match_id, team, travel_km]``. The first appearance has ``travel_km`` 0.0.
    """

    required = {"team", "date", "match_id", "venue"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule frame missing columns: {sorted(missing)}")

    coords = coords if coords is not None else HOST_CITY_COORDS
    df = schedule.sort(["team", "date", "match_id"])
    rows: list[dict[str, object]] = []
    prev_city: dict[str, str] = {}
    for rec in df.iter_rows(named=True):
        team, venue = rec["team"], rec["venue"]
        last = prev_city.get(team)
        travel = 0.0 if last is None else (venue_distance(last, venue, coords) or 0.0)
        rows.append({"match_id": rec["match_id"], "team": team, "travel_km": float(travel)})
        prev_city[team] = venue
    return pl.DataFrame(
        rows, schema={"match_id": pl.Utf8, "team": pl.Utf8, "travel_km": pl.Float64}
    )


def build_travel_features_from_tables(
    schedule: pl.DataFrame, venues: pl.DataFrame
) -> pl.DataFrame:
    """Travel features straight from the ingested ``schedule`` + ``venues`` tables.

    Convenience wrapper: derive the coordinate lookup from ``venues``
    (:func:`coord_lookup_from_venues`), explode the schedule into per-team appearances
    (:func:`schedule_to_appearances`), and compute per-appearance travel
    (:func:`build_travel_features`). Returns ``[match_id, team, travel_km]``.
    """

    coords = coord_lookup_from_venues(venues)
    appearances = schedule_to_appearances(schedule)
    return build_travel_features(appearances, coords)


def add_fatigue_flag(
    rest_features: pl.DataFrame, threshold: int = FATIGUE_REST_THRESHOLD_DAYS
) -> pl.DataFrame:
    """Add a ``fatigued`` boolean to a rest-days frame (``rest_days < threshold``)."""

    if "rest_days" not in rest_features.columns:
        raise ValueError("rest_features must contain a 'rest_days' column.")
    return rest_features.with_columns(
        (pl.col("rest_days") < threshold).fill_null(False).alias("fatigued")
    )


def build_season_load_features(minutes: pl.DataFrame) -> pl.DataFrame:
    """Normalize a club season-minutes table to a per-team load feature.

    Args:
        minutes: Frame with columns ``[team, tournament, season_minutes]`` (total expected-XI
            minutes in the preceding club season).

    Returns:
        Frame ``[team, tournament, season_minutes, season_load]`` where ``season_load`` is
        z-scored within the tournament (positive = more loaded than peers).
    """

    required = {"team", "tournament", "season_minutes"}
    missing = required - set(minutes.columns)
    if missing:
        raise ValueError(f"minutes frame missing columns: {sorted(missing)}")

    mean = pl.col("season_minutes").mean().over("tournament")
    std = pl.col("season_minutes").std().over("tournament")
    return minutes.with_columns(
        pl.when(std > 0)
        .then((pl.col("season_minutes") - mean) / std)
        .otherwise(0.0)
        .alias("season_load")
    )
