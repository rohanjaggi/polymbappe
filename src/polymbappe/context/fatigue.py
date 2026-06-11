"""Fatigue & schedule features (spec 2.2 Group D).

Three signals:

* **Rest days** between a team's consecutive matches (``<4`` days flags fatigue). The
  point-in-time rest-day builder already lives in :mod:`polymbappe.features.context`; this
  module adds the fatigue flag and the tournament-schedule travel/load signals.
* **Travel distance** — haversine distance between a team's consecutive match venues
  (2026 has 16 host cities spread across three countries).
* **Club season minutes load** — total minutes played by the expected XI in the preceding
  club season ("tired from a 60-game season").
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import polars as pl

#: Approximate (lat, lon) of the 16 FIFA 2026 host cities. Keys match the venue strings
#: used in the tournament schedule; spelling normalization is handled upstream.
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


def venue_distance(home_city: str, away_city: str) -> float | None:
    """Haversine distance between two host cities (None if either is unknown)."""

    a = HOST_CITY_COORDS.get(home_city)
    b = HOST_CITY_COORDS.get(away_city)
    if a is None or b is None:
        return None
    return haversine_km(a, b)


def build_travel_features(schedule: pl.DataFrame) -> pl.DataFrame:
    """Per-appearance travel distance from a team's previous venue.

    Args:
        schedule: Frame with columns ``[team, date, match_id, venue]`` (one row per team
            appearance, venue a key in :data:`HOST_CITY_COORDS`).

    Returns:
        Frame ``[match_id, team, travel_km]``. The first appearance has ``travel_km`` 0.0.
    """

    required = {"team", "date", "match_id", "venue"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule frame missing columns: {sorted(missing)}")

    df = schedule.sort(["team", "date", "match_id"])
    rows: list[dict[str, object]] = []
    prev_city: dict[str, str] = {}
    for rec in df.iter_rows(named=True):
        team, venue = rec["team"], rec["venue"]
        last = prev_city.get(team)
        travel = 0.0 if last is None else (venue_distance(last, venue) or 0.0)
        rows.append({"match_id": rec["match_id"], "team": team, "travel_km": float(travel)})
        prev_city[team] = venue
    return pl.DataFrame(
        rows, schema={"match_id": pl.Utf8, "team": pl.Utf8, "travel_km": pl.Float64}
    )


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
