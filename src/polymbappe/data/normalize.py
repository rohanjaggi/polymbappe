"""Pure normalization transforms.

Every function here is side-effect free: raw bytes / parsed HTML / raw dataframes in,
schema-shaped Polars dataframes out. No network, no disk. This keeps the brittle parts
of ingestion fully unit-testable without hitting any external source.
"""

from __future__ import annotations

import re
from datetime import date

import polars as pl
from bs4 import BeautifulSoup

from polymbappe.data.tables import TABLE_COLUMNS, Table

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Lowercase, collapse non-alphanumerics to single underscores."""

    return _SLUG_RE.sub("_", value.strip().lower()).strip("_")


def make_match_id(match_date: date | str, home_team: str, away_team: str) -> str:
    """Deterministic match id from date and the two teams."""

    return f"{match_date}__{slugify(home_team)}__{slugify(away_team)}"


# ---------------------------------------------------------------------------
# Kaggle international results (martj42/international_results)
# ---------------------------------------------------------------------------

# Raw columns: date, home_team, away_team, home_score, away_score, tournament,
#              city, country, neutral
_KAGGLE_RENAME = {
    "home_score": "home_goals",
    "away_score": "away_goals",
    "tournament": "competition",
    "neutral": "neutral_site",
}


def normalize_kaggle_results(raw: pl.DataFrame) -> pl.DataFrame:
    """Normalize the Kaggle international results CSV into the ``matches`` schema.

    Drops unplayed fixtures (null scores) and coerces types. ``is_knockout`` and
    ``group`` are left at structural defaults — the source carries no stage metadata.
    """

    present = {k: v for k, v in _KAGGLE_RENAME.items() if k in raw.columns}
    df = raw.rename(present)

    df = df.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("date"),
        pl.col("home_team").cast(pl.Utf8),
        pl.col("away_team").cast(pl.Utf8),
        pl.col("home_goals").cast(pl.Int64, strict=False),
        pl.col("away_goals").cast(pl.Int64, strict=False),
        pl.col("competition").cast(pl.Utf8),
    )

    if "neutral_site" in df.columns:
        df = df.with_columns(
            pl.col("neutral_site").cast(pl.Boolean, strict=False).fill_null(False)
        )
    else:
        df = df.with_columns(pl.lit(False).alias("neutral_site"))

    df = df.drop_nulls(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    df = df.with_columns(
        pl.format("{}__{}__{}", pl.col("date"), pl.col("home_team"), pl.col("away_team"))
        .alias("match_id"),
        pl.lit(False).alias("is_knockout"),
        pl.lit(None, dtype=pl.Utf8).alias("group"),
    )

    return df.select(TABLE_COLUMNS[Table.MATCHES])


# ---------------------------------------------------------------------------
# EloRatings.net
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^-?\d{2,5}$")


def parse_eloratings(soup: BeautifulSoup, as_of: date) -> pl.DataFrame:
    """Extract (team, date, rating) rows from an EloRatings.net ranking table.

    Best-effort and structure-tolerant: for each table row, the first anchor's text is
    taken as the team and the first standalone integer cell as the rating. Rows without
    both are skipped.
    """

    teams: list[str] = []
    ratings: list[int] = []

    for row in soup.find_all("tr"):
        anchor = row.find("a")
        if anchor is None:
            continue
        team = anchor.get_text(strip=True)
        if not team:
            continue
        rating: int | None = None
        for cell in row.find_all("td"):
            text = cell.get_text(strip=True).replace(",", "")
            if _NUMERIC_RE.match(text):
                rating = int(text)
                break
        if rating is None:
            continue
        teams.append(team)
        ratings.append(rating)

    return pl.DataFrame(
        {
            "team": teams,
            "date": [as_of] * len(teams),
            "rating": [float(r) for r in ratings],
        },
        schema={"team": pl.Utf8, "date": pl.Date, "rating": pl.Float64},
    )


# ---------------------------------------------------------------------------
# Market odds (decimal odds -> overround-removed probabilities)
# ---------------------------------------------------------------------------


def implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Convert decimal H/D/A odds to overround-removed probabilities (sum to 1).

    Uses the basic normalization (proportional margin removal). Raises ``ValueError``
    for non-positive odds.
    """

    if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
        raise ValueError("Decimal odds must be positive.")
    raw_h, raw_d, raw_a = 1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds
    overround = raw_h + raw_d + raw_a
    return raw_h / overround, raw_d / overround, raw_a / overround


#: Bookmaker odds-column prefixes in Football-Data.co.uk CSVs, best first: Bet365,
#: market average, Pinnacle, then the older Bet&Win / average columns.
_FOOTBALLDATA_PREFIXES: tuple[str, ...] = ("B365", "Avg", "PS", "P", "BW", "BbAv")


def normalize_footballdata_odds(
    raw: pl.DataFrame, *, source: str = "football-data"
) -> pl.DataFrame:
    """Normalize a Football-Data.co.uk CSV into the ``market_odds`` schema.

    Picks the first available bookmaker odds triple (``{prefix}H/D/A`` for a prefix in
    :data:`_FOOTBALLDATA_PREFIXES`), builds the ``date__home__away`` match id (matching the
    matches table convention so odds join by id), and removes the overround. Football-Data
    covers club leagues, so these odds join any match sharing that id convention. Rows
    missing the chosen odds, date, or teams are dropped.
    """

    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Football-Data CSV missing columns: {sorted(missing)}")

    prefix = next(
        (
            p
            for p in _FOOTBALLDATA_PREFIXES
            if {f"{p}H", f"{p}D", f"{p}A"}.issubset(raw.columns)
        ),
        None,
    )
    if prefix is None:
        raise ValueError("Football-Data CSV has no recognized H/D/A odds columns.")

    iso_date = pl.coalesce(
        pl.col("Date").cast(pl.Utf8).str.to_date("%d/%m/%Y", strict=False),
        pl.col("Date").cast(pl.Utf8).str.to_date("%d/%m/%y", strict=False),
    )
    prepared = raw.with_columns(iso_date.alias("_date")).drop_nulls("_date").with_columns(
        pl.format("{}__{}__{}", pl.col("_date"), pl.col("HomeTeam"), pl.col("AwayTeam"))
        .alias("match_id")
    )
    return normalize_odds_frame(
        prepared,
        source=source,
        home_col=f"{prefix}H",
        draw_col=f"{prefix}D",
        away_col=f"{prefix}A",
        match_id_col="match_id",
        timestamp_col=None,
    )


def normalize_odds_frame(
    raw: pl.DataFrame,
    *,
    source: str,
    home_col: str,
    draw_col: str,
    away_col: str,
    match_id_col: str = "match_id",
    timestamp_col: str | None = "timestamp",
) -> pl.DataFrame:
    """Normalize a frame of decimal odds into the ``market_odds`` schema.

    Removes the bookmaker overround per row. Rows with non-positive or null odds are
    dropped. If ``timestamp_col`` is absent, the timestamp column is filled with nulls.
    """

    df = raw.drop_nulls(subset=[home_col, draw_col, away_col]).filter(
        (pl.col(home_col) > 0) & (pl.col(draw_col) > 0) & (pl.col(away_col) > 0)
    )

    inv_h = 1.0 / pl.col(home_col)
    inv_d = 1.0 / pl.col(draw_col)
    inv_a = 1.0 / pl.col(away_col)
    overround = inv_h + inv_d + inv_a

    if timestamp_col is not None and timestamp_col in df.columns:
        timestamp_expr = pl.col(timestamp_col).cast(pl.Datetime, strict=False)
    else:
        timestamp_expr = pl.lit(None, dtype=pl.Datetime)

    return df.select(
        pl.col(match_id_col).cast(pl.Utf8).alias("match_id"),
        pl.lit(source).alias("source"),
        (inv_h / overround).alias("home_win_prob"),
        (inv_d / overround).alias("draw_prob"),
        (inv_a / overround).alias("away_win_prob"),
        timestamp_expr.alias("timestamp"),
    )
