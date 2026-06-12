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

#: Competitions whose knockout stage is inferred by :func:`infer_knockout_stage`. Strings
#: must match the Kaggle ``tournament`` field exactly (cf.
#: ``polymbappe.eval.backtest.DEFAULT_TOURNAMENTS``). Scoped to the major single-elimination
#: tournaments the model evaluates on; qualifiers, leagues, and round-robin competitions are
#: deliberately excluded.
_KNOCKOUT_COMPETITIONS: frozenset[str] = frozenset(
    {"FIFA World Cup", "UEFA Euro", "Copa América"}
)


def infer_knockout_stage(matches: pl.DataFrame) -> pl.Series:
    """Infer a per-row ``is_knockout`` flag for major-tournament matches.

    The Kaggle results feed carries no stage metadata, so knockout matches are inferred
    structurally. Within each *edition* (competition + calendar year) of a major tournament
    (:data:`_KNOCKOUT_COMPETITIONS`) the group stage is a round-robin in which every team
    plays the same number of games, while the knockout stage is single-elimination. A team
    eliminated in the group stage therefore plays the *minimum* number of matches in that
    edition; any match a team plays beyond that minimum is a knockout match. ``group_size``
    is taken per edition as that minimum appearance count, which self-calibrates to each
    format (3 group games for the modern World Cup / Euro / Copa, fewer for older ones).

    A match is flagged knockout when **both** sides have already played at least
    ``group_size`` matches in the edition — i.e. it is each team's ``group_size + 1``-th
    appearance or later. In a single-elimination bracket both teams in a tie have survived
    the same number of rounds, so their appearance counts are always equal. Non-major
    competitions and round-robin-only editions (where no team exceeds the minimum) yield
    all-False.

    Assumes complete editions: a partially-ingested in-progress tournament can under-label,
    and the rare historical double-group-stage formats (e.g. the 1974/1978/1982 World Cups)
    over-label their second group round. Neither affects the modern editions the model
    trains and evaluates on.

    Returns a Boolean Series ``is_knockout`` aligned to ``matches`` (same length and order).
    Requires ``date``, ``home_team``, ``away_team``, and ``competition`` columns.
    """

    n = matches.height
    if n == 0:
        return pl.Series("is_knockout", [], dtype=pl.Boolean)

    work = matches.select(
        pl.int_range(0, n, dtype=pl.Int64).alias("_row"),
        "date",
        "home_team",
        "away_team",
        "competition",
    ).with_columns(
        (pl.col("competition") + pl.lit("|") + pl.col("date").dt.year().cast(pl.Utf8))
        .alias("_edition")
    )

    major = work.filter(pl.col("competition").is_in(list(_KNOCKOUT_COMPETITIONS)))
    if major.is_empty():
        return pl.Series("is_knockout", [False] * n, dtype=pl.Boolean)

    # One row per (match, team), so each team's appearances within an edition can be counted.
    long = pl.concat(
        [
            major.select("_row", "_edition", "date", pl.col("home_team").alias("team")),
            major.select("_row", "_edition", "date", pl.col("away_team").alias("team")),
        ]
    ).sort(["_edition", "team", "date", "_row"])

    # 1-based chronological appearance index per (edition, team).
    long = long.with_columns(
        pl.int_range(1, pl.len() + 1).over(["_edition", "team"]).alias("_appearance")
    )
    # group_size := fewest matches any team played in the edition (group-stage-only teams).
    group_size = (
        long.group_by(["_edition", "team"])
        .agg(pl.len().alias("_team_matches"))
        .group_by("_edition")
        .agg(pl.col("_team_matches").min().alias("_group_size"))
    )
    long = long.join(group_size, on="_edition", how="left").with_columns(
        (pl.col("_appearance") > pl.col("_group_size")).alias("_past_group")
    )
    # A match is knockout when both of its team-rows are past the group stage.
    per_match = long.group_by("_row").agg(pl.col("_past_group").all().alias("_knockout"))

    flags = (
        work.join(per_match, on="_row", how="left")
        .with_columns(pl.col("_knockout").fill_null(False))
        .sort("_row")
    )
    return flags.get_column("_knockout").rename("is_knockout")


def normalize_kaggle_results(raw: pl.DataFrame) -> pl.DataFrame:
    """Normalize the Kaggle international results CSV into the ``matches`` schema.

    Drops unplayed fixtures (null scores) and coerces types. The source carries no stage
    metadata, so ``is_knockout`` is inferred structurally for the major tournaments by
    :func:`infer_knockout_stage`; ``group`` is left null (no group labels in the feed).
    """

    present = {k: v for k, v in _KAGGLE_RENAME.items() if k in raw.columns}
    df = raw.rename(present)

    from polymbappe.data.aliases import normalize_team_expr

    df = df.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("date"),
        normalize_team_expr("home_team").alias("home_team"),
        normalize_team_expr("away_team").alias("away_team"),
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
        pl.lit(None, dtype=pl.Utf8).alias("group"),
    )
    df = df.with_columns(infer_knockout_stage(df))

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

    from polymbappe.data.aliases import normalize_team_expr

    iso_date = pl.coalesce(
        pl.col("Date").cast(pl.Utf8).str.to_date("%d/%m/%Y", strict=False),
        pl.col("Date").cast(pl.Utf8).str.to_date("%d/%m/%y", strict=False),
    )
    prepared = (
        raw.with_columns(
            iso_date.alias("_date"),
            normalize_team_expr("HomeTeam").alias("HomeTeam"),
            normalize_team_expr("AwayTeam").alias("AwayTeam"),
        )
        .drop_nulls("_date")
        .with_columns(
            pl.format("{}__{}__{}", pl.col("_date"), pl.col("HomeTeam"), pl.col("AwayTeam"))
            .alias("match_id")
        )
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


# ---------------------------------------------------------------------------
# Player attributes (EA FC / FIFA via stefanoleone992, FM25)
# ---------------------------------------------------------------------------

#: Candidate raw column names per logical field, in priority order. EA FC / FIFA exports
#: use ``short_name`` / ``nationality_name`` / ``overall`` (older editions: ``nationality``);
#: Football Manager exports are not standardized — common spellings are listed so a single
#: reconciler covers both without per-source branching. See
#: ``docs/.../2026-06-09-data-ingestion-requirements-spec.md`` ("Player attributes").
_PLAYER_ATTR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "player": ("short_name", "long_name", "name", "Name", "player", "Player"),
    "team": (
        "nationality_name",
        "nationality",
        "nation",
        "Nation",
        "country",
        "Country",
        "team",
    ),
    "overall": ("overall", "Overall", "overall_rating", "CA", "ability"),
}


def _resolve_attr_column(field: str, override: str | None, columns: list[str]) -> str:
    """Resolve the raw column for a logical field (``override`` wins, else candidates).

    Raises ``ValueError`` (listing the available columns) when neither the override nor any
    candidate from :data:`_PLAYER_ATTR_CANDIDATES` is present.
    """

    if override is not None:
        return override
    present = set(columns)
    found = next((c for c in _PLAYER_ATTR_CANDIDATES[field] if c in present), None)
    if found is None:
        raise ValueError(
            f"player attributes source missing a column for {field!r}; "
            f"available columns: {sorted(columns)}"
        )
    return found


def normalize_player_attributes(
    raw: pl.DataFrame,
    *,
    player_col: str | None = None,
    team_col: str | None = None,
    overall_col: str | None = None,
) -> pl.DataFrame:
    """Reconcile an EA FC / FM player-attributes export to ``team, player, overall`` rows.

    Pure column-reconciliation: EA FC and Football Manager exports name the same fields
    differently (and FM names are not standardized), so the player / national-team / rating
    columns are resolved from :data:`_PLAYER_ATTR_CANDIDATES` unless explicitly overridden.
    ``team`` is the player's *national* team (canonicalized to a tournament squad later, at
    ingest); ``overall`` is cast to ``Int64``. Rows missing a name or rating are dropped.
    Team-name canonicalization is **not** done here (it happens in
    :func:`~polymbappe.data.ingest.ingest_player_attributes`, mirroring the squad sources).

    Raises ``ValueError`` if a required column can't be resolved.
    """

    columns = raw.columns
    team = _resolve_attr_column("team", team_col, columns)
    player = _resolve_attr_column("player", player_col, columns)
    overall = _resolve_attr_column("overall", overall_col, columns)

    return (
        raw.select(
            pl.col(team).cast(pl.Utf8).str.strip_chars().alias("team"),
            pl.col(player).cast(pl.Utf8).str.strip_chars().alias("player"),
            pl.col(overall).cast(pl.Int64, strict=False).alias("overall"),
        )
        .drop_nulls(subset=["player", "overall"])
        .filter(pl.col("player") != "")
    )
