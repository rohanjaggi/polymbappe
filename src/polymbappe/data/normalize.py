"""Pure normalization transforms.

Every function here is side-effect free: raw bytes / parsed HTML / raw dataframes in,
schema-shaped Polars dataframes out. No network, no disk. This keeps the brittle parts
of ingestion fully unit-testable without hitting any external source.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

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

#: First day of the 2026 World Cup — scopes the ``wc2026_ko_start`` override in
#: :func:`normalize_kaggle_results` to the 2026 edition only.
_WC2026_START = date(2026, 6, 11)


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


def normalize_kaggle_results(
    raw: pl.DataFrame, *, wc2026_ko_start: date | None = None
) -> pl.DataFrame:
    """Normalize the Kaggle international results CSV into the ``matches`` schema.

    Drops unplayed fixtures (null scores) and coerces types. The source carries no stage
    metadata, so ``is_knockout`` is inferred structurally for the major tournaments by
    :func:`infer_knockout_stage`; ``group`` is left null (no group labels in the feed).

    ``wc2026_ko_start`` (the first Round-of-32 date from the ingested schedule) overrides
    the heuristic for WC 2026 rows: group stage and knockout rounds don't overlap in the
    2026 calendar, so ``date >= wc2026_ko_start`` is exact — unlike the structural
    heuristic, which mislabels matchday-3 fixtures while the tournament is in progress.
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
    if wc2026_ko_start is not None:
        df = df.with_columns(
            pl.when(
                (pl.col("competition") == "FIFA World Cup")
                & (pl.col("date") >= pl.lit(_WC2026_START))
            )
            .then(pl.col("date") >= pl.lit(wc2026_ko_start))
            .otherwise(pl.col("is_knockout"))
            .alias("is_knockout")
        )

    # ``city`` / ``country`` are the match venue, carried straight from the feed (they are the
    # per-match venue signal the travel feature backfills against). Absent columns are filled
    # null so the matches schema stays dense for older mirrors that omit them.
    for col in ("city", "country"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).str.strip_chars().alias(col))
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))

    return df.select(TABLE_COLUMNS[Table.MATCHES])


# ---------------------------------------------------------------------------
# StatsBomb Open Data (events -> team xG / PPDA)
# ---------------------------------------------------------------------------

#: Pitch length in StatsBomb coordinates (x runs 0 = own goal-line to 120 = opponent's).
_SB_PITCH_LENGTH = 120.0
#: Fraction of the pitch (from a team's own goal) treated as the PPDA "build-up" zone — the
#: classic definition presses in the opponent's defensive 60%.
_SB_PPDA_ZONE_FRACTION = 0.6


def _sb_type(event: dict[str, Any]) -> str:
    return str(event.get("type", {}).get("name", ""))


def statsbomb_team_match_xg(
    events: list[dict[str, Any]], *, home_team: str, away_team: str, match_date: str
) -> list[dict[str, object]]:
    """Sum a match's shot xG per team into two ``team_xg`` rows.

    Team xG is the sum of ``shot.statsbomb_xg`` over that team's shots; ``xga`` is the
    opponent's total. Penalty-shootout shots (period 5) are excluded. Returns
    ``[{team, date, xg, xga}, ...]`` for the home and away teams (raw StatsBomb names;
    canonicalization happens at ingest).
    """

    xg = {home_team: 0.0, away_team: 0.0}
    for e in events:
        if e.get("period") == 5 or _sb_type(e) != "Shot":
            continue
        team = e.get("team", {}).get("name")
        value = e.get("shot", {}).get("statsbomb_xg")
        if team in xg and value is not None:
            xg[team] += float(value)
    return [
        {"team": home_team, "date": match_date, "xg": xg[home_team], "xga": xg[away_team]},
        {"team": away_team, "date": match_date, "xg": xg[away_team], "xga": xg[home_team]},
    ]


def _sb_is_defensive_action(event: dict[str, Any]) -> bool:
    """A PPDA defensive action: interception, foul committed, or a tackle-type duel."""

    kind = _sb_type(event)
    if kind in ("Interception", "Foul Committed"):
        return True
    return kind == "Duel" and event.get("duel", {}).get("type", {}).get(
        "name", ""
    ).startswith("Tackle")


def statsbomb_team_match_ppda(
    events: list[dict[str, Any]],
    *,
    home_team: str,
    away_team: str,
    match_date: str,
    zone_fraction: float = _SB_PPDA_ZONE_FRACTION,
) -> list[dict[str, object]]:
    """Compute each team's PPDA for one match from its event stream.

    PPDA = opponent passes in their build-up zone ÷ this team's defensive actions in the
    pressing zone (tackles + interceptions + fouls). Both restricted to the same physical
    band: a team's own passes count where ``x <= 120*zone_fraction`` (its defensive 60%);
    the pressing team's defensive actions count where ``x >= 120*(1 - zone_fraction)`` — the
    mirror of that band in the presser's attacking-right frame. PPDA is ``None`` when a team
    made no defensive actions in the zone (avoids divide-by-zero). Shootout events (period 5)
    are excluded. Returns ``[{team, date, ppda}, ...]`` for home and away.
    """

    pass_line = _SB_PITCH_LENGTH * zone_fraction
    press_line = _SB_PITCH_LENGTH * (1.0 - zone_fraction)
    passes = {home_team: 0, away_team: 0}
    def_actions = {home_team: 0, away_team: 0}
    for e in events:
        if e.get("period") == 5:
            continue
        team = e.get("team", {}).get("name")
        location = e.get("location")
        if team not in passes or not location:
            continue
        x = location[0]
        if _sb_type(e) == "Pass":
            if x <= pass_line:
                passes[team] += 1
        elif x >= press_line and _sb_is_defensive_action(e):
            def_actions[team] += 1

    def ppda(team: str, opponent: str) -> float | None:
        actions = def_actions[team]
        return float(passes[opponent]) / actions if actions > 0 else None

    return [
        {"team": home_team, "date": match_date, "ppda": ppda(home_team, away_team)},
        {"team": away_team, "date": match_date, "ppda": ppda(away_team, home_team)},
    ]


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


def parse_eloratings_team_codes(teams_tsv: str) -> dict[str, str]:
    """Build a ``team-code -> English name`` map from EloRatings.net ``en.teams.tsv``.

    Each line is tab-separated ``CODE\\tName[\\tAlternateName...]``; the first name column is
    the canonical English spelling (alternate columns are short/casual variants and are
    ignored — alias normalization happens downstream at ingest). Blank or malformed lines
    (no code or no name) are skipped. Codes are taken verbatim, including the rare
    non-2-letter ones (e.g. ``US_loc``), which simply never match a ``World.tsv`` row.
    """

    mapping: dict[str, str] = {}
    for line in teams_tsv.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, name = parts[0].strip(), parts[1].strip()
        if code and name:
            mapping[code] = name
    return mapping


def parse_eloratings_tsv(world_tsv: str, teams_tsv: str, as_of: date) -> pl.DataFrame:
    """Extract ``(team, date, rating)`` rows from EloRatings.net ``World.tsv`` + ``en.teams.tsv``.

    ``World.tsv`` is the backend ranking feed for the JS-rendered ranking page: each line is
    tab-separated with the 2-letter team **code** in column 3 (index 2) and the current Elo
    rating in column 4 (index 3). Codes are resolved to English names via
    :func:`parse_eloratings_team_codes`; rows whose code is unknown or whose rating won't
    parse as a number are skipped. Team names are returned as-is (the site's English
    spelling) and canonicalized through the alias map downstream at ingest time. All rows are
    stamped with ``as_of``.
    """

    codes = parse_eloratings_team_codes(teams_tsv)
    teams: list[str] = []
    ratings: list[float] = []
    for line in world_tsv.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name = codes.get(parts[2].strip())
        if not name:
            continue
        try:
            rating = float(parts[3].strip())
        except ValueError:
            continue
        teams.append(name)
        ratings.append(rating)

    return pl.DataFrame(
        {
            "team": teams,
            "date": [as_of] * len(teams),
            "rating": ratings,
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
# openfootball 2026 World Cup (schedule + venue coordinates)
# ---------------------------------------------------------------------------

#: One coordinate of a ``"LAT LON"`` pair, e.g. ``49°16'36"N`` or ``37.403°N`` or
#: ``40°48'48.7"N``. Degrees are mandatory; minutes/seconds (with optional decimals) and the
#: N/S/E/W hemisphere letter are optional. Both ASCII (``'`` ``"``) and unicode (``′`` ``″``)
#: minute/second marks are accepted.
_COORD_RE = re.compile(
    r"(?P<deg>\d+(?:\.\d+)?)°"
    r"(?:(?P<min>\d+(?:\.\d+)?)['′])?"
    r"(?:(?P<sec>\d+(?:\.\d+)?)[\"″])?"
    r"\s*(?P<hemi>[NSEWnsew])?"
)


def _parse_one_coord(token: str) -> float | None:
    """Parse a single DMS-or-decimal coordinate token into signed decimal degrees."""

    match = _COORD_RE.search(token)
    if match is None:
        return None
    deg = float(match.group("deg"))
    deg += float(match.group("min") or 0.0) / 60.0
    deg += float(match.group("sec") or 0.0) / 3600.0
    hemi = (match.group("hemi") or "").upper()
    if hemi in ("S", "W"):
        deg = -deg
    return deg


def parse_geo_coords(text: str | None) -> tuple[float | None, float | None]:
    """Parse an openfootball ``coords`` string (``"<lat> <lon>"``) into ``(lat, lon)``.

    Handles both DMS (``49°16'36"N 123°6'43"W``) and decimal-degree (``37.403°N 121.970°W``)
    forms, including DMS with a decimal seconds component. Returns ``(None, None)`` when the
    string is empty or does not contain two parseable coordinates.
    """

    if not text:
        return None, None
    parts = text.split()
    if len(parts) < 2:
        return None, None
    return _parse_one_coord(parts[0]), _parse_one_coord(parts[1])


def normalize_openfootball_stadiums(stadiums: list[dict[str, object]]) -> pl.DataFrame:
    """Normalize openfootball ``stadiums`` dicts into the ``venues`` schema.

    Each raw dict carries ``name`` (stadium), ``city`` (the host-city string, e.g.
    ``"Boston (Foxborough)"`` — kept verbatim so it joins the schedule's ``ground``), ``cc``
    (ISO country code), and ``coords`` (a DMS-or-decimal ``"<lat> <lon>"`` string parsed by
    :func:`parse_geo_coords`). Venues whose coordinates do not parse are dropped (a venue with
    no usable location cannot contribute a travel distance).
    """

    rows: list[dict[str, object]] = []
    for s in stadiums:
        city = str(s.get("city") or "").strip()
        if not city:
            continue
        lat, lon = parse_geo_coords(str(s.get("coords")) if s.get("coords") else None)
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "venue": str(s.get("name") or "").strip(),
                "city": city,
                "country": (str(s.get("cc")).strip() or None) if s.get("cc") else None,
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "venue": pl.Utf8,
            "city": pl.Utf8,
            "country": pl.Utf8,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
        },
    ).select(TABLE_COLUMNS[Table.VENUES])


#: City names are matched case-folded; this keeps only Latin-script aliases (after
#: lowercasing) so transliterations like "Munich"/"Cologne" survive while non-Latin
#: alternatenames (Cyrillic/Arabic/CJK) are dropped — bounding the exploded gazetteer.
_LATIN_CITY_RE = r"^[a-z0-9 .'\-]+$"


def normalize_geonames_cities(raw: pl.DataFrame) -> pl.DataFrame:
    """Normalize a GeoNames ``cities*`` dump into the ``city_coords`` gazetteer schema.

    Produces one ``(city, country, latitude, longitude, population)`` row per *name alias* of
    each city — its ``name``, ``asciiname``, and every Latin-script ``alternatenames`` entry —
    so a match's English city string resolves even when it differs from the local spelling
    (``"Munich"`` -> München). ``city`` is lower-cased and ``country`` is the ISO-2 code; rows
    are de-duplicated on ``(city, country)`` keeping the highest population (the resolver later
    breaks tournament-host ambiguities by population). Empty input yields the empty schema.
    """

    if raw.is_empty():
        return pl.DataFrame(
            schema={
                "city": pl.Utf8, "country": pl.Utf8, "latitude": pl.Float64,
                "longitude": pl.Float64, "population": pl.Int64,
            }
        ).select(TABLE_COLUMNS[Table.CITY_COORDS])

    base = raw.select(
        pl.col("name").cast(pl.Utf8),
        pl.col("asciiname").cast(pl.Utf8),
        pl.col("alternatenames").cast(pl.Utf8),
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
        pl.col("country_code").cast(pl.Utf8).alias("country"),
        pl.col("population").cast(pl.Int64, strict=False).fill_null(0),
    ).drop_nulls(["latitude", "longitude"])

    def _alias(expr: pl.Expr) -> pl.DataFrame:
        return base.select(
            expr.alias("city"), "country", "latitude", "longitude", "population"
        )

    aliases = pl.concat(
        [
            _alias(pl.col("name")),
            _alias(pl.col("asciiname")),
            base.select(
                pl.col("alternatenames").str.split(",").alias("city"),
                "country", "latitude", "longitude", "population",
            ).explode("city"),
        ],
        how="vertical",
    ).with_columns(pl.col("city").str.strip_chars().str.to_lowercase())

    aliases = aliases.filter(
        (pl.col("city").str.len_chars() > 0) & pl.col("city").str.contains(_LATIN_CITY_RE)
    )
    # Highest-population entry wins each (city, country) collision.
    aliases = aliases.sort("population", descending=True).unique(
        subset=["city", "country"], keep="first"
    )
    return aliases.select(TABLE_COLUMNS[Table.CITY_COORDS])


def normalize_openfootball_schedule(matches: list[dict[str, object]]) -> pl.DataFrame:
    """Normalize openfootball ``matches`` dicts into the ``schedule`` schema.

    Each raw dict carries ``round`` (e.g. ``"Matchday 1"`` / ``"Round of 32"`` → ``stage``),
    ``date``, ``team1`` / ``team2`` (real nations for group games, bracket placeholders such
    as ``"2A"`` for knockouts), an optional ``group`` (``"Group A"`` → ``"A"``; absent for
    knockouts), and ``ground`` (the host-city string → ``city``, joining the ``venues``
    table). Team names are canonicalized via :func:`normalize_team_expr` so group-stage
    fixtures join the matches / Elo tables; placeholders pass through unchanged. ``match_id``
    follows the ``date__home__away`` convention shared with the matches and odds tables.

    Rows missing a date or either team are dropped.
    """

    from polymbappe.data.aliases import normalize_team_expr

    rows = [
        {
            "stage": str(m.get("round") or "").strip(),
            "date": str(m.get("date") or "").strip(),
            "group": str(m.get("group")).strip() if m.get("group") else None,
            "home_team": str(m.get("team1") or "").strip(),
            "away_team": str(m.get("team2") or "").strip(),
            "city": str(m.get("ground") or "").strip(),
            # Official FIFA match number (73-104 for the knockout tree). It disambiguates
            # same-date fixtures when wiring W##/L## placeholders; group fixtures may omit it.
            "match_number": int(m["num"]) if m.get("num") is not None else None,
        }
        for m in matches
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "stage": pl.Utf8,
            "date": pl.Utf8,
            "group": pl.Utf8,
            "home_team": pl.Utf8,
            "away_team": pl.Utf8,
            "city": pl.Utf8,
            "match_number": pl.Int32,
        },
    )
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("match_id")).select(
            TABLE_COLUMNS[Table.SCHEDULE]
        )

    df = df.with_columns(
        pl.col("date").str.to_date(strict=False).alias("date"),
        # Drop the "Group " prefix so the label matches the draw config (A..L).
        pl.col("group").str.replace(r"(?i)^group\s+", "").alias("group"),
        normalize_team_expr("home_team").alias("home_team"),
        normalize_team_expr("away_team").alias("away_team"),
    )
    df = df.filter(
        pl.col("date").is_not_null()
        & (pl.col("home_team").str.len_chars() > 0)
        & (pl.col("away_team").str.len_chars() > 0)
    )
    df = df.with_columns(
        pl.format("{}__{}__{}", pl.col("date"), pl.col("home_team"), pl.col("away_team"))
        .alias("match_id")
    )
    return df.select(TABLE_COLUMNS[Table.SCHEDULE])


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
