from datetime import date

import polars as pl
import pytest
from bs4 import BeautifulSoup

from polymbappe.data.normalize import (
    implied_probabilities,
    infer_knockout_stage,
    make_match_id,
    normalize_geonames_cities,
    normalize_kaggle_results,
    normalize_odds_frame,
    normalize_openfootball_schedule,
    normalize_openfootball_stadiums,
    parse_eloratings,
    parse_geo_coords,
    slugify,
)
from polymbappe.data.tables import TABLE_COLUMNS, Table
from polymbappe.polymarket.adapter import parse_market_outcomes


def test_slugify_and_match_id() -> None:
    assert slugify("South Korea") == "south_korea"
    assert make_match_id(date(2022, 11, 22), "Saudi Arabia", "Argentina") == (
        "2022-11-22__saudi_arabia__argentina"
    )


def test_normalize_kaggle_results_shape_and_filtering() -> None:
    raw = pl.DataFrame(
        {
            "date": ["2018-06-14", "2026-07-01"],  # second is an unplayed fixture
            "home_team": ["Russia", "Spain"],
            "away_team": ["Saudi Arabia", "Brazil"],
            "home_score": [5, None],
            "away_score": [0, None],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "city": ["Moscow", "Dallas"],
            "country": ["Russia", "USA"],
            "neutral": [False, True],
        }
    )
    out = normalize_kaggle_results(raw)

    assert out.columns == list(TABLE_COLUMNS[Table.MATCHES])
    # Unplayed fixture (null score) dropped.
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["match_id"] == "2018-06-14__Russia__Saudi Arabia"
    assert row["home_goals"] == 5 and row["away_goals"] == 0
    assert row["competition"] == "FIFA World Cup"
    assert row["neutral_site"] is False
    assert row["is_knockout"] is False
    assert row["group"] is None
    assert out.schema["date"] == pl.Date


def _wc_edition_rows() -> list[tuple[str, str, str, str]]:
    """A small FIFA World Cup edition: a 4-team round-robin group then a single final.

    Each group team plays 3 games (group_size == 3); the final is each finalist's 4th
    appearance and is the only knockout match.
    """

    return [
        ("2018-06-15", "FIFA World Cup", "A", "B"),
        ("2018-06-15", "FIFA World Cup", "C", "D"),
        ("2018-06-19", "FIFA World Cup", "A", "C"),
        ("2018-06-19", "FIFA World Cup", "B", "D"),
        ("2018-06-23", "FIFA World Cup", "A", "D"),
        ("2018-06-23", "FIFA World Cup", "B", "C"),
        ("2018-07-15", "FIFA World Cup", "A", "B"),  # final -> knockout
    ]


def _rows_to_frame(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date.fromisoformat(d) for d, _, _, _ in rows],
            "competition": [c for _, c, _, _ in rows],
            "home_team": [h for _, _, h, _ in rows],
            "away_team": [a for _, _, _, a in rows],
        }
    )


def test_infer_knockout_flags_only_post_group_matches() -> None:
    rows = _wc_edition_rows()
    # A friendly between the same teams must never be flagged (non-major competition).
    rows.append(("2018-03-01", "Friendly", "A", "B"))
    # A round-robin-only major edition (every team plays twice, no knockout) -> all False.
    rows += [
        ("2021-06-12", "UEFA Euro", "E", "F"),
        ("2021-06-16", "UEFA Euro", "F", "E"),
    ]
    frame = _rows_to_frame(rows)

    flags = infer_knockout_stage(frame)
    out = frame.with_columns(flags).sort("date")

    knockout = out.filter(pl.col("is_knockout"))
    assert knockout.height == 1
    final = knockout.row(0, named=True)
    assert (final["competition"], final["home_team"], final["away_team"]) == (
        "FIFA World Cup",
        "A",
        "B",
    )
    assert final["date"] == date(2018, 7, 15)
    # Everything else (group games, friendly, round-robin Euro) stays group-stage.
    assert out.filter(~pl.col("is_knockout")).height == len(rows) - 1


def test_infer_knockout_labels_every_bracket_round() -> None:
    # Two groups of 4 (each team plays 3 group games; C/D/G/H are eliminated and play only
    # 3, fixing group_size at 3), then SF -> Final + third place. Every knockout round must
    # be flagged, not just the final.
    wc = "FIFA World Cup"
    group = [
        ("2022-06-15", wc, "A", "B"), ("2022-06-15", wc, "C", "D"),
        ("2022-06-15", wc, "E", "F"), ("2022-06-15", wc, "G", "H"),
        ("2022-06-19", wc, "A", "C"), ("2022-06-19", wc, "B", "D"),
        ("2022-06-19", wc, "E", "G"), ("2022-06-19", wc, "F", "H"),
        ("2022-06-23", wc, "A", "D"), ("2022-06-23", wc, "B", "C"),
        ("2022-06-23", wc, "E", "H"), ("2022-06-23", wc, "F", "G"),
    ]
    knockout = [
        ("2022-07-05", wc, "A", "F"),  # SF
        ("2022-07-05", wc, "E", "B"),  # SF
        ("2022-07-09", wc, "F", "B"),  # third place
        ("2022-07-10", wc, "A", "E"),  # final
    ]
    frame = _rows_to_frame(group + knockout)

    out = frame.with_columns(infer_knockout_stage(frame))
    flagged = {
        (r["home_team"], r["away_team"])
        for r in out.filter(pl.col("is_knockout")).iter_rows(named=True)
    }
    assert flagged == {("A", "F"), ("E", "B"), ("F", "B"), ("A", "E")}
    assert out.filter(~pl.col("is_knockout")).height == len(group)


def test_infer_knockout_empty_frame() -> None:
    empty = _rows_to_frame([]).clear()
    flags = infer_knockout_stage(empty)
    assert flags.len() == 0
    assert flags.dtype == pl.Boolean


def test_implied_probabilities_remove_overround() -> None:
    p_h, p_d, p_a = implied_probabilities(2.0, 3.0, 4.0)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-12
    # Favorite (lowest odds) carries the highest probability.
    assert p_h > p_d > p_a

    with pytest.raises(ValueError):
        implied_probabilities(0.0, 3.0, 4.0)


def test_normalize_odds_frame_sums_to_one_per_row() -> None:
    raw = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "B365H": [1.5, 2.5],
            "B365D": [4.0, 3.2],
            "B365A": [7.0, 2.8],
        }
    )
    out = normalize_odds_frame(
        raw,
        source="B365",
        home_col="B365H",
        draw_col="B365D",
        away_col="B365A",
        timestamp_col=None,
    )
    assert out.columns == list(TABLE_COLUMNS[Table.MARKET_ODDS])
    totals = (
        out.select(pl.col("home_win_prob") + pl.col("draw_prob") + pl.col("away_win_prob"))
        .to_series()
        .to_list()
    )
    assert all(abs(t - 1.0) < 1e-12 for t in totals)
    assert out["source"].to_list() == ["B365", "B365"]


def test_parse_eloratings_extracts_team_and_rating() -> None:
    html = """
    <table>
      <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
      <tr><td>1</td><td><a href='/brazil'>Brazil</a></td><td>2169</td></tr>
      <tr><td>2</td><td><a href='/argentina'>Argentina</a></td><td>2145</td></tr>
      <tr><td>—</td><td>header row no anchor</td><td>x</td></tr>
    </table>
    """
    out = parse_eloratings(BeautifulSoup(html, "html.parser"), as_of=date(2026, 6, 1))
    assert out.height == 2
    assert out["team"].to_list() == ["Brazil", "Argentina"]
    assert out["rating"].to_list() == [2169.0, 2145.0]
    assert out["date"].to_list() == [date(2026, 6, 1), date(2026, 6, 1)]


def test_parse_eloratings_team_codes_maps_code_to_first_name() -> None:
    from polymbappe.data.normalize import parse_eloratings_team_codes

    teams_tsv = (
        "AR\tArgentina\n"
        "AG\tAntigua and Barbuda\tAntigua & Barbuda\tAntigua/Barb\n"  # alt names ignored
        "US\tUSA\n"
        "\n"  # blank line skipped
        "BADLINE_NO_TAB\n"  # no name column -> skipped
    )
    codes = parse_eloratings_team_codes(teams_tsv)
    assert codes == {"AR": "Argentina", "AG": "Antigua and Barbuda", "US": "USA"}


def test_parse_eloratings_tsv_joins_codes_and_ratings() -> None:
    from polymbappe.data.normalize import parse_eloratings_tsv

    # World.tsv: code in column 3 (index 2), rating in column 4 (index 3).
    world_tsv = (
        "1\t1\tBR\t2169\t1\t2200\n"
        "2\t2\tAR\t2145\t1\t2180\n"
        "3\t3\tZZ\t1500\t1\t1500\n"  # unknown code -> dropped
        "4\t4\tNR\tnope\t1\t1\n"  # unparseable rating -> dropped
        "short\tline\n"  # too few columns -> dropped
    )
    teams_tsv = "BR\tBrazil\nAR\tArgentina\nNR\tNoRating\n"
    out = parse_eloratings_tsv(world_tsv, teams_tsv, as_of=date(2026, 6, 1))
    assert out.height == 2
    assert out["team"].to_list() == ["Brazil", "Argentina"]
    assert out["rating"].to_list() == [2169.0, 2145.0]
    assert out["date"].to_list() == [date(2026, 6, 1), date(2026, 6, 1)]


def test_parse_eloratings_tsv_empty_when_no_codes_match() -> None:
    from polymbappe.data.normalize import parse_eloratings_tsv

    # A code dictionary that matches nothing in World.tsv -> empty frame (ingest then
    # self-computes), with the correct schema preserved.
    out = parse_eloratings_tsv("1\t1\tBR\t2169\n", "US\tUSA\n", as_of=date(2026, 6, 1))
    assert out.is_empty()
    assert out.schema == {"team": pl.Utf8, "date": pl.Date, "rating": pl.Float64}


def test_parse_market_outcomes_handles_json_arrays() -> None:
    raw = {
        "id": "0xabc",
        "question": "Will Brazil win the 2026 World Cup?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.18", "0.82"]',
    }
    rows = parse_market_outcomes(raw)
    assert len(rows) == 2
    assert rows[0] == {
        "market_id": "0xabc",
        "question": "Will Brazil win the 2026 World Cup?",
        "outcome": "Yes",
        "price": 0.18,
    }
    # Mismatched lengths -> empty.
    assert parse_market_outcomes({"outcomes": '["Yes"]', "outcomePrices": '["0.1","0.9"]'}) == []


# -- openfootball venues + schedule --------------------------------------------

def test_parse_geo_coords_dms_and_decimal() -> None:
    # DMS form (Vancouver, BC Place) -> ~(49.277, -123.112).
    lat, lon = parse_geo_coords("49°16'36\"N 123°6'43\"W")
    assert lat is not None and lon is not None
    assert lat == pytest.approx(49.2767, abs=1e-3)
    assert lon == pytest.approx(-123.1119, abs=1e-3)

    # Decimal-degree form (Levi's Stadium).
    assert parse_geo_coords("37.403°N 121.970°W") == pytest.approx((37.403, -121.97))

    # DMS with a decimal seconds component (MetLife Stadium).
    lat, lon = parse_geo_coords("40°48'48.7\"N 74°4'27.7\"W")
    assert lat == pytest.approx(40.8135, abs=1e-3)
    assert lon == pytest.approx(-74.0744, abs=1e-3)

    # Unparseable / empty -> (None, None).
    assert parse_geo_coords(None) == (None, None)
    assert parse_geo_coords("not coords") == (None, None)


def test_normalize_openfootball_stadiums() -> None:
    stadiums = [
        {"city": "Mexico City", "cc": "mx", "name": "Estadio Azteca",
         "coords": "19°18'11\"N 99°09'02\"W"},
        {"city": "Los Angeles (Inglewood)", "cc": "us", "name": "SoFi Stadium",
         "coords": "33.953°N 118.339°W"},
        # Unparseable coords -> dropped.
        {"city": "Nowhere", "cc": "us", "name": "Ghost Stadium", "coords": ""},
    ]
    venues = normalize_openfootball_stadiums(stadiums)
    assert venues.columns == list(TABLE_COLUMNS[Table.VENUES])
    assert venues.height == 2  # the coordless venue is dropped
    azteca = venues.filter(pl.col("venue") == "Estadio Azteca").row(0, named=True)
    assert azteca["city"] == "Mexico City"  # host-city string kept verbatim
    assert azteca["country"] == "mx"
    assert azteca["latitude"] == pytest.approx(19.303, abs=1e-2)
    # The district qualifier is preserved so the schedule's ground joins it.
    assert "Los Angeles (Inglewood)" in venues["city"].to_list()


def test_normalize_openfootball_schedule() -> None:
    matches = [
        {"round": "Matchday 1", "date": "2026-06-11", "team1": "Mexico",
         "team2": "South Africa", "group": "Group A", "ground": "Mexico City"},
        # Knockout fixture: no group, bracket placeholders pass through.
        {"round": "Round of 32", "date": "2026-06-28", "team1": "2A", "team2": "2B",
         "ground": "Los Angeles (Inglewood)"},
        # Missing date -> dropped.
        {"round": "Matchday 1", "date": "", "team1": "Brazil", "team2": "Haiti",
         "group": "Group C", "ground": "Atlanta"},
    ]
    sched = normalize_openfootball_schedule(matches)
    assert sched.columns == list(TABLE_COLUMNS[Table.SCHEDULE])
    assert sched.height == 2  # the dateless row is dropped

    md1 = sched.filter(pl.col("stage") == "Matchday 1").row(0, named=True)
    assert md1["group"] == "A"  # "Group " prefix stripped
    assert md1["date"] == date(2026, 6, 11)
    assert md1["city"] == "Mexico City"
    assert md1["match_id"] == "2026-06-11__Mexico__South Africa"  # date__home__away

    ko = sched.filter(pl.col("stage") == "Round of 32").row(0, named=True)
    assert ko["group"] is None
    assert ko["home_team"] == "2A" and ko["away_team"] == "2B"


def test_normalize_openfootball_schedule_empty() -> None:
    empty = normalize_openfootball_schedule([])
    assert empty.columns == list(TABLE_COLUMNS[Table.SCHEDULE])
    assert empty.height == 0


# -- GeoNames city gazetteer ---------------------------------------------------

def _geonames_raw() -> pl.DataFrame:
    cols = (
        "geonameid name asciiname alternatenames latitude longitude feature_class "
        "feature_code country_code cc2 admin1_code admin2_code admin3_code admin4_code "
        "population elevation dem timezone modification_date"
    ).split()
    rows = [
        # Moscow RU (big) and Moscow US/Idaho (small) -> RU wins the bare name.
        ["1", "Moscow", "Moscow", "Moskva,Moscow", "55.7522", "37.6156", "P", "PPLC",
         "RU", "", "", "", "", "", "10381222", "", "", "", ""],
        ["2", "Moscow", "Moscow", "", "46.7324", "-117.0002", "P", "PPL", "US", "", "",
         "", "", "", "25000", "", "", "", ""],
        # München: accented name dropped by the Latin filter, but asciiname + "Munich"
        # alternatename survive.
        ["3", "München", "Munchen", "Munich,Muenchen,Москва-нет", "48.1374", "11.5755",
         "P", "PPLA", "DE", "", "", "", "", "", "1488202", "", "", "", ""],
    ]
    return pl.DataFrame(
        {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    )


def test_normalize_geonames_cities_aliases_and_population() -> None:
    g = normalize_geonames_cities(_geonames_raw())
    assert g.columns == list(TABLE_COLUMNS[Table.CITY_COORDS])
    # Keys are lower-cased; both Moscow entries kept (distinct countries).
    cities = set(g["city"].to_list())
    assert "moscow" in cities
    assert "munich" in cities  # English alternatename resolved
    assert "munchen" in cities  # asciiname resolved
    # Non-Latin alternatename was filtered out.
    assert all(c.isascii() for c in cities)
    # The RU Moscow (higher population) is retained for the RU row.
    ru = g.filter((pl.col("city") == "moscow") & (pl.col("country") == "RU")).row(0, named=True)
    assert ru["latitude"] == pytest.approx(55.7522)
    assert ru["population"] == 10381222


def test_normalize_geonames_cities_empty() -> None:
    empty = normalize_geonames_cities(pl.DataFrame())
    assert empty.columns == list(TABLE_COLUMNS[Table.CITY_COORDS])
    assert empty.height == 0


def test_normalize_kaggle_results_preserves_city_country() -> None:
    raw = pl.DataFrame(
        {
            "date": ["2014-06-12"],
            "home_team": ["Brazil"],
            "away_team": ["Croatia"],
            "home_score": [3],
            "away_score": [1],
            "tournament": ["FIFA World Cup"],
            "city": ["São Paulo"],
            "country": ["Brazil"],
            "neutral": [False],
        }
    )
    out = normalize_kaggle_results(raw)
    assert "city" in out.columns and "country" in out.columns
    row = out.row(0, named=True)
    assert row["city"] == "São Paulo"  # kept verbatim (accent-folded only at lookup time)
    assert row["country"] == "Brazil"


def test_normalize_kaggle_results_wc2026_ko_start_overrides_heuristic() -> None:
    """The schedule-derived KO start date labels WC2026 rows exactly, both directions:
    a matchday-3 false positive is cleared and an R32 false negative is set. Non-2026
    rows keep the structural heuristic's output."""

    raw = pl.DataFrame(
        {
            "date": ["2026-06-25", "2026-06-29", "2018-06-15"],
            "home_team": ["Japan", "Germany", "Portugal"],
            "away_team": ["Sweden", "Paraguay", "Spain"],
            "home_score": [1, 1, 3],
            "away_score": [0, 1, 3],
            "tournament": ["FIFA World Cup"] * 3,
            "city": ["Arlington"] * 3,
            "country": ["United States"] * 3,
            "neutral": [True] * 3,
        }
    )
    out = normalize_kaggle_results(raw, wc2026_ko_start=date(2026, 6, 28))
    flags = dict(zip(out["date"].to_list(), out["is_knockout"].to_list(), strict=True))
    assert flags[date(2026, 6, 25)] is False  # group stage regardless of heuristic
    assert flags[date(2026, 6, 29)] is True  # on/after KO start -> knockout
    assert flags[date(2018, 6, 15)] is False  # pre-2026 rows untouched by the override


def test_normalize_openfootball_schedule_carries_match_number() -> None:
    matches = [
        {"round": "Matchday 1", "date": "2026-06-11", "team1": "Mexico",
         "team2": "South Africa", "group": "Group A", "ground": "Mexico City"},
        {"round": "Round of 32", "date": "2026-06-28", "team1": "2A", "team2": "2B",
         "ground": "Los Angeles (Inglewood)", "num": 73},
    ]
    sched = normalize_openfootball_schedule(matches)
    assert "match_number" in sched.columns
    by_stage = {r["stage"]: r["match_number"] for r in sched.iter_rows(named=True)}
    assert by_stage["Round of 32"] == 73
    assert by_stage["Matchday 1"] is None  # group fixtures carry no number
