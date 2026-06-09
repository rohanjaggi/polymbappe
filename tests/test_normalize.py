from datetime import date

import polars as pl
import pytest
from bs4 import BeautifulSoup

from polymbappe.data.normalize import (
    implied_probabilities,
    make_match_id,
    normalize_kaggle_results,
    normalize_odds_frame,
    parse_eloratings,
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
