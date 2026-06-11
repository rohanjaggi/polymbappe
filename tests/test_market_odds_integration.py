"""Integration tests for market odds ingestion and edge detection."""

import polars as pl
import pytest

from polymbappe.data.normalize import normalize_footballdata_odds, normalize_odds_frame
from polymbappe.eval.market import compute_edges, kelly_fraction


@pytest.fixture
def sample_footballdata_csv() -> pl.DataFrame:
    """Minimal Football-Data.co.uk style CSV (3 matches, Bet365 odds)."""
    return pl.DataFrame(
        {
            "Date": ["14/06/2018", "15/06/2018", "15/06/2018"],
            "HomeTeam": ["Russia", "Egypt", "Morocco"],
            "AwayTeam": ["Saudi Arabia", "Uruguay", "Iran"],
            "FTHG": [5, 0, 0],
            "FTAG": [0, 1, 1],
            "B365H": [1.53, 3.40, 2.70],
            "B365D": [4.20, 3.30, 3.10],
            "B365A": [7.00, 2.15, 2.75],
        }
    )


def test_normalize_footballdata_produces_correct_schema(
    sample_footballdata_csv: pl.DataFrame,
) -> None:
    result = normalize_footballdata_odds(sample_footballdata_csv)
    assert set(result.columns) == {
        "match_id",
        "source",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "timestamp",
    }
    assert result.height == 3


def test_normalize_footballdata_devigged(
    sample_footballdata_csv: pl.DataFrame,
) -> None:
    result = normalize_footballdata_odds(sample_footballdata_csv)
    sums = (
        result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    ).to_list()
    for s in sums:
        assert abs(s - 1.0) < 1e-9, f"Devigged probs should sum to 1.0, got {s}"


def test_normalize_footballdata_match_id_format(
    sample_footballdata_csv: pl.DataFrame,
) -> None:
    result = normalize_footballdata_odds(sample_footballdata_csv)
    ids = result["match_id"].to_list()
    assert ids[0] == "2018-06-14__Russia__Saudi Arabia"


def test_normalize_odds_frame_drops_invalid_rows() -> None:
    raw = pl.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "home_odds": [1.5, 0.0, 2.0],
            "draw_odds": [3.5, 3.5, None],
            "away_odds": [5.0, 5.0, 3.0],
        }
    )
    result = normalize_odds_frame(
        raw, source="test", home_col="home_odds", draw_col="draw_odds", away_col="away_odds"
    )
    assert result.height == 1
    assert result["match_id"].to_list() == ["m1"]


def test_compute_edges_basic() -> None:
    model = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "model_home": [0.60, 0.40],
            "model_draw": [0.25, 0.30],
            "model_away": [0.15, 0.30],
        }
    )
    market = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_win_prob": [0.50, 0.42],
            "draw_prob": [0.28, 0.30],
            "away_win_prob": [0.22, 0.28],
        }
    )
    edges = compute_edges(model, market, threshold=0.05)
    assert edges.height > 0
    assert all(abs(e) > 500 for e in edges["edge_bps"].to_list())


def test_compute_edges_respects_threshold() -> None:
    model = pl.DataFrame(
        {"match_id": ["m1"], "model_home": [0.52], "model_draw": [0.26], "model_away": [0.22]}
    )
    market = pl.DataFrame(
        {"match_id": ["m1"], "home_win_prob": [0.50], "draw_prob": [0.28], "away_win_prob": [0.22]}
    )
    edges = compute_edges(model, market, threshold=0.05)
    assert edges.is_empty()


def test_compute_edges_no_join_returns_empty() -> None:
    model = pl.DataFrame(
        {"match_id": ["m1"], "model_home": [0.60], "model_draw": [0.25], "model_away": [0.15]}
    )
    market = pl.DataFrame(
        {"match_id": ["m99"], "home_win_prob": [0.50], "draw_prob": [0.28], "away_win_prob": [0.22]}
    )
    edges = compute_edges(model, market, threshold=0.01)
    assert edges.is_empty()


def test_kelly_fraction_positive_edge() -> None:
    frac = kelly_fraction(0.60, 0.50)
    assert frac > 0.0


def test_kelly_fraction_no_edge() -> None:
    frac = kelly_fraction(0.40, 0.50)
    assert frac == 0.0


def test_kelly_fraction_boundary() -> None:
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
