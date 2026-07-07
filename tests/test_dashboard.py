"""Tests for the pure dashboard data-access layer (spec sections 6.2 & 11).

Exercises :mod:`polymbappe.dashboard.data` only — no ``streamlit``/``plotly`` is
imported or tested here (those are optional, lazily-imported deps). Covers the
graceful empty-frame-on-missing-file contract and the helper functions.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from polymbappe.config import Settings
from polymbappe.dashboard import data


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


# -- empty-frame-on-missing-file contract -------------------------------------


def test_load_stage_probabilities_empty_schema(tmp_path: Path) -> None:
    df = data.load_stage_probabilities(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.STAGE_SCHEMA.keys())


def test_load_group_probabilities_empty_schema(tmp_path: Path) -> None:
    df = data.load_group_probabilities(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.GROUP_SCHEMA.keys())


def test_load_match_predictions_empty_schema(tmp_path: Path) -> None:
    df = data.load_match_predictions(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.MATCH_SCHEMA.keys())


def test_load_edges_empty_schema(tmp_path: Path) -> None:
    df = data.load_edges(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.EDGES_SCHEMA.keys())


def test_load_agent_changelog_empty_schema(tmp_path: Path) -> None:
    df = data.load_agent_changelog(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.CHANGELOG_SCHEMA.keys())


# -- round-trip: written artifacts are read back ------------------------------


def _write_stage(settings: Settings) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "team": ["BRA", "ARG", "FRA"],
            "R32": [1.0, 1.0, 1.0],
            "R16": [0.9, 0.8, 0.85],
            "QF": [0.7, 0.6, 0.65],
            "SF": [0.5, 0.4, 0.45],
            "FINAL": [0.3, 0.25, 0.28],
            "champion": [0.20, 0.15, 0.18],
        }
    )
    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(settings.outputs_data_dir / "stage_probabilities.parquet")
    return df


def test_load_stage_probabilities_reads_written_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    written = _write_stage(settings)
    loaded = data.load_stage_probabilities(settings)
    assert loaded.height == written.height
    assert set(loaded["team"]) == {"BRA", "ARG", "FRA"}


# -- helpers ------------------------------------------------------------------


def test_top_contenders_sorts_and_limits() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "champion": [0.1, 0.4, 0.2, 0.3],
        }
    )
    top = data.top_contenders(df, n=2)
    assert top["team"].to_list() == ["B", "D"]


def test_top_contenders_empty_frame_passthrough() -> None:
    empty = data.load_stage_probabilities(Settings(data_dir=Path("/nonexistent-xyz")))
    assert data.top_contenders(empty).is_empty()


def test_available_teams_sorted_unique() -> None:
    df = pl.DataFrame({"team": ["C", "A", "B", "A"], "champion": [0.1, 0.2, 0.3, 0.2]})
    assert data.available_teams(df) == ["A", "B", "C"]


def test_team_stage_row_maps_stages() -> None:
    df = pl.DataFrame(
        {
            "team": ["BRA"],
            "R32": [1.0],
            "R16": [0.9],
            "QF": [0.7],
            "SF": [0.5],
            "FINAL": [0.3],
            "champion": [0.2],
        }
    )
    row = data.team_stage_row(df, "BRA")
    assert row["R32"] == 1.0
    assert row["champion"] == 0.2
    assert list(row.keys()) == list(data.STAGE_COLUMNS)


def test_team_stage_row_missing_team() -> None:
    df = pl.DataFrame({"team": ["BRA"], "R32": [1.0], "R16": [0.9], "QF": [0.7],
                       "SF": [0.5], "FINAL": [0.3], "champion": [0.2]})
    assert data.team_stage_row(df, "NOPE") == {}


def test_match_row_lookup() -> None:
    df = pl.DataFrame(
        {
            "match_id": ["A-0"],
            "group": ["A"],
            "home_team": ["BRA"],
            "away_team": ["ARG"],
            "model_home": [0.5],
            "model_draw": [0.25],
            "model_away": [0.25],
        }
    )
    record = data.match_row(df, "BRA", "ARG")
    assert record is not None
    assert record["model_home"] == 0.5
    assert data.match_row(df, "ARG", "BRA") is None


def test_upset_candidates_without_elo_ranks_by_r16() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "B", "C"],
            "R16": [0.2, 0.8, 0.5],
            "champion": [0.01, 0.05, 0.02],
        }
    )
    result = data.upset_candidates(df, elo=None, n=2)
    assert result["team"].to_list() == ["B", "C"]


def test_upset_candidates_with_elo_filters_and_scores() -> None:
    df = pl.DataFrame(
        {
            "team": ["Strong", "Weak", "Mid"],
            "R16": [0.9, 0.6, 0.7],
            "champion": [0.3, 0.01, 0.05],
        }
    )
    elo = {"Strong": 2000.0, "Weak": 1500.0, "Mid": 1850.0}
    # Strong has zero deficit (excluded); Weak has a 500 deficit (kept).
    result = data.upset_candidates(df, elo=elo, min_elo_gap=300.0)
    assert "Strong" not in result["team"].to_list()
    assert "Weak" in result["team"].to_list()
    assert "elo_gap" in result.columns
    assert "upset_score" in result.columns


def test_upset_candidates_empty_passthrough() -> None:
    empty = data.load_stage_probabilities(Settings(data_dir=Path("/nonexistent-xyz")))
    assert data.upset_candidates(empty).is_empty()


def test_edges_by_priority_orders_by_magnitude_times_kelly() -> None:
    df = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "outcome": ["home", "away"],
            "model_prob": [0.6, 0.7],
            "market_prob": [0.5, 0.5],
            "edge": [0.1, 0.2],
            "edge_bps": [1000.0, 2000.0],
            "kelly_fraction": [0.5, 0.1],
        }
    )
    # m1: 1000 * 0.5 = 500; m2: 2000 * 0.1 = 200 -> m1 first.
    result = data.edges_by_priority(df)
    assert result["match_id"].to_list() == ["m1", "m2"]
    assert "priority" in result.columns


def test_data_freshness_reports_missing_and_present(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_stage(settings)
    freshness = data.data_freshness(settings)
    assert freshness["edges.parquet"] == "missing"
    assert freshness["stage_probabilities.parquet"] != "missing"


# -- recorded results & fixture splitting -------------------------------------


def test_load_recorded_results_empty_schema(tmp_path: Path) -> None:
    df = data.load_recorded_results(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.RESULTS_SCHEMA.keys())


def _fixtures() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": ["2026__BRA__SRB", "2026__ARG__MEX", "2026__FRA__CAN"],
            "group": ["A", "B", "C"],
            "home_team": ["BRA", "ARG", "FRA"],
            "away_team": ["SRB", "MEX", "CAN"],
            "model_home": [0.6, 0.5, 0.7],
            "model_draw": [0.25, 0.3, 0.2],
            "model_away": [0.15, 0.2, 0.1],
        }
    )


def _results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "home_team": ["BRA", "BRA", "FRA"],
            "away_team": ["SRB", "SRB", "CAN"],
            # An old friendly (2018) plus the actual 2026 fixture for BRA-SRB.
            "date": [date(2018, 6, 27), date(2026, 6, 12), date(2026, 6, 13)],
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 1],
            "competition": ["Friendly", "FIFA World Cup", "FIFA World Cup"],
        }
    )


def test_tournament_results_filters_by_year() -> None:
    filtered = data.tournament_results(_results(), year=2026)
    assert filtered.height == 2
    assert set(filtered["date"].dt.year().to_list()) == {2026}


def test_tournament_results_competition_substr() -> None:
    filtered = data.tournament_results(_results(), year=2026, competition_substr="world cup")
    assert filtered.height == 2
    assert all("World Cup" in c for c in filtered["competition"].to_list())


def test_tournament_results_empty_passthrough() -> None:
    empty = data.load_recorded_results(Settings(data_dir=Path("/nonexistent-xyz")))
    assert data.tournament_results(empty).is_empty()


def test_split_fixtures_partitions_upcoming_and_finished() -> None:
    results = data.tournament_results(_results(), year=2026)
    upcoming, finished = data.split_fixtures(_fixtures(), results)

    # ARG-MEX has no recorded result -> upcoming; BRA-SRB and FRA-CAN played -> finished.
    assert upcoming["match_id"].to_list() == ["2026__ARG__MEX"]
    assert set(finished["match_id"]) == {"2026__BRA__SRB", "2026__FRA__CAN"}
    assert "model_pick" in upcoming.columns


def test_split_fixtures_uses_latest_result_and_flags_correctness() -> None:
    results = data.tournament_results(_results(), year=2026)
    _, finished = data.split_fixtures(_fixtures(), results)

    bra = finished.filter(pl.col("match_id") == "2026__BRA__SRB").row(0, named=True)
    # The 2026 result (1-1 draw) is used, not the 2018 friendly (2-0).
    assert bra["home_goals"] == 1 and bra["away_goals"] == 1
    assert bra["actual_outcome"] == "draw"
    # Model favoured BRA (home) but it was a draw -> incorrect.
    assert bra["model_pick"] == "home"
    assert bra["model_correct"] is False

    fra = finished.filter(pl.col("match_id") == "2026__FRA__CAN").row(0, named=True)
    # FRA lost 0-1 -> away; model favoured FRA -> incorrect.
    assert fra["actual_outcome"] == "away"
    assert fra["model_correct"] is False


def test_split_fixtures_no_results_all_upcoming() -> None:
    empty = data.load_recorded_results(Settings(data_dir=Path("/nonexistent-xyz")))
    upcoming, finished = data.split_fixtures(_fixtures(), empty)
    assert upcoming.height == 3
    assert finished.is_empty()


def test_split_fixtures_empty_fixtures_passthrough() -> None:
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    upcoming, finished = data.split_fixtures(empty, empty)
    assert upcoming.is_empty()
    assert finished.is_empty()


def _finished() -> pl.DataFrame:
    """The finished frame for the canonical fixtures+results fixtures (both incorrect)."""
    results = data.tournament_results(_results(), year=2026)
    _, finished = data.split_fixtures(_fixtures(), results)
    return finished


def test_prediction_scorecard_metrics() -> None:
    import math

    scorecard = data.prediction_scorecard(_finished())
    assert scorecard["n"] == 2.0
    # Both finished matches were model misses (BRA draw, FRA loss).
    assert scorecard["accuracy"] == 0.0
    # BRA-SRB draw: (.6)^2+(.25-1)^2+(.15)^2 = 0.945; FRA-CAN away: (.7)^2+(.2)^2+(.1-1)^2 = 1.34.
    assert scorecard["brier_score"] == pytest.approx((0.945 + 1.34) / 2)
    # log loss = mean(-log P(actual)) = mean(-log .25, -log .1).
    expected_log = (-math.log(0.25) - math.log(0.1)) / 2
    assert scorecard["log_loss"] == pytest.approx(expected_log)


def test_prediction_scorecard_empty_zeroed() -> None:
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    scorecard = data.prediction_scorecard(empty)
    assert scorecard == {"n": 0.0, "accuracy": 0.0, "brier_score": 0.0, "log_loss": 0.0, "rps": 0.0}


def test_accuracy_by_outcome_groups_and_scores() -> None:
    by_outcome = data.accuracy_by_outcome(_finished())
    # One draw (BRA) and one away (FRA), both missed -> accuracy 0 in each group.
    assert by_outcome["actual_outcome"].to_list() == ["away", "draw"]
    assert by_outcome["n"].to_list() == [1, 1]
    assert by_outcome["hits"].to_list() == [0, 0]
    assert by_outcome["accuracy"].to_list() == [0.0, 0.0]


def test_accuracy_by_outcome_empty_schema() -> None:
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    by_outcome = data.accuracy_by_outcome(empty)
    assert by_outcome.is_empty()
    assert by_outcome.columns == list(data.OUTCOME_ACCURACY_SCHEMA.keys())


def test_calibration_bins_buckets_confidence() -> None:
    bins = data.calibration_bins(_finished(), n_bins=5)
    # Favourite confidences are 0.6 and 0.7 -> both fall in the [0.6, 0.8) bucket.
    assert bins.height == 1
    row = bins.row(0, named=True)
    assert row["bin_lower"] == pytest.approx(0.6)
    assert row["bin_upper"] == pytest.approx(0.8)
    assert row["mean_confidence"] == pytest.approx(0.65)
    assert row["hit_rate"] == 0.0  # both misses
    assert row["count"] == 2


def test_calibration_bins_empty_schema() -> None:
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    bins = data.calibration_bins(empty)
    assert bins.is_empty()
    assert bins.columns == list(data.CALIBRATION_SCHEMA.keys())
