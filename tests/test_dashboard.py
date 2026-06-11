"""Tests for the pure dashboard data-access layer (spec sections 6.2 & 11).

Exercises :mod:`polymbappe.dashboard.data` only — no ``streamlit``/``plotly`` is
imported or tested here (those are optional, lazily-imported deps). Covers the
graceful empty-frame-on-missing-file contract and the helper functions.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

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
