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


def test_load_knockout_bracket_empty_schema(tmp_path: Path) -> None:
    df = data.load_knockout_bracket(_settings(tmp_path))
    assert df.is_empty()
    assert df.columns == list(data.KO_BRACKET_SCHEMA.keys())


def _write_bracket(settings: Settings) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "round": ["R32", "R16", "QF", "QF"],
            "match_number": [73, 89, 97, 97],
            "rank": [1, 1, 1, 2],
            "team_a": ["BRA", "BRA", "BRA", "ARG"],
            "team_b": ["SRB", "FRA", "ESP", "ESP"],
            "matchup_prob": [1.0, 1.0, 0.4, 0.35],
            "p_a_advance": [0.7, 0.55, 0.52, 0.48],
            "p_b_advance": [0.3, 0.45, 0.48, 0.52],
            "p_decided_reg": [0.72, 0.68, 0.6, 0.62],
            "p_decided_et": [0.18, 0.2, 0.24, 0.22],
            "p_decided_pens": [0.10, 0.12, 0.16, 0.16],
            "model_a": [0.5, 0.42, 0.4, 0.38],
            "model_draw": [0.25, 0.28, 0.3, 0.3],
            "model_b": [0.25, 0.30, 0.3, 0.32],
            "exp_a_goals": [1.6, 1.3, 1.4, 1.2],
            "exp_b_goals": [0.9, 1.2, 1.3, 1.3],
        }
    ).with_columns(pl.col("match_number").cast(pl.Int32), pl.col("rank").cast(pl.Int32))
    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(settings.outputs_data_dir / "knockout_bracket.parquet")
    return df


def test_load_knockout_bracket_reads_written_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_bracket(settings)
    loaded = data.load_knockout_bracket(settings)
    assert loaded.height == 4
    assert loaded.columns == list(data.KO_BRACKET_SCHEMA.keys())


def test_bracket_slots_and_candidates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_bracket(settings)
    bracket = data.load_knockout_bracket(settings)
    # One rank-1 fixture per slot in the QF (match 97), regardless of how many candidates.
    qf_slots = data.bracket_slots(bracket, "QF")
    assert qf_slots.height == 1
    assert qf_slots.row(0, named=True)["team_a"] == "BRA"  # the rank-1 (most-likely) pairing
    # All candidate matchups for that fixture, most-probable first.
    cands = data.bracket_slot_candidates(bracket, 97)
    assert cands.height == 2
    assert cands["rank"].to_list() == [1, 2]
    assert data.bracket_slots(bracket, "SF").is_empty()


def test_knockout_results_filters_to_played_knockouts() -> None:
    from datetime import date

    results = pl.DataFrame(
        {
            "match_id": ["ko1", "grp1", "old"],
            "date": [date(2026, 7, 5), date(2026, 6, 20), date(2018, 7, 5)],
            "home_team": ["BRA", "ARG", "FRA"],
            "away_team": ["SRB", "MEX", "CRO"],
            "home_goals": [2, 1, 4],
            "away_goals": [1, 1, 2],
            "competition": ["FIFA World Cup"] * 3,
            "is_knockout": [True, False, True],
            "neutral_site": [True, True, True],
            "group": [None, "C", None],
        },
        schema_overrides={"group": pl.Utf8},
    )
    ko = data.knockout_results(results)
    # Only the 2026 knockout game survives (group-stage and pre-2026 dropped).
    assert ko["match_id"].to_list() == ["ko1"]


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
    assert scorecard == {
        "n": 0.0, "accuracy": 0.0, "brier_score": 0.0, "log_loss": 0.0,
        "rps": 0.0, "rps_skill": 0.0, "log_loss_skill": 0.0, "brier_skill": 0.0,
    }


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


# -- extended scoring rules / calibration / segmentation ----------------------


def test_prediction_scorecard_includes_rps_and_skill() -> None:
    card = data.prediction_scorecard(_finished())
    # RPS present and matches the ordinal H-D-A computation on this frame.
    assert 0.0 <= card["rps"] <= 1.0
    # These fixtures are both misses, so the model trails a uniform guess -> negative skill.
    assert card["rps_skill"] <= 0.0
    for key in ("rps", "rps_skill", "log_loss_skill", "brier_skill"):
        assert key in card


def test_calibration_summary_keys_and_empty() -> None:
    summary = data.calibration_summary(_finished())
    assert set(summary) == {"n", "ece", "mce", "slope", "intercept"}
    assert summary["n"] == 2.0 and summary["ece"] >= 0.0
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    assert data.calibration_summary(empty)["n"] == 0.0


def test_competitive_subset_filters_by_favourite_band() -> None:
    # Favourite confidences are 0.6 and 0.7; only the 0.6 fixture is in [0.40, 0.60].
    subset = data.competitive_subset(_finished())
    assert subset.height == 1
    fav = max(
        subset.row(0, named=True)[c] for c in ("model_home", "model_draw", "model_away")
    )
    assert 0.40 <= fav <= 0.60


def test_rps_significance_contract() -> None:
    sig = data.rps_significance(_finished())
    assert set(sig) >= {"n", "mean_diff", "ci_low", "ci_high", "bootstrap_p", "wilcoxon_p"}
    assert sig["n"] == 2.0
    empty = data.load_match_predictions(Settings(data_dir=Path("/nonexistent-xyz")))
    assert data.rps_significance(empty)["n"] == 0.0


def test_bookmaker_comparison_unavailable_without_workbook(tmp_path: Path) -> None:
    cmp = data.bookmaker_comparison(
        _finished(), Settings(data_dir=tmp_path / "data"), path=tmp_path / "missing.xlsx"
    )
    assert cmp["available"] is False
    # Market-probability metrics are always stubbed with an explanatory reason.
    assert cmp["market_rps_skill"] is None
    assert cmp["roi_vs_closing"] is None
    assert "closing odds" in cmp["market_prob_reason"]


# -- knockout classification & bracket resolution ------------------------------
#
# Mini 8-team bracket: groups A–D each send their top two straight into the
# quarter-finals. Two ties finish level to exercise the extra-time/penalties
# inference paths: the winner must be deduced from the next round's fixtures
# (Ares) or, when the next round hasn't been played, from stage probabilities
# (Cato via FINAL == 0 for Ceres).

_GROUP_WINNERS = {"A": "Alpha", "B": "Bravo", "C": "Ceres", "D": "Delta"}
_GROUP_RUNNERS = {"A": "Ares", "B": "Boreas", "C": "Cato", "D": "Dione"}


def _ko_schedule() -> pl.DataFrame:
    rows = [
        # (date, stage, home_code, away_code) — enumeration order gives match numbers 73-80.
        (date(2026, 7, 9), "Quarter-final", "1A", "2B"),
        (date(2026, 7, 9), "Quarter-final", "1B", "2A"),
        (date(2026, 7, 10), "Quarter-final", "1C", "2D"),
        (date(2026, 7, 10), "Quarter-final", "1D", "2C"),
        (date(2026, 7, 14), "Semi-final", "W73", "W74"),
        (date(2026, 7, 15), "Semi-final", "W75", "W76"),
        (date(2026, 7, 18), "Match for third place", "L77", "L78"),
        (date(2026, 7, 19), "Final", "W77", "W78"),
    ]
    return pl.DataFrame(
        {
            "match_id": [f"ko{i}" for i in range(len(rows))],
            "date": [r[0] for r in rows],
            "stage": [r[1] for r in rows],
            "group": [None] * len(rows),
            "home_team": [r[2] for r in rows],
            "away_team": [r[3] for r in rows],
            "city": ["Testville"] * len(rows),
        },
        schema_overrides={"group": pl.Utf8},
    )


def _ko_match_df(extra_pairs: list[tuple[str, str]] | None = None) -> pl.DataFrame:
    group_rows = [
        (g, _GROUP_WINNERS[g], _GROUP_RUNNERS[g]) for g in ("A", "B", "C", "D")
    ]
    ko_pairs = [
        ("Alpha", "Boreas"), ("Bravo", "Ares"), ("Ceres", "Dione"), ("Delta", "Cato"),
        ("Alpha", "Ares"), ("Ceres", "Cato"),
    ] + (extra_pairs or [])
    return pl.DataFrame(
        {
            "match_id": [f"g{g}" for g, _, _ in group_rows]
            + [f"k{i}" for i in range(len(ko_pairs))],
            "group": [g for g, _, _ in group_rows] + ["KO"] * len(ko_pairs),
            "home_team": [h for _, h, _ in group_rows] + [h for h, _ in ko_pairs],
            "away_team": [a for _, _, a in group_rows] + [a for _, a in ko_pairs],
            "model_home": [0.5] * (len(group_rows) + len(ko_pairs)),
            "model_draw": [0.3] * (len(group_rows) + len(ko_pairs)),
            "model_away": [0.2] * (len(group_rows) + len(ko_pairs)),
        }
    )


def _ko_results(extra: list[tuple[date, str, str, int, int]] | None = None) -> pl.DataFrame:
    rows = [
        (date(2026, 7, 9), "Alpha", "Boreas", 2, 0),
        (date(2026, 7, 9), "Bravo", "Ares", 1, 1),   # pens — Ares advanced
        (date(2026, 7, 10), "Ceres", "Dione", 1, 0),
        (date(2026, 7, 10), "Delta", "Cato", 0, 3),
        (date(2026, 7, 14), "Alpha", "Ares", 1, 0),
        (date(2026, 7, 15), "Ceres", "Cato", 2, 2),  # pens — Cato advanced
    ] + (extra or [])
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "home_goals": [r[3] for r in rows],
            "away_goals": [r[4] for r in rows],
        }
    )


def _ko_group_probs() -> pl.DataFrame:
    teams = list(_GROUP_WINNERS.values()) + list(_GROUP_RUNNERS.values())
    return pl.DataFrame(
        {
            "team": teams,
            "finish_1": [1.0] * 4 + [0.0] * 4,
            "finish_2": [0.0] * 4 + [1.0] * 4,
            "finish_3": [0.0] * 8,
            "finish_4": [0.0] * 8,
        }
    )


def _ko_stage_probs() -> pl.DataFrame:
    teams = ["Alpha", "Ares", "Bravo", "Boreas", "Ceres", "Cato", "Delta", "Dione"]
    in_sf = {"Alpha", "Ares", "Ceres", "Cato"}
    in_final = {"Alpha", "Cato"}
    return pl.DataFrame(
        {
            "team": teams,
            "R32": [1.0] * 8,
            "R16": [1.0] * 8,
            "QF": [1.0] * 8,
            "SF": [1.0 if t in in_sf else 0.0 for t in teams],
            "FINAL": [1.0 if t in in_final else 0.0 for t in teams],
            "champion": [0.6 if t == "Alpha" else 0.4 if t == "Cato" else 0.0 for t in teams],
        }
    )


def test_classify_ko_fixtures_assigns_rounds_from_schedule() -> None:
    ko = data.classify_ko_fixtures(_ko_match_df(), _ko_results(), schedule_df=_ko_schedule())
    stages = {
        (r["home_team"], r["away_team"]): r["stage"] for r in ko.iter_rows(named=True)
    }
    assert stages[("Alpha", "Boreas")] == "QF"
    assert stages[("Alpha", "Ares")] == "SF"
    # Level ties keep the drawn scoreline and a "draw" outcome.
    drawn = ko.filter(pl.col("home_team") == "Bravo").row(0, named=True)
    assert drawn["actual_outcome"] == "draw"


def test_classify_ko_fixtures_separates_third_place_from_final() -> None:
    extra_pairs = [("Ares", "Ceres"), ("Alpha", "Cato")]
    extra_results = [
        (date(2026, 7, 18), "Ares", "Ceres", 1, 0),
        (date(2026, 7, 19), "Alpha", "Cato", 2, 1),
    ]
    ko = data.classify_ko_fixtures(
        _ko_match_df(extra_pairs), _ko_results(extra_results), schedule_df=_ko_schedule()
    )
    stages = {
        (r["home_team"], r["away_team"]): r["stage"] for r in ko.iter_rows(named=True)
    }
    # The third-place match (played the day before the final) is not a semi-final.
    assert stages[("Ares", "Ceres")] == "TP"
    assert stages[("Alpha", "Cato")] == "F"


def test_dark_horses_excludes_eliminated_teams() -> None:
    df = pl.DataFrame(
        {
            # Out: eliminated in the QF (conditioned champion prob is 0).
            # Alive: a live underdog. Fav: the live favourite.
            "team": ["Out", "Alive", "Fav"],
            "R16": [1.0, 1.0, 1.0],
            "QF": [1.0, 1.0, 1.0],
            "SF": [0.0, 0.6, 0.9],
            "FINAL": [0.0, 0.3, 0.7],
            "champion": [0.0, 0.02, 0.5],
        }
    )
    horses = data.dark_horses(df)
    assert horses["team"].to_list() == ["Alive"]
    assert horses["overperformance"].to_list() == [pytest.approx(50.0)]


def test_resolve_bracket_cascades_winners_to_final() -> None:
    match_df = _ko_match_df()
    ko = data.classify_ko_fixtures(match_df, _ko_results(), schedule_df=_ko_schedule())
    bracket = data.resolve_bracket(
        _ko_schedule(), ko, _ko_group_probs(), match_df, stage_probs=_ko_stage_probs()
    )
    by_number = {r["match_number"]: r for r in bracket.iter_rows(named=True)}

    # Quarter-finals resolve from group positions; the drawn tie (74) resolves
    # its winner from Ares' semi-final appearance.
    assert (by_number[73]["home_resolved"], by_number[73]["away_resolved"]) == (
        "Alpha", "Boreas",
    )
    assert by_number[74]["status"] == "played"

    # Semi-finals carry the QF winners, including the shootout winner Ares.
    assert (by_number[77]["home_resolved"], by_number[77]["away_resolved"]) == (
        "Alpha", "Ares",
    )
    assert (by_number[78]["home_resolved"], by_number[78]["away_resolved"]) == (
        "Ceres", "Cato",
    )
    assert by_number[77]["status"] == "played"
    assert by_number[78]["status"] == "played"

    # The drawn semi (78) resolves via stage probabilities (Ceres' FINAL prob is 0),
    # so the final and third-place slots are fully resolved but not yet played.
    assert (by_number[79]["home_resolved"], by_number[79]["away_resolved"]) == (
        "Ares", "Ceres",
    )
    assert (by_number[80]["home_resolved"], by_number[80]["away_resolved"]) == (
        "Alpha", "Cato",
    )
    assert by_number[79]["status"] == "upcoming"
    assert by_number[80]["status"] == "upcoming"
