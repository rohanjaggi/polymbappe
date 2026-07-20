import math
from datetime import date

import polars as pl

from polymbappe.eval.backtest import Tournament
from polymbappe.features.context import (
    build_form_features,
    build_h2h_features,
    build_rest_features,
    build_structural_features,
)
from polymbappe.features.elo import EloConfig, build_elo_features
from polymbappe.features.pipeline import (
    FeaturePipeline,
    build_contextual_matrix,
    result_label,
)
from polymbappe.features.squad import build_squad_features, build_squad_match_features
from polymbappe.features.xg import build_xg_features

# Chronological synthetic history. A is strong, plays B twice and C once.
MATCHES = pl.DataFrame(
    {
        "match_id": ["m1", "m2", "m3", "m4"],
        "date": [date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 20), date(2020, 2, 1)],
        "home_team": ["A", "B", "A", "A"],
        "away_team": ["B", "A", "C", "B"],
        "home_goals": [2, 1, 0, 3],
        "away_goals": [0, 1, 1, 1],
        "competition": ["Friendly"] * 4,
        "is_knockout": [False, False, False, True],
        "neutral_site": [False, False, False, True],
        "group": [None, None, None, None],
    }
)


def _row(df: pl.DataFrame, match_id: str, team: str) -> dict[str, object]:
    return df.filter((pl.col("match_id") == match_id) & (pl.col("team") == team)).row(
        0, named=True
    )


def test_result_label() -> None:
    assert result_label(2, 0) == "H"
    assert result_label(1, 1) == "D"
    assert result_label(0, 3) == "A"


def test_elo_is_point_in_time() -> None:
    elo = build_elo_features(MATCHES, config=EloConfig())
    # First match: both teams start at the base rating (own result not leaked).
    assert _row(elo, "m1", "A")["elo_pre"] == 1500.0
    assert _row(elo, "m1", "B")["elo_pre"] == 1500.0
    # After A wins m1, A's pre-rating at m2 exceeds B's.
    assert _row(elo, "m2", "A")["elo_pre"] > 1500.0
    assert _row(elo, "m2", "B")["elo_pre"] < 1500.0


def test_elo_update_respects_neutral_site() -> None:
    from polymbappe.features.elo import EloRatings

    # Neutral venue: equal-rated teams have expected score 0.5, so a win moves
    # exactly k * 0.5. With home advantage the home side is expected to win more
    # often, so the same result earns less.
    neutral = EloRatings(EloConfig())
    neutral.update("A", "B", 1, 0, neutral=True)
    assert math.isclose(neutral.rating("A") - 1500.0, EloConfig().k_factor * 0.5)

    home_adv = EloRatings(EloConfig())
    home_adv.update("A", "B", 1, 0)
    assert home_adv.rating("A") < neutral.rating("A")


def test_build_elo_features_threads_neutral_site() -> None:
    # Same fixtures/results, differing only in the neutral_site flag of m1: the
    # walker must credit the m1 winner differently, visible in m2's pre-ratings.
    flagged = MATCHES.with_columns(
        pl.when(pl.col("match_id") == "m1")
        .then(True)
        .otherwise(pl.col("neutral_site"))
        .alias("neutral_site")
    )
    elo_home = build_elo_features(MATCHES, config=EloConfig())
    elo_neutral = build_elo_features(flagged, config=EloConfig())
    assert _row(elo_neutral, "m2", "A")["elo_pre"] > _row(elo_home, "m2", "A")["elo_pre"]


def test_form_excludes_current_match() -> None:
    form = build_form_features(MATCHES, windows=(5,))
    # A's first appearance has no prior form.
    assert _row(form, "m1", "A")["gs_5"] is None
    # At m3, A has played m1 (gf2,ga0,pts3) and m2 (gf1,ga1,pts1).
    a_m3 = _row(form, "m3", "A")
    assert a_m3["gs_5"] == 1.5
    assert a_m3["ga_5"] == 0.5
    assert a_m3["pts_5"] == 2.0


def test_rest_days() -> None:
    rest = build_rest_features(MATCHES)
    assert _row(rest, "m1", "A")["rest_days"] is None
    assert _row(rest, "m2", "A")["rest_days"] == 9  # 2020-01-10 minus 2020-01-01


def test_h2h_uses_only_prior_meetings() -> None:
    h2h = build_h2h_features(MATCHES, window=5)
    m1 = h2h.filter(pl.col("match_id") == "m1").row(0, named=True)
    assert m1["h2h_meetings"] == 0
    assert m1["h2h_home_winrate"] is None
    # m4 (A vs B): prior meetings m1 (A won) and m2 (draw) -> (1 + 0.5)/2 = 0.75.
    m4 = h2h.filter(pl.col("match_id") == "m4").row(0, named=True)
    assert m4["h2h_meetings"] == 2
    assert m4["h2h_home_winrate"] == 0.75


def test_structural_host_flags() -> None:
    struct = build_structural_features(MATCHES, hosts=frozenset({"A"}))
    m4 = struct.filter(pl.col("match_id") == "m4").row(0, named=True)
    assert m4["home_is_host"] is True
    assert m4["away_is_host"] is False
    assert m4["is_knockout"] is True


def test_xg_proxy_falls_back_to_goals() -> None:
    xg = build_xg_features(MATCHES, team_xg=None, window=5)
    assert _row(xg, "m1", "A")["xg_is_proxy"] is True
    # At m3, A's rolling proxy xg_for = mean(2, 1) = 1.5 (same as goals form).
    assert _row(xg, "m3", "A")["xg_for"] == 1.5


# A World Cup fixture (A strong, B weak) plus the squad-valuation snapshot for that
# tournament. Dates fall inside the WC2022 window so ``select_fixtures`` maps the match.
_WC_MATCHES = pl.DataFrame(
    {
        "match_id": ["w1"],
        "date": [date(2022, 11, 21)],
        "home_team": ["A"],
        "away_team": ["B"],
        "home_goals": [2],
        "away_goals": [0],
        "competition": ["FIFA World Cup"],
        "is_knockout": [False],
        "neutral_site": [True],
        "group": ["A"],
    }
)
_WC_VALUATIONS = pl.DataFrame(
    {
        "team": ["A", "B"],
        "tournament": ["WC2022", "WC2022"],
        "total_value": [1000.0, 100.0],
        "median_value": [50.0, 5.0],
        "player_count": [26, 26],
    }
)
_WC2022 = Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18))


def test_squad_match_features_map_tournament_snapshot() -> None:
    feats = build_squad_match_features(_WC_MATCHES, _WC_VALUATIONS, (_WC2022,))
    # One row per fixture-team, keyed for the team-table join.
    assert set(feats.columns) == {"match_id", "team", "date", "log_total_value"}
    assert feats.height == 2
    assert _row(feats, "w1", "A")["log_total_value"] == math.log1p(1000.0)
    # A team with no snapshot value for the tournament yields no row.
    empty = build_squad_match_features(_WC_MATCHES, _WC_VALUATIONS, ())
    assert empty.is_empty()


def test_core_matrix_attaches_squad_value_ratio() -> None:
    matrix = FeaturePipeline().build_core_matrix(
        _WC_MATCHES, squad_valuations=_WC_VALUATIONS, tournaments=(_WC2022,)
    )
    row = matrix.row(0, named=True)
    assert "squad_value_ratio" in matrix.columns
    assert row["home_log_total_value"] == math.log1p(1000.0)
    assert row["away_log_total_value"] == math.log1p(100.0)
    # Tier-1 spec feature: log(value_home / value_away), positive when home is stronger.
    assert row["squad_value_ratio"] == math.log1p(1000.0) - math.log1p(100.0)


def test_core_matrix_skips_squad_when_absent() -> None:
    # Default (no valuations) leaves the squad columns off entirely — graceful degradation.
    matrix = FeaturePipeline().build_core_matrix(MATCHES)
    assert "squad_value_ratio" not in matrix.columns
    assert "home_log_total_value" not in matrix.columns
    # build_squad_features still produces the per-team log values it always did.
    per_team = build_squad_features(_WC_VALUATIONS)
    assert "log_total_value" in per_team.columns


def test_pipeline_assembles_matrix_with_label_and_diff() -> None:
    matrix = FeaturePipeline().build_core_matrix(MATCHES)
    assert matrix.height == MATCHES.height
    # Labels.
    labels = dict(zip(matrix["match_id"], matrix["label"], strict=True))
    assert labels == {"m1": "H", "m2": "D", "m3": "A", "m4": "H"}
    # m1 elo_diff is zero (both base) -> no leakage of m1's own result.
    m1 = matrix.filter(pl.col("match_id") == "m1").row(0, named=True)
    assert m1["elo_diff"] == 0.0
    # Home/away feature columns are present.
    assert "home_elo_pre" in matrix.columns
    assert "away_gs_5" in matrix.columns
    assert "h2h_home_winrate" in matrix.columns


def test_contextual_matrix_ppda_diff_null_without_table() -> None:
    # Without a team_ppda table, ppda is null on both sides -> ppda_diff is all-null.
    matrix = build_contextual_matrix(MATCHES)
    assert "ppda_diff" in matrix.columns
    assert matrix["ppda_diff"].null_count() == matrix.height


def test_contextual_matrix_ppda_diff_populated_with_table() -> None:
    # Supplying team_ppda lights up the otherwise-dead ppda_diff feature.
    long_dates = [date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 20), date(2020, 2, 1)]
    team_ppda = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B", "A", "C", "A", "B"],
            "date": [long_dates[0]] * 2
            + [long_dates[1]] * 2
            + [long_dates[2]] * 2
            + [long_dates[3]] * 2,
            "ppda": [9.0, 12.0, 8.0, 11.0, 10.0, 13.0, 9.5, 12.5],
        }
    )
    matrix = build_contextual_matrix(MATCHES, team_ppda=team_ppda)
    # The rolling PPDA excludes a team's own current match, so m1 (first appearance for both)
    # is still null, but later matches carry a real difference.
    assert matrix["ppda_diff"].null_count() < matrix.height
