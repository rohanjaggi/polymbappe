from datetime import date

import polars as pl

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
