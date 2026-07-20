"""Tests for the contextual feature builders and the residual adjuster."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polymbappe.config import Settings
from polymbappe.context.cohesion import build_cohesion_features, club_cluster_index
from polymbappe.context.draw_pressure import (
    draw_pressure_features,
    low_scoring_probability,
    mutual_qualification_incentive,
    stage_elo_interaction,
)
from polymbappe.context.fatigue import (
    add_fatigue_flag,
    build_city_coord_lookup,
    build_match_travel_features,
    build_season_load_features,
    build_travel_features,
    build_travel_features_from_tables,
    coord_lookup_from_venues,
    haversine_km,
    schedule_to_appearances,
    venue_distance,
)
from polymbappe.context.manager import ManagerConfig, build_manager_features, shrink
from polymbappe.context.ppda import build_ppda_features, ppda_difference, ppda_similarity
from polymbappe.context.runtime import (
    SIM_CONTEXT_FEATURES,
    build_tournament_context_features,
    cohesion_lookup,
    gated_feature_groups,
    manager_lookup,
)
from polymbappe.context.sentiment import build_xg_overperformance, score_text_vader
from polymbappe.data.store import write_table
from polymbappe.data.tables import Table
from polymbappe.eval.backtest import Tournament


def _matches() -> pl.DataFrame:
    rows = []
    day = date(2020, 1, 1)
    for i in range(12):
        rows.append(
            {
                "match_id": f"m{i}", "date": day + timedelta(days=i * 7),
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_goals": 2, "away_goals": 1,
                "competition": "Friendly", "is_knockout": False,
                "neutral_site": False, "group": None,
            }
        )
    return pl.DataFrame(rows)


# -- PPDA ----------------------------------------------------------------------

def test_ppda_difference_and_similarity() -> None:
    assert ppda_difference(8.0, 12.0) == -4.0
    assert ppda_difference(None, 12.0) is None
    assert ppda_similarity(10.0, 10.0) == 1.0
    assert ppda_similarity(0.0, 20.0) == 0.0
    assert ppda_similarity(None, 5.0) is None


def test_build_ppda_features_proxy_and_real() -> None:
    matches = _matches()
    proxy = build_ppda_features(matches)
    assert proxy["ppda_available"].to_list() == [False] * proxy.height
    team_ppda = pl.DataFrame(
        {"team": ["A"] * 6, "date": matches["date"][:6], "ppda": [10.0, 9, 11, 8, 12, 10]}
    )
    real = build_ppda_features(matches, team_ppda)
    assert real.filter(pl.col("ppda_available"))["ppda"].null_count() == 0


# -- cohesion ------------------------------------------------------------------

def test_club_cluster_index() -> None:
    assert club_cluster_index({"X": 3, "Y": 2, "Z": 1}) == 3 + 1 + 0


def test_build_cohesion_features() -> None:
    squads = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B"],
            "tournament": ["2026"] * 5,
            "player": ["p1", "p2", "p3", "p4", "p5"],
            "club": ["City", "City", "Madrid", "PSG", None],
            "age": [27, 29, 31, 25, None],
        }
    )
    out = build_cohesion_features(squads).sort("team")
    a = out.filter(pl.col("team") == "A").row(0, named=True)
    assert a["club_cluster_index"] == 1  # City pair (2 players) -> 1, Madrid -> 0
    assert a["median_age"] == 29.0
    assert a["player_count"] == 3


# -- manager -------------------------------------------------------------------

def test_shrink_pulls_toward_prior() -> None:
    # 1 win in 1 match, prior 0.5 with prior_n=4 -> well below 1.0.
    assert shrink(1.0, 1.0, 0.5, 4.0) == pytest.approx((1 + 2.0) / 5.0)


def test_build_manager_features_shrinkage_and_recency() -> None:
    records = pl.DataFrame(
        {
            "manager": ["X", "X", "Y"],
            "team": ["A", "A", "B"],
            "tournament": ["2018", "2022", "2022"],
            "tournament_order": [1, 2, 2],
            "stage_reached": ["QF", "FINAL", "R16"],
            "knockout_matches": [3, 6, 2],
            "knockout_wins": [2, 5, 0],
        }
    )
    out = build_manager_features(records, ManagerConfig()).sort("manager")
    x = out.filter(pl.col("manager") == "X").row(0, named=True)
    y = out.filter(pl.col("manager") == "Y").row(0, named=True)
    # X has a strong, deep record; Y none -> X ranks higher on both signals.
    assert x["knockout_win_rate"] > y["knockout_win_rate"]
    assert x["deepest_run_weighted"] > y["deepest_run_weighted"]
    assert x["tenure_matches"] == 9


# -- runtime assembly (cohesion + manager) -------------------------------------

def _assembly_matches() -> pl.DataFrame:
    """Two World Cups (T1 then T2) of A-vs-B fixtures, with friendly history before each."""

    rows = []
    # Pre-history friendlies so each tournament has Elo/overperf history.
    for i in range(6):
        rows.append(
            {
                "match_id": f"h{i}", "date": date(2017, 1, 1) + timedelta(days=i * 30),
                "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0,
                "competition": "Friendly", "is_knockout": False,
                "neutral_site": False, "group": None,
            }
        )
    # T1 fixture (FIFA World Cup, 2018).
    rows.append(
        {
            "match_id": "t1", "date": date(2018, 6, 20), "home_team": "A", "away_team": "B",
            "home_goals": 2, "away_goals": 1, "competition": "FIFA World Cup",
            "is_knockout": False, "neutral_site": True, "group": "A",
        }
    )
    # T2 fixture (FIFA World Cup, 2022).
    rows.append(
        {
            "match_id": "t2", "date": date(2022, 11, 25), "home_team": "A", "away_team": "B",
            "home_goals": 0, "away_goals": 0, "competition": "FIFA World Cup",
            "is_knockout": False, "neutral_site": True, "group": "A",
        }
    )
    return pl.DataFrame(rows)


def _write_context_tables(settings: Settings) -> None:
    """Squads + manager records for tournament WC2022; team A populated, B absent."""

    squads = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "tournament": ["WC2022"] * 3,
            "player": ["p1", "p2", "p3"],
            "club": ["City", "City", "Madrid"],
            "age": [27.0, 29.0, 31.0],
        }
    )
    write_table(Table.SQUADS, squads, settings=settings)
    records = pl.DataFrame(
        {
            "manager": ["mgrA", "mgrA"],
            "team": ["A", "A"],
            "tournament": ["WC2018", "WC2022"],
            "stage_reached": ["FINAL", "QF"],
            "knockout_matches": [6, 3],
            "knockout_wins": [5, 2],
            "tournament_order": [1, 2],
        }
    )
    write_table(Table.MANAGER_RECORDS, records, settings=settings)


def test_build_tournament_context_features_uses_team_xg(tmp_path) -> None:
    """With a team_xg table, the fit path's xg-overperformance is real (non-zero).

    Guards the train/serve skew where the fit path used the goals-vs-goals proxy
    (identically zero) while the live simulation path passed real xG.
    """

    settings = Settings(data_dir=tmp_path)
    history = _assembly_matches()
    xg_rows = [
        {"team": r[team_col], "date": r["date"], "xg": 0.3, "xga": 0.9}
        for r in history.iter_rows(named=True)
        for team_col in ("home_team", "away_team")
    ]
    write_table(Table.TEAM_XG, pl.DataFrame(xg_rows), settings=settings)

    tournaments = (Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18)),)
    out = build_tournament_context_features(history, tournaments, settings)
    row = out.filter(pl.col("match_id") == "t2").row(0, named=True)
    # A scores ~1/match against xg 0.3 -> clearly positive overperformance.
    assert row["home_xg_overperf"] > 0.0


def test_load_live_wc2026_matches_captures_group_and_knockout() -> None:
    from polymbappe.context.adaptive import load_live_wc2026_matches

    matches = pl.DataFrame(
        {
            "match_id": ["pre", "grp", "ko", "other"],
            "date": [date(2026, 3, 1), date(2026, 6, 12), date(2026, 7, 15), date(2026, 6, 20)],
            "home_team": ["A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B"],
            "home_goals": [1, 2, 1, 0],
            "away_goals": [0, 0, 1, 0],
            "competition": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "Friendly"],
            "is_knockout": [False, False, True, False],
            "neutral_site": [True] * 4,
            "group": [None] * 4,
        },
        schema_overrides={"group": pl.Utf8},
    )
    live = load_live_wc2026_matches(matches)
    # Group AND knockout rows inside the tournament window; nothing else.
    assert sorted(live["match_id"].to_list()) == ["grp", "ko"]


def test_build_tournament_context_features_no_tables(tmp_path) -> None:
    """Without squads/manager tables: original 3 cols + zero-filled new cols, full schema."""

    settings = Settings(data_dir=tmp_path)
    tournaments = (Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18)),)
    out = build_tournament_context_features(_assembly_matches(), tournaments, settings)
    assert out.columns == ["match_id", *SIM_CONTEXT_FEATURES]
    row = out.filter(pl.col("match_id") == "t2").row(0, named=True)
    for col in SIM_CONTEXT_FEATURES:
        if col not in ("home_xg_overperf", "away_xg_overperf", "draw_pressure"):
            assert row[col] == 0.0


def test_build_tournament_context_features_with_tables(tmp_path) -> None:
    """Cohesion/manager non-zero for the present team (A) and 0.0 for the absent team (B)."""

    settings = Settings(data_dir=tmp_path)
    _write_context_tables(settings)
    tournaments = (Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18)),)
    out = build_tournament_context_features(_assembly_matches(), tournaments, settings)
    assert out.columns == ["match_id", *SIM_CONTEXT_FEATURES]
    row = out.filter(pl.col("match_id") == "t2").row(0, named=True)
    # Home = A (present): cohesion + manager pedigree (from WC2018, the only pre-cutoff rec).
    assert row["home_club_cluster_index"] == 1.0  # City pair
    assert row["home_median_age"] == 29.0
    assert row["home_knockout_win_rate"] > 0.0
    assert row["home_deepest_run_weighted"] > 0.0
    # Away = B (absent from both tables): all new cols 0.0.
    assert row["away_club_cluster_index"] == 0.0
    assert row["away_median_age"] == 0.0
    assert row["away_knockout_win_rate"] == 0.0
    assert row["away_deepest_run_weighted"] == 0.0


def test_cohesion_lookup_filters_to_tournament() -> None:
    squads = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "tournament": ["WC2022", "WC2022", "WC2018"],
            "player": ["p1", "p2", "p3"],
            "club": ["City", "City", "Madrid"],
            "age": [27.0, 29.0, 40.0],
        }
    )
    t = Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18))
    out = cohesion_lookup(squads, t)
    cluster, age = out["A"]
    assert cluster == 1.0  # only the 2 WC2022 City players form a pair
    assert age == 28.0  # median of (27, 29); the WC2018 row (40) is excluded


def test_manager_lookup_excludes_own_tournament() -> None:
    """Critical leakage guard: pedigree for T uses only records before T's order."""

    records = pl.DataFrame(
        {
            "manager": ["mgr", "mgr"],
            "team": ["A", "A"],
            "tournament": ["WC2018", "WC2022"],
            "stage_reached": ["group", "FINAL"],
            "knockout_matches": [0, 6],
            "knockout_wins": [0, 5],
            "tournament_order": [1, 2],
        }
    )
    t2022 = Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18))
    t2018 = Tournament("WC2018", "FIFA World Cup", date(2018, 6, 14), date(2018, 7, 15))
    ped_2022 = manager_lookup(records, t2022)  # cutoff=2 -> only WC2018 (group, no KO)
    ped_2018 = manager_lookup(records, t2018)  # cutoff=1 -> no pre-cutoff records at all
    # WC2022 pedigree must NOT see its own strong FINAL record.
    assert "A" in ped_2022
    assert ped_2022["A"]["deepest_run_weighted"] == 0.0  # WC2018 was a group exit
    # WC2018 is the earliest record -> no history -> team omitted (0-filled downstream).
    assert "A" not in ped_2018


def test_manager_lookup_live_uses_all_records() -> None:
    """Live case: tournament absent from records -> cutoff=+inf -> all records used."""

    records = pl.DataFrame(
        {
            "manager": ["mgr", "mgr"],
            "team": ["A", "A"],
            "tournament": ["WC2018", "WC2022"],
            "stage_reached": ["QF", "FINAL"],
            "knockout_matches": [3, 6],
            "knockout_wins": [2, 5],
            "tournament_order": [1, 2],
        }
    )
    live = Tournament("WC2026", "FIFA World Cup", date(2026, 6, 11), date(2026, 7, 19))
    ped = manager_lookup(records, live)
    assert ped["A"]["deepest_run_weighted"] > 0.0  # both records contribute


def test_gated_feature_groups_drops_thin_data() -> None:
    """Coverage gate: a group with non-zero data for < K tournaments is dropped."""

    # One tournament has cohesion data (cluster=5), the rest are zero -> below K=3.
    frame = pl.DataFrame(
        {
            "match_id": ["m0", "m1", "m2"],
            "tournament": ["T0", "T1", "T2"],
            **{c: [0.0, 0.0, 0.0] for c in SIM_CONTEXT_FEATURES},
        }
    )
    frame = frame.with_columns(
        pl.Series("home_club_cluster_index", [5.0, 0.0, 0.0])
    )
    gated = gated_feature_groups(frame)
    assert "cohesion" not in gated  # only 1 tournament has data, < K
    assert "manager" not in gated
    assert "xg_overperformance" in gated  # data-light groups always kept


def test_gated_feature_groups_keeps_covered_group() -> None:
    cols = ["home_club_cluster_index", "away_club_cluster_index"]
    frame = pl.DataFrame(
        {
            "match_id": ["m0", "m1", "m2"],
            "tournament": ["T0", "T1", "T2"],
            **{c: [0.0, 0.0, 0.0] for c in SIM_CONTEXT_FEATURES},
        }
    )
    frame = frame.with_columns(pl.Series(cols[0], [1.0, 2.0, 3.0]))
    gated = gated_feature_groups(frame)
    assert "cohesion" in gated  # 3 tournaments with data >= K


# -- fatigue -------------------------------------------------------------------

def test_haversine_and_venue_distance() -> None:
    # NY to LA is ~3900 km.
    d = venue_distance("New York", "Los Angeles")
    assert d is not None and 3500 < d < 4300
    assert haversine_km((0, 0), (0, 0)) == 0.0
    assert venue_distance("New York", "Atlantis") is None


def test_build_travel_and_load_and_flag() -> None:
    schedule = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "date": [date(2026, 6, 11), date(2026, 6, 16), date(2026, 6, 21)],
            "match_id": ["g1", "g2", "g3"],
            "venue": ["New York", "Los Angeles", "New York"],
        }
    )
    travel = build_travel_features(schedule).sort("match_id")
    assert travel.filter(pl.col("match_id") == "g1")["travel_km"].item() == 0.0
    assert travel.filter(pl.col("match_id") == "g2")["travel_km"].item() > 3000

    minutes = pl.DataFrame(
        {"team": ["A", "B", "C"], "tournament": ["2026"] * 3,
         "season_minutes": [3000.0, 2000.0, 1000.0]}
    )
    load = build_season_load_features(minutes)
    assert load.filter(pl.col("team") == "A")["season_load"].item() > 0

    rest = pl.DataFrame({"match_id": ["x"], "team": ["A"], "rest_days": [3]})
    flagged = add_fatigue_flag(rest)
    assert flagged["fatigued"].item() is True


# -- draw pressure -------------------------------------------------------------

def test_draw_pressure_components() -> None:
    assert mutual_qualification_incentive(True, True, True) == 1
    assert mutual_qualification_incentive(True, True, False) == 0
    assert mutual_qualification_incentive(False, True, True) == 0

    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0.3
    matrix[1, 0] = 0.2
    matrix[0, 1] = 0.2
    matrix[2, 2] = 0.3
    assert low_scoring_probability(matrix) == pytest.approx(0.7)

    # Group stage, small gap -> positive; knockout -> negative.
    assert stage_elo_interaction(False, 20.0) > 0
    assert stage_elo_interaction(True, 20.0) < 0
    assert stage_elo_interaction(False, 500.0) == 0.0


def test_draw_pressure_features_dict() -> None:
    matrix = np.full((4, 4), 1 / 16)
    feats = draw_pressure_features(
        is_final_matchday=True, draw_qualifies_home=True, draw_qualifies_away=True,
        home_ppda=10.0, away_ppda=11.0, score_matrix=matrix,
        is_knockout=False, elo_gap=50.0,
    )
    assert feats["mutual_qual_incentive"] == 1.0
    assert 0.0 <= feats["ppda_similarity"] <= 1.0
    assert set(feats) == {
        "mutual_qual_incentive", "ppda_similarity", "low_scoring_prob",
        "stage_elo_interaction",
    }


# -- sentiment -----------------------------------------------------------------

def test_xg_overperformance_zero_under_proxy() -> None:
    # With no real xG, the proxy makes overperformance ~0 (goals - goals proxy).
    out = build_xg_overperformance(_matches())
    nonnull = out["xg_overperformance"].drop_nulls().to_numpy()
    assert np.allclose(nonnull, 0.0, atol=1e-9)


def test_score_text_vader_graceful() -> None:
    assert score_text_vader([]) == 0.0
    # Returns a float in [-1, 1] whether or not vader is installed.
    val = score_text_vader(["great win", "terrible loss"])
    assert -1.0 <= val <= 1.0


# -- travel features from ingested venues + schedule ---------------------------

def _venues_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "venue": ["Estadio Azteca", "SoFi Stadium", "MetLife Stadium"],
            "city": [
                "Mexico City",
                "Los Angeles (Inglewood)",
                "New York/New Jersey (East Rutherford)",
            ],
            "country": ["mx", "us", "us"],
            "latitude": [19.3029, 33.953, 40.8135],
            "longitude": [-99.1505, -118.339, -74.0744],
        }
    )


def test_coord_lookup_from_venues_keys_full_and_bare() -> None:
    coords = coord_lookup_from_venues(_venues_frame())
    # Both the full host-city string (as the schedule uses) and the bare city resolve.
    assert "Los Angeles (Inglewood)" in coords
    assert "Los Angeles" in coords
    assert coords["Los Angeles (Inglewood)"] == coords["Los Angeles"]
    # venue_distance accepts the ingested lookup and matches its bare-name fallback.
    d = venue_distance("Mexico City", "Los Angeles (Inglewood)", coords)
    assert d is not None and d == pytest.approx(
        venue_distance("Mexico City", "Los Angeles", coords)
    )


def test_schedule_to_appearances_explodes_home_and_away() -> None:
    schedule = pl.DataFrame(
        {
            "match_id": ["m1"],
            "date": [date(2026, 6, 11)],
            "stage": ["Matchday 1"],
            "group": ["A"],
            "home_team": ["Mexico"],
            "away_team": ["South Africa"],
            "city": ["Mexico City"],
        }
    )
    appearances = schedule_to_appearances(schedule)
    assert appearances.columns == ["team", "date", "match_id", "venue"]
    assert appearances.height == 2  # one row per side
    assert set(appearances["team"].to_list()) == {"Mexico", "South Africa"}
    assert appearances["venue"].to_list() == ["Mexico City", "Mexico City"]


def test_build_travel_features_from_tables_uses_ingested_coords() -> None:
    venues = _venues_frame()
    schedule = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": [date(2026, 6, 11), date(2026, 6, 17)],
            "stage": ["Matchday 1", "Matchday 2"],
            "group": ["A", "A"],
            "home_team": ["Mexico", "Mexico"],
            "away_team": ["South Africa", "Canada"],
            "city": ["Mexico City", "Los Angeles (Inglewood)"],
        }
    )
    travel = build_travel_features_from_tables(schedule, venues)
    # Mexico's first appearance is 0; its Mexico City -> LA hop is ~2500 km.
    mex = travel.filter(pl.col("team") == "Mexico").sort("match_id")
    assert mex.filter(pl.col("match_id") == "m1")["travel_km"].item() == 0.0
    assert mex.filter(pl.col("match_id") == "m2")["travel_km"].item() == pytest.approx(
        2500.0, abs=150.0
    )


# -- historical travel backfill (GeoNames gazetteer) ---------------------------

def _city_coords_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "city": ["moscow", "moscow", "rio de janeiro", "sao paulo"],
            "country": ["RU", "US", "BR", "BR"],
            "latitude": [55.7522, 46.7324, -22.9111, -23.5475],
            "longitude": [37.6156, -117.0002, -43.2056, -46.6361],
            "population": [10381222, 25000, 6320446, 10021295],
        }
    )


def test_build_city_coord_lookup_picks_highest_population() -> None:
    coords = build_city_coord_lookup(_city_coords_frame())
    # Moscow resolves to the RU entry (10M) over Idaho's (25k).
    assert coords["moscow"] == pytest.approx((55.7522, 37.6156))
    assert "sao paulo" in coords


def test_build_match_travel_features_accent_folds_city() -> None:
    coords = build_city_coord_lookup(_city_coords_frame())
    matches = pl.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": [date(2014, 6, 12), date(2014, 6, 17)],
            "home_team": ["Brazil", "Brazil"],
            "away_team": ["Croatia", "Mexico"],
            # Accented spellings must still resolve against the ASCII gazetteer.
            "city": ["São Paulo", "Rio de Janeiro"],
        }
    )
    travel = build_match_travel_features(matches, coords).filter(pl.col("team") == "Brazil")
    by_match = {r["match_id"]: r["travel_km"] for r in travel.iter_rows(named=True)}
    assert by_match["m1"] == 0.0  # first appearance
    assert by_match["m2"] == pytest.approx(360.0, abs=60.0)  # São Paulo -> Rio ~360 km


def test_context_features_carry_travel_from_gazetteer(tmp_path) -> None:
    """With a city_coords table + city-bearing matches, the fatigue columns are nonzero."""

    settings = Settings(data_dir=tmp_path)
    write_table(Table.CITY_COORDS, _city_coords_frame(), settings=settings)
    matches = pl.DataFrame(
        {
            "match_id": ["h0", "wc1", "wc2"],
            "date": [date(2013, 6, 1), date(2014, 6, 12), date(2014, 6, 17)],
            "home_team": ["Brazil", "Brazil", "Brazil"],
            "away_team": ["Chile", "Croatia", "Mexico"],
            "home_goals": [2, 3, 1],
            "away_goals": [0, 1, 0],
            "competition": ["Friendly", "FIFA World Cup", "FIFA World Cup"],
            "is_knockout": [False, False, False],
            "neutral_site": [False, False, False],
            "group": [None, "A", "A"],
            "city": ["Sao Paulo", "São Paulo", "Rio de Janeiro"],
        }
    )
    tournaments = (Tournament("WC2014", "FIFA World Cup", date(2014, 6, 12), date(2014, 7, 13)),)
    out = build_tournament_context_features(matches, tournaments, settings)
    assert "home_travel_km" in out.columns and "away_travel_km" in out.columns
    wc2 = out.filter(pl.col("match_id") == "wc2").row(0, named=True)
    # Brazil (home) travelled São Paulo -> Rio between its two group games.
    assert wc2["home_travel_km"] == pytest.approx(360.0, abs=60.0)


def test_labels_from_matches_row_aligned() -> None:
    import polars as pl

    from polymbappe.context.adaptive import labels_from_matches

    live = pl.DataFrame(
        {
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
        }
    )
    assert labels_from_matches(live) == ["H", "D", "A"]
