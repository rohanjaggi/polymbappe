"""Tests for the data-ingestion layer (offline, via local raw files)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from polymbappe.config import Settings
from polymbappe.data.ingest import (
    derive_manager_records,
    ingest_all_sources,
    ingest_elo,
    ingest_manager_records,
    ingest_market_odds,
    ingest_squads,
    ingest_team_xg,
)
from polymbappe.data.store import read_table, table_exists
from polymbappe.data.tables import TABLE_COLUMNS, Table

_RESULTS_CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2018-06-14,Russia,Saudi Arabia,5,0,FIFA World Cup,Moscow,Russia,False\n"
    "2018-06-15,Egypt,Uruguay,0,1,FIFA World Cup,Yekaterinburg,Russia,True\n"
    "2018-06-19,Russia,Egypt,3,1,FIFA World Cup,Saint Petersburg,Russia,False\n"
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _write_results(settings: Settings) -> None:
    (settings.raw_data_dir / "results.csv").write_text(_RESULTS_CSV)


def test_ingest_elo_from_matches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    n = ingest_elo(settings)
    assert n == 6  # 3 matches x 2 teams
    snaps = read_table(Table.ELO_SNAPSHOTS, settings)
    assert set(snaps.columns) == set(TABLE_COLUMNS[Table.ELO_SNAPSHOTS])
    # Russia won twice by big margins -> ends above the default 1500 baseline.
    russia_last = snaps.filter(pl.col("team") == "Russia").sort("date").row(-1, named=True)
    assert russia_last["rating"] > 1500.0


def test_ingest_elo_requires_matches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    try:
        ingest_elo(settings)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError without a matches table")


def test_ingest_market_odds_aligns_match_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    (settings.raw_data_dir / "odds.csv").write_text(
        "date,home_team,away_team,home_odds,draw_odds,away_odds\n"
        "2018-06-14,Russia,Saudi Arabia,1.5,4.0,7.0\n"
    )
    n = ingest_market_odds(settings)
    assert n == 1
    odds = read_table(Table.MARKET_ODDS, settings)
    row = odds.row(0, named=True)
    # match_id must match the matches-table convention so a join lands.
    matches = read_table(Table.MATCHES, settings)
    assert row["match_id"] in set(matches["match_id"].to_list())
    # Overround removed -> probabilities sum to 1; strong favorite has the most mass.
    assert abs(row["home_win_prob"] + row["draw_prob"] + row["away_win_prob"] - 1.0) < 1e-9
    assert row["home_win_prob"] > row["away_win_prob"]


def test_ingest_market_odds_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_market_odds(settings) == 0
    assert not table_exists(Table.MARKET_ODDS, settings)


def test_ingest_team_xg_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "team_xg.csv").write_text(
        "team,date,xg,xga\nRussia,2018-06-14,2.7,0.4\n"
    )
    n = ingest_team_xg(settings)
    assert n == 1
    xg = read_table(Table.TEAM_XG, settings)
    assert set(xg.columns) == set(TABLE_COLUMNS[Table.TEAM_XG])
    assert xg.row(0, named=True)["xg"] == 2.7


def test_ingest_squads_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squads.csv").write_text(
        "team,tournament,player,club,age\n"
        "USA,WC2018,Christian Pulisic, Borussia Dortmund ,19\n"
        "USA,WC2018,Weston McKennie,Schalke 04,20\n"
        "Brazil,WC2018,Neymar,Paris Saint-Germain,26\n"
    )
    n = ingest_squads(settings)
    assert n == 3
    squads = read_table(Table.SQUADS, settings)
    assert tuple(squads.columns) == TABLE_COLUMNS[Table.SQUADS]
    assert squads.schema["age"] == pl.Float64
    # team normalized via alias (USA -> United States) and club trimmed.
    usa = squads.filter(pl.col("player") == "Christian Pulisic").row(0, named=True)
    assert usa["team"] == "United States"
    assert usa["club"] == "Borussia Dortmund"
    assert "USA" not in set(squads["team"].to_list())


def test_ingest_squads_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_squads(settings) == 0
    assert not table_exists(Table.SQUADS, settings)


def test_ingest_manager_records_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "manager_records.csv").write_text(
        "manager,team,tournament,stage_reached,knockout_matches,knockout_wins,tournament_order\n"
        "Gregg Berhalter,USA,WC2022,R16,1,0,3\n"
        "Tite,Brazil,WC2018,QF,2,1,2\n"
    )
    n = ingest_manager_records(settings)
    assert n == 2
    records = read_table(Table.MANAGER_RECORDS, settings)
    assert tuple(records.columns) == TABLE_COLUMNS[Table.MANAGER_RECORDS]
    assert records.schema["knockout_matches"] == pl.Int64
    assert records.schema["tournament_order"] == pl.Int64
    usa = records.filter(pl.col("manager") == "Gregg Berhalter").row(0, named=True)
    assert usa["team"] == "United States"  # alias normalized
    assert "USA" not in set(records["team"].to_list())


def test_ingest_manager_records_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_manager_records(settings) == 0
    assert not table_exists(Table.MANAGER_RECORDS, settings)


def test_derive_manager_records_from_matches() -> None:
    from datetime import date

    from polymbappe.eval.backtest import Tournament

    tours = (Tournament("WC2018", "FIFA World Cup", date(2018, 6, 14), date(2018, 7, 15)),)
    matches = pl.DataFrame(
        {
            "home_team": ["Brazil", "Brazil"],
            "away_team": ["Mexico", "Belgium"],
            "home_goals": [2, 1],
            "away_goals": [0, 2],
            "competition": ["FIFA World Cup", "FIFA World Cup"],
            "is_knockout": [True, True],
            "date": [date(2018, 7, 2), date(2018, 7, 6)],
        }
    )
    out = derive_manager_records(
        [{"manager": "Tite", "team": "Brazil", "start_year": 2016, "end_year": 2022}],
        matches,
        tournaments=tours,
    )
    assert tuple(out.columns) == TABLE_COLUMNS[Table.MANAGER_RECORDS]
    row = out.row(0, named=True)
    assert row["tournament"] == "WC2018"
    assert row["knockout_matches"] == 2  # won R16, lost QF
    assert row["knockout_wins"] == 1
    assert row["stage_reached"] == "QF"  # 2 knockout matches reached -> QF
    assert row["tournament_order"] == 0


def test_normalize_footballdata_odds() -> None:
    from polymbappe.data.normalize import normalize_footballdata_odds

    fd = pl.DataFrame(
        {
            "Date": ["11/08/2023", "12/08/2023"],
            "HomeTeam": ["Arsenal", "Spurs"],
            "AwayTeam": ["Forest", "Brentford"],
            "B365H": [1.30, 2.0], "B365D": [5.5, 3.4], "B365A": [9.0, 3.6],
        }
    )
    out = normalize_footballdata_odds(fd)
    assert out.columns == list(TABLE_COLUMNS[Table.MARKET_ODDS])
    row = out.filter(pl.col("match_id") == "2023-08-11__Arsenal__Forest").row(0, named=True)
    assert row["source"] == "football-data"
    assert abs(row["home_win_prob"] + row["draw_prob"] + row["away_win_prob"] - 1.0) < 1e-9
    assert row["home_win_prob"] > row["away_win_prob"]  # heavy home favorite


def test_normalize_footballdata_prefix_fallback() -> None:
    from polymbappe.data.normalize import normalize_footballdata_odds

    fd = pl.DataFrame(
        {
            "Date": ["01/06/2024"], "HomeTeam": ["A"], "AwayTeam": ["B"],
            "AvgH": [2.0], "AvgD": [3.3], "AvgA": [3.7],  # no B365 -> falls back to Avg
        }
    )
    out = normalize_footballdata_odds(fd)
    assert out.height == 1


def test_polymarket_three_way_and_alignment() -> None:
    from polymbappe.polymarket.adapter import (
        align_polymarket_to_fixtures,
        normalize_polymarket_three_way,
    )

    long = pl.DataFrame(
        {
            "market_id": ["m1", "m1", "m1", "m2", "m2"],  # m2 is not a clean 3-way
            "question": ["Spain vs Brazil"] * 3 + ["Coin"] * 2,
            "outcome": ["Spain", "Draw", "Brazil", "Heads", "Tails"],
            "price": [0.55, 0.25, 0.30, 0.5, 0.5],
        }
    )
    tw = normalize_polymarket_three_way(long)
    assert tw.height == 1  # m2 skipped (no draw / 2 outcomes)
    r = tw.row(0, named=True)
    assert abs(r["prob_a"] + r["prob_draw"] + r["prob_b"] - 1.0) < 1e-9  # overround removed

    fixtures = pl.DataFrame(
        {"match_id": ["2026__Brazil__Spain"], "home_team": ["Brazil"], "away_team": ["Spain"]}
    )
    aligned = align_polymarket_to_fixtures(tw, fixtures)
    row = aligned.row(0, named=True)
    assert row["match_id"] == "2026__Brazil__Spain" and row["source"] == "polymarket"
    # Brazil is home and the underdog here -> home prob < away prob.
    assert row["home_win_prob"] < row["away_win_prob"]
    # An unknown pairing is dropped.
    assert align_polymarket_to_fixtures(
        tw, pl.DataFrame({"match_id": ["x"], "home_team": ["X"], "away_team": ["Y"]})
    ).is_empty()


def test_ingest_market_odds_multi_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Local normalized odds.csv
    (settings.raw_data_dir / "odds.csv").write_text(
        "date,home_team,away_team,home_odds,draw_odds,away_odds\n"
        "2018-06-14,Russia,Saudi Arabia,1.5,4.0,7.0\n"
    )
    # Football-Data CSV in the football_data/ dir
    fd_dir = settings.raw_data_dir / "football_data"
    fd_dir.mkdir()
    (fd_dir / "E0.csv").write_text(
        "Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n11/08/2023,Arsenal,Forest,1.3,5.5,9.0\n"
    )
    n = ingest_market_odds(settings)
    assert n == 2  # one local + one football-data row
    odds = read_table(Table.MARKET_ODDS, settings)
    assert set(odds["source"].to_list()) == {"local-csv", "football-data"}


def test_ingest_all_sources_orchestration(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_results(settings)
    (settings.raw_data_dir / "odds.csv").write_text(
        "date,home_team,away_team,home_odds,draw_odds,away_odds\n"
        "2018-06-14,Russia,Saudi Arabia,1.5,4.0,7.0\n"
    )
    report = ingest_all_sources(settings=settings)
    assert report["results"] == 3
    assert report["elo"] == 6
    assert report["market_odds"] == 1
    assert report["team_xg"] == 0  # optional, no file -> skipped cleanly
    assert report["squads"] == 0  # optional, no file/scraper -> skipped cleanly
    assert report["manager_records"] == 0  # optional, no file/scraper -> skipped cleanly
    for table in (Table.MATCHES, Table.ELO_SNAPSHOTS, Table.MARKET_ODDS):
        assert table_exists(table, settings)
