"""Tests for the data-ingestion layer (offline, via local raw files)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from polymbappe.config import Settings
from polymbappe.data.ingest import (
    ingest_all_sources,
    ingest_elo,
    ingest_market_odds,
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
    for table in (Table.MATCHES, Table.ELO_SNAPSHOTS, Table.MARKET_ODDS):
        assert table_exists(table, settings)
