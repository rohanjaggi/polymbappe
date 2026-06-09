from datetime import date
from pathlib import Path

import polars as pl

from polymbappe.config import Settings
from polymbappe.data.ingest import ingest_results
from polymbappe.data.store import connect, read_table, table_exists, write_table
from polymbappe.data.tables import Table


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


def test_write_read_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    df = pl.DataFrame(
        {"team": ["BRA", "ARG"], "date": [date(2026, 6, 1)] * 2, "rating": [1.0, 2.0]}
    )

    assert not table_exists(Table.ELO_SNAPSHOTS, settings)
    write_table(Table.ELO_SNAPSHOTS, df, settings=settings)
    assert table_exists(Table.ELO_SNAPSHOTS, settings)

    back = read_table(Table.ELO_SNAPSHOTS, settings)
    assert back.sort("team").to_dict(as_series=False) == df.sort("team").to_dict(as_series=False)


def test_append_mode_dedupes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    df = pl.DataFrame({"team": ["BRA"], "date": [date(2026, 6, 1)], "rating": [1.0]})
    write_table(Table.ELO_SNAPSHOTS, df, settings=settings)
    write_table(Table.ELO_SNAPSHOTS, df, mode="append", settings=settings)  # identical row
    assert read_table(Table.ELO_SNAPSHOTS, settings).height == 1


def test_connect_registers_views(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_table(
        Table.MATCHES,
        pl.DataFrame(
            {
                "match_id": ["m1"],
                "date": [date(2018, 6, 14)],
                "home_team": ["Russia"],
                "away_team": ["Saudi Arabia"],
                "home_goals": [5],
                "away_goals": [0],
                "competition": ["FIFA World Cup"],
                "is_knockout": [False],
                "neutral_site": [False],
                "group": [None],
            }
        ),
        settings=settings,
    )
    con = connect(settings)
    try:
        count = con.execute("SELECT COUNT(*) FROM matches").fetchone()
        assert count is not None and count[0] == 1
    finally:
        con.close()


def test_ingest_results_from_local_raw_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "results.csv").write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-06-14,Russia,Saudi Arabia,5,0,FIFA World Cup,Moscow,Russia,False\n"
    )
    n = ingest_results(settings)
    assert n == 1
    assert read_table(Table.MATCHES, settings).row(0, named=True)["home_team"] == "Russia"
