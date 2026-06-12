"""Tests for the data-ingestion layer (offline, via local raw files)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polymbappe.config import Settings
from polymbappe.data.ingest import (
    derive_manager_records,
    ingest_all_sources,
    ingest_elo,
    ingest_manager_records,
    ingest_market_odds,
    ingest_schedule,
    ingest_squad_valuations,
    ingest_squads,
    ingest_team_xg,
    ingest_venues,
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


# A trimmed-down clone of the Wikipedia "<tournament> squads" rendered-HTML shape: a team
# heading followed by a squad table whose header row names the Player / Date of birth / Club
# columns, plus a spanning sub-row that must be skipped.
_WIKI_SQUADS_HTML = """
<h3 id="Brazil"><span class="mw-headline">Brazil</span></h3>
<p>Head coach: Someone</p>
<table class="wikitable">
  <tr><th>No.</th><th>Pos.</th><th>Player</th><th>Date of birth (age)</th>
      <th>Caps</th><th>Club</th></tr>
  <tr><td>1</td><td>GK</td><td><a href="/wiki/Alisson">Alisson</a></td>
      <td>(1992-10-02) 2 October 1992 (aged 29)</td><td>50</td>
      <td><span class="flagicon"><a href="/wiki/England"></a></span>
          <a href="/wiki/Liverpool_F.C.">Liverpool</a></td></tr>
  <tr><td colspan="6">Coach note spanning the whole row</td></tr>
  <tr><td>10</td><td>FW</td><td><a href="/wiki/Neymar">Neymar</a></td>
      <td>(1992-02-05) 5 February 1992 (aged 30)</td><td>120</td>
      <td><a href="/wiki/Al_Hilal">Al Hilal</a></td></tr>
</table>
<h3 id="Serbia"><span class="mw-headline">Serbia</span></h3>
<table class="wikitable">
  <tr><th>No.</th><th>Player</th><th>Date of birth (age)</th><th>Club</th></tr>
  <tr><td>1</td><td><a href="/wiki/Goalkeeper">Someone Else</a></td>
      <td>(1990-01-01) 1 January 1990 (aged 32)</td>
      <td><a href="/wiki/Club">Other Club</a></td></tr>
</table>
"""


def test_parse_wikipedia_squad_extracts_player_club_age() -> None:
    from polymbappe.data.sources import _parse_wikipedia_squad

    rows = _parse_wikipedia_squad(_WIKI_SQUADS_HTML, team="Brazil", tournament="WC2022")
    # Only Brazil's two players (the spanning sub-row is skipped; Serbia is a different section).
    assert [r["player"] for r in rows] == ["Alisson", "Neymar"]
    alisson = rows[0]
    assert alisson["club"] == "Liverpool"
    assert alisson["age"] == 29.0
    assert alisson["team"] == "Brazil" and alisson["tournament"] == "WC2022"


def test_fetch_wikipedia_squad_unknown_tournament_returns_empty() -> None:
    from polymbappe.data import sources

    # No page override and the tournament isn't in WIKIPEDIA_SQUADS_PAGES -> no network, [].
    assert sources.fetch_wikipedia_squad("NOPE2099", "Brazil", min_interval=0) == []


def test_scrape_squads_falls_back_to_wikipedia(tmp_path: Path, monkeypatch) -> None:
    """When Transfermarkt yields nothing for a manifest team, Wikipedia is used instead."""

    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squads_manifest.csv").write_text(
        "tournament,team\nWC2022,Brazil\n"
    )

    tm_calls: list[tuple[str, str]] = []
    wiki_calls: list[tuple[str, str]] = []

    def _fake_tm(tournament, team, **kwargs):
        tm_calls.append((tournament, team))
        return []  # Transfermarkt unavailable

    def _fake_wiki(tournament, team, **kwargs):
        wiki_calls.append((tournament, team))
        return [{"player": "Neymar", "club": "Al Hilal", "age": 30.0,
                 "team": team, "tournament": tournament}]

    monkeypatch.setattr(ingest_mod.sources, "fetch_transfermarkt_squad", _fake_tm)
    monkeypatch.setattr(ingest_mod.sources, "fetch_wikipedia_squad", _fake_wiki)

    n = ingest_squads(settings)
    assert n == 1
    assert tm_calls == [("WC2022", "Brazil")]  # tried Transfermarkt first
    assert wiki_calls == [("WC2022", "Brazil")]  # then fell back to Wikipedia
    squads = read_table(Table.SQUADS, settings)
    assert squads.row(0, named=True)["player"] == "Neymar"


def test_scrape_squads_prefers_transfermarkt(tmp_path: Path, monkeypatch) -> None:
    """When Transfermarkt returns rows, Wikipedia is not consulted."""

    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squads_manifest.csv").write_text(
        "tournament,team,tm_id\nWC2022,Brazil,3439\n"
    )
    wiki_called = False

    def _fake_tm(tournament, team, **kwargs):
        return [{"player": "Casemiro", "club": "Manchester United", "age": 30.0,
                 "team": team, "tournament": tournament}]

    def _fake_wiki(*args, **kwargs):
        nonlocal wiki_called
        wiki_called = True
        return []

    monkeypatch.setattr(ingest_mod.sources, "fetch_transfermarkt_squad", _fake_tm)
    monkeypatch.setattr(ingest_mod.sources, "fetch_wikipedia_squad", _fake_wiki)

    assert ingest_squads(settings) == 1
    assert wiki_called is False


def test_ingest_squad_valuations_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squad_valuations.csv").write_text(
        "team,tournament,total_value,median_value,player_count\n"
        "USA,WC2018,150000000,5000000,23\n"
        "Brazil,WC2018,900000000,40000000,23\n"
    )
    n = ingest_squad_valuations(settings)
    assert n == 2
    vals = read_table(Table.SQUAD_VALUATIONS, settings)
    assert tuple(vals.columns) == TABLE_COLUMNS[Table.SQUAD_VALUATIONS]
    assert vals.schema["total_value"] == pl.Float64
    assert vals.schema["player_count"] == pl.Int64
    # team normalized via alias (USA -> United States).
    usa = vals.filter(pl.col("tournament") == "WC2018").filter(
        pl.col("total_value") == 150000000.0
    ).row(0, named=True)
    assert usa["team"] == "United States"
    assert "USA" not in set(vals["team"].to_list())


def test_ingest_squad_valuations_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_squad_valuations(settings) == 0
    assert not table_exists(Table.SQUAD_VALUATIONS, settings)


def test_scrape_squad_valuations_aggregates_transfermarkt(tmp_path: Path, monkeypatch) -> None:
    """Manifest teams are scraped from Transfermarkt and aggregated per (team, tournament)."""

    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squads_manifest.csv").write_text(
        "tournament,team,tm_id\nWC2022,Brazil,3439\n"
    )

    def _fake_tm_value(tournament, team, **kwargs):
        return [
            {"player": "Neymar", "market_value": 80_000_000.0,
             "team": team, "tournament": tournament},
            {"player": "Casemiro", "market_value": 40_000_000.0,
             "team": team, "tournament": tournament},
            # A player with no listed value still counts toward player_count, not the totals.
            {"player": "Weverton", "market_value": None,
             "team": team, "tournament": tournament},
        ]

    monkeypatch.setattr(
        ingest_mod.sources, "fetch_transfermarkt_squad_valuation", _fake_tm_value
    )

    n = ingest_squad_valuations(settings)
    assert n == 1
    vals = read_table(Table.SQUAD_VALUATIONS, settings)
    row = vals.row(0, named=True)
    assert row["team"] == "Brazil"
    assert row["total_value"] == 120_000_000.0
    assert row["median_value"] == 60_000_000.0  # median of the two valued players
    assert row["player_count"] == 3


def test_parse_transfermarkt_valuations_extracts_market_value() -> None:
    from polymbappe.data.sources import _parse_transfermarkt_valuations

    html = """
    <table class="items"><tbody>
      <tr>
        <td class="posrela">
          <table class="inline-table"><tr><td class="hauptlink">
            <a href="/neymar/profil/spieler/68290">Neymar</a>
          </td></tr></table>
        </td>
        <td class="zentriert">31</td>
        <td class="rechts hauptlink"><a href="#">€60.00m</a></td>
      </tr>
      <tr>
        <td class="posrela">
          <table class="inline-table"><tr><td class="hauptlink">
            <a href="/weverton/profil/spieler/0">Weverton</a>
          </td></tr></table>
        </td>
        <td class="zentriert">35</td>
        <td class="rechts hauptlink">-</td>
      </tr>
    </tbody></table>
    """
    rows = _parse_transfermarkt_valuations(html, team="Brazil", tournament="WC2022")
    assert [r["player"] for r in rows] == ["Neymar", "Weverton"]
    assert rows[0]["market_value"] == 60_000_000.0
    assert rows[1]["market_value"] is None
    assert rows[0]["team"] == "Brazil" and rows[0]["tournament"] == "WC2022"


def test_parse_market_value_handles_suffixes() -> None:
    from polymbappe.data.sources import _parse_market_value

    assert _parse_market_value("€80.00m") == 80_000_000.0
    assert _parse_market_value("€500k") == 500_000.0
    assert _parse_market_value("€1.20bn") == 1_200_000_000.0
    assert _parse_market_value("-") is None
    assert _parse_market_value("") is None
    assert _parse_market_value("n/a") is None


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
    # venues/schedule prefer a local CSV (mirrors results.csv) so the run stays offline.
    (settings.raw_data_dir / "venues.csv").write_text(
        "venue,city,country,latitude,longitude\n"
        "Estadio Azteca,Mexico City,mx,19.3029,-99.1505\n"
        "SoFi Stadium,Los Angeles (Inglewood),us,33.953,-118.339\n"
    )
    (settings.raw_data_dir / "schedule.csv").write_text(
        "match_id,date,stage,group,home_team,away_team,city\n"
        "x,2026-06-11,Matchday 1,A,Mexico,South Africa,Mexico City\n"
    )
    (settings.raw_data_dir / "city_coords.csv").write_text(
        "city,country,latitude,longitude,population\n"
        "moscow,RU,55.7522,37.6156,10381222\n"
    )
    report = ingest_all_sources(settings=settings)
    assert report["results"] == 3
    assert report["elo"] == 6
    assert report["market_odds"] == 1
    assert report["team_xg"] == 0  # optional, no file -> skipped cleanly
    assert report["venues"] == 2
    assert report["schedule"] == 1
    assert report["city_coords"] == 1
    assert report["squads"] == 0  # optional, no file/scraper -> skipped cleanly
    assert report["manager_records"] == 0  # optional, no file/scraper -> skipped cleanly
    for table in (Table.MATCHES, Table.ELO_SNAPSHOTS, Table.MARKET_ODDS):
        assert table_exists(table, settings)


# -- venues + schedule (openfootball) ------------------------------------------

def test_ingest_venues_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "venues.csv").write_text(
        "venue,city,country,latitude,longitude\n"
        "Estadio Azteca,Mexico City,mx,19.3029,-99.1505\n"
        "SoFi Stadium,Los Angeles (Inglewood),us,33.953,-118.339\n"
    )
    n = ingest_venues(settings)
    assert n == 2
    venues = read_table(Table.VENUES, settings)
    assert set(venues.columns) == set(TABLE_COLUMNS[Table.VENUES])
    assert venues.filter(pl.col("city") == "Mexico City").row(0, named=True)["latitude"] == 19.3029


def test_ingest_venues_skips_when_absent(tmp_path: Path, monkeypatch) -> None:
    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    # No local file and the openfootball feed yields nothing -> clean skip (no network).
    monkeypatch.setattr(ingest_mod.sources, "fetch_openfootball_stadiums", lambda **k: [])
    assert ingest_venues(settings) == 0
    assert not table_exists(Table.VENUES, settings)


def test_ingest_venues_from_openfootball(tmp_path: Path, monkeypatch) -> None:
    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)

    def _fake_stadiums(**kwargs):
        return [
            {"city": "Mexico City", "cc": "mx", "name": "Estadio Azteca",
             "coords": "19°18'11\"N 99°09'02\"W"},
            {"city": "Vancouver", "cc": "ca", "name": "BC Place",
             "coords": "49°16'36\"N 123°6'43\"W"},
        ]

    monkeypatch.setattr(ingest_mod.sources, "fetch_openfootball_stadiums", _fake_stadiums)
    n = ingest_venues(settings)
    assert n == 2
    venues = read_table(Table.VENUES, settings)
    van = venues.filter(pl.col("city") == "Vancouver").row(0, named=True)
    assert van["latitude"] == pytest.approx(49.2767, abs=1e-3)


def test_ingest_schedule_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "schedule.csv").write_text(
        "match_id,date,stage,group,home_team,away_team,city\n"
        "seed,2026-06-11,Matchday 1,A,Mexico,South Africa,Mexico City\n"
    )
    n = ingest_schedule(settings)
    assert n == 1
    sched = read_table(Table.SCHEDULE, settings)
    assert set(sched.columns) == set(TABLE_COLUMNS[Table.SCHEDULE])
    row = sched.row(0, named=True)
    # match_id is rebuilt on the date__home__away convention (not the CSV's "seed").
    assert row["match_id"] == "2026-06-11__Mexico__South Africa"
    assert sched.schema["date"] == pl.Date


def test_ingest_schedule_skips_when_absent(tmp_path: Path, monkeypatch) -> None:
    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    # No local file and the openfootball feed yields nothing -> clean skip (no network).
    monkeypatch.setattr(ingest_mod.sources, "fetch_openfootball_schedule", lambda **k: [])
    assert ingest_schedule(settings) == 0
    assert not table_exists(Table.SCHEDULE, settings)


def test_ingest_schedule_from_openfootball(tmp_path: Path, monkeypatch) -> None:
    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)

    def _fake_matches(**kwargs):
        return [
            {"round": "Matchday 1", "date": "2026-06-11", "team1": "Mexico",
             "team2": "South Africa", "group": "Group A", "ground": "Mexico City"},
            {"round": "Round of 32", "date": "2026-06-28", "team1": "2A", "team2": "2B",
             "ground": "Los Angeles (Inglewood)"},
        ]

    monkeypatch.setattr(ingest_mod.sources, "fetch_openfootball_schedule", _fake_matches)
    n = ingest_schedule(settings)
    assert n == 2
    sched = read_table(Table.SCHEDULE, settings)
    assert sched.filter(pl.col("stage") == "Matchday 1").row(0, named=True)["group"] == "A"


# -- city gazetteer (GeoNames) -------------------------------------------------

def test_ingest_city_coords_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "city_coords.csv").write_text(
        "city,country,latitude,longitude,population\n"
        "Moscow,RU,55.7522,37.6156,10381222\n"
        "London,GB,51.5085,-0.1257,8961989\n"
    )
    from polymbappe.data.ingest import ingest_city_coords

    n = ingest_city_coords(settings)
    assert n == 2
    coords = read_table(Table.CITY_COORDS, settings)
    assert set(coords.columns) == set(TABLE_COLUMNS[Table.CITY_COORDS])
    # city is lower-cased on ingest so the resolver keys match.
    assert "moscow" in coords["city"].to_list()


def test_ingest_city_coords_skips_when_absent(tmp_path: Path, monkeypatch) -> None:
    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    # No local file and an empty GeoNames fetch -> clean skip (no network).
    monkeypatch.setattr(
        ingest_mod.sources, "fetch_geonames_cities", lambda *a, **k: pl.DataFrame()
    )
    from polymbappe.data.ingest import ingest_city_coords

    assert ingest_city_coords(settings) == 0
    assert not table_exists(Table.CITY_COORDS, settings)


def test_ingest_results_preserves_city_country(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    matches = read_table(Table.MATCHES, settings)
    assert {"city", "country"}.issubset(matches.columns)
    moscow = matches.filter(pl.col("home_team") == "Russia").sort("date").row(0, named=True)
    assert moscow["city"] == "Moscow"
