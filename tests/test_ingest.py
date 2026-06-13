"""Tests for the data-ingestion layer (offline, via local raw files)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl

from polymbappe.config import Settings
from polymbappe.data import ingest as ingest_mod
from polymbappe.data.ingest import (
    derive_manager_records,
    ingest_all_sources,
    ingest_elo,
    ingest_manager_records,
    ingest_market_odds,
    ingest_player_attributes,
    ingest_ppda,
    ingest_squad_valuations,
    ingest_squads,
    ingest_team_xg,
)
from polymbappe.data.store import read_parquet, read_table, table_exists, write_table
from polymbappe.data.tables import TABLE_COLUMNS, Table
from polymbappe.features.pipeline import build_feature_matrix

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


# A trimmed EloRatings.net ranking table: each row's first anchor is the team and the
# first standalone integer cell its rating (see normalize.parse_eloratings).
_ELO_HTML = """
<table>
  <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
  <tr><td>1</td><td><a href="/Brazil">Brazil</a></td><td>2169</td></tr>
  <tr><td>2</td><td><a href="/USA">USA</a></td><td>1821</td></tr>
</table>
"""

# Trimmed EloRatings.net backend feeds: World.tsv has the 2-letter team code in column 3
# and the rating in column 4; en.teams.tsv maps each code to its English name (the first
# name column). "US" -> "USA" exercises the USA -> United States alias downstream, and the
# unused "XX" code confirms a code with no World.tsv row is harmless.
_ELO_WORLD_TSV = "1\t1\tBR\t2169\t1\t2200\n2\t2\tUS\t1821\t1\t1850\n"
_ELO_TEAMS_TSV = "BR\tBrazil\nUS\tUSA\nXX\tNowhere\n"


def test_ingest_elo_prefers_local_tsv(tmp_path: Path) -> None:
    from datetime import date

    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    (settings.raw_data_dir / "elo_world.tsv").write_text(_ELO_WORLD_TSV)
    (settings.raw_data_dir / "elo_teams.tsv").write_text(_ELO_TEAMS_TSV)

    n = ingest_elo(settings, as_of=date(2026, 6, 1))
    assert n == 2  # published TSV snapshot, not the 6-row self-computed series
    snaps = read_table(Table.ELO_SNAPSHOTS, settings)
    assert set(snaps.columns) == set(TABLE_COLUMNS[Table.ELO_SNAPSHOTS])
    assert set(snaps["date"].to_list()) == {date(2026, 6, 1)}
    # Code resolved (US -> "USA") and alias canonicalized (USA -> United States).
    usa = snaps.filter(pl.col("team") == "United States").row(0, named=True)
    assert usa["rating"] == 1821.0
    assert "USA" not in set(snaps["team"].to_list())
    assert snaps.filter(pl.col("team") == "Brazil").row(0, named=True)["rating"] == 2169.0


def test_ingest_elo_prefers_published_local(tmp_path: Path) -> None:
    from datetime import date

    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    (settings.raw_data_dir / "elo.html").write_text(_ELO_HTML)

    n = ingest_elo(settings, as_of=date(2026, 6, 1))
    assert n == 2  # published snapshot, not the 6-row self-computed series
    snaps = read_table(Table.ELO_SNAPSHOTS, settings)
    assert set(snaps.columns) == set(TABLE_COLUMNS[Table.ELO_SNAPSHOTS])
    # All rows stamped with the as_of date; team alias canonicalized (USA -> United States).
    assert set(snaps["date"].to_list()) == {date(2026, 6, 1)}
    usa = snaps.filter(pl.col("team") == "United States").row(0, named=True)
    assert usa["rating"] == 1821.0  # published value, not self-computed
    assert "USA" not in set(snaps["team"].to_list())


def test_ingest_elo_published_empty_falls_back(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_results(settings)
    from polymbappe.data.ingest import ingest_results

    ingest_results(settings)
    # A page with no parseable rating rows (JS-populated table) -> self-compute fallback.
    (settings.raw_data_dir / "elo.html").write_text("<html><body>loading...</body></html>")

    n = ingest_elo(settings)
    assert n == 6  # fell back to the self-computed 3-matches x 2-teams series


def test_ingest_elo_fetches_when_url_configured(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    # No local matches and no local elo files: the only source is the opt-in TSV fetch.
    (settings.raw_data_dir / "elo_url.txt").write_text("https://www.eloratings.net/World.tsv\n")

    calls: list[str] = []

    def _fake_fetch(
        world_url: str = "", teams_url: str = "", timeout: float = 20.0
    ) -> tuple[str, str]:
        calls.append(world_url)
        return _ELO_WORLD_TSV, _ELO_TEAMS_TSV

    monkeypatch.setattr(ingest_mod.sources, "fetch_eloratings_tsv", _fake_fetch)

    n = ingest_elo(settings, as_of=date(2026, 6, 1))
    assert n == 2
    assert calls == ["https://www.eloratings.net/World.tsv"]  # used the configured World.tsv URL
    snaps = read_table(Table.ELO_SNAPSHOTS, settings)
    assert snaps.filter(pl.col("team") == "Brazil").row(0, named=True)["rating"] == 2169.0
    assert snaps.filter(pl.col("team") == "United States").row(0, named=True)["rating"] == 1821.0


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


def test_ingest_team_xg_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_team_xg(settings) == 0
    assert not table_exists(Table.TEAM_XG, settings)


def test_ingest_ppda_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "team_ppda.csv").write_text(
        "team,date,ppda\nUSA,2018-06-14,8.5\n"  # "USA" -> canonicalized to "United States"
    )
    n = ingest_ppda(settings)
    assert n == 1
    ppda = read_table(Table.TEAM_PPDA, settings)
    assert set(ppda.columns) == set(TABLE_COLUMNS[Table.TEAM_PPDA])
    row = ppda.row(0, named=True)
    assert row["team"] == "United States" and row["ppda"] == 8.5


def test_ingest_ppda_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_ppda(settings) == 0
    assert not table_exists(Table.TEAM_PPDA, settings)


# -- StatsBomb open-data live path (synthetic events, no network) ---------------

def _sb_event(type_name, team, *, x=None, xg=None, duel_type=None, period=1):
    e = {"type": {"name": type_name}, "team": {"name": team}, "period": period}
    if x is not None:
        e["location"] = [x, 40.0]
    if xg is not None:
        e["shot"] = {"statsbomb_xg": xg}
    if duel_type is not None:
        e["duel"] = {"type": {"name": duel_type}}
    return e


# One synthetic match (zone_fraction=0.6 -> build-up x<=72, pressing x>=48):
#   xG: Home 0.3+0.2=0.5 (shootout 0.8 in period 5 excluded), Away 0.1
#   PPDA_home = Away build-up passes (6) / Home defensive actions (3) = 2.0
#   PPDA_away = Home build-up passes (4) / Away defensive actions (2) = 2.0
_SB_EVENTS = [
    _sb_event("Shot", "Home", xg=0.3),
    _sb_event("Shot", "Home", xg=0.2),
    _sb_event("Shot", "Away", xg=0.1),
    _sb_event("Shot", "Home", xg=0.8, period=5),  # shootout -> excluded
    *[_sb_event("Pass", "Away", x=30.0) for _ in range(6)],
    *[_sb_event("Interception", "Home", x=60.0) for _ in range(3)],
    *[_sb_event("Pass", "Home", x=20.0) for _ in range(4)],
    *[_sb_event("Duel", "Away", x=55.0, duel_type="Tackle") for _ in range(2)],
    _sb_event("Pass", "Away", x=90.0),  # final-third pass -> not build-up
    _sb_event("Duel", "Home", x=60.0, duel_type="Aerial Lost"),  # aerial -> not a tackle
]
_SB_MATCHES = [
    {
        "match_id": 999,
        "match_date": "2022-12-01",
        "home_team": {"home_team_name": "Home"},
        "away_team": {"away_team_name": "Away"},
    }
]


def _patch_statsbomb(monkeypatch) -> None:
    monkeypatch.setattr(ingest_mod.sources, "STATSBOMB_COMPETITIONS", ((43, 3),))
    monkeypatch.setattr(
        ingest_mod.sources, "fetch_statsbomb_matches", lambda *a, **k: _SB_MATCHES
    )
    monkeypatch.setattr(
        ingest_mod.sources, "fetch_statsbomb_events", lambda *a, **k: _SB_EVENTS
    )


def test_ingest_team_xg_live_statsbomb(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    _patch_statsbomb(monkeypatch)
    n = ingest_team_xg(settings, live=True)
    assert n == 2  # one match -> two team rows
    xg = read_table(Table.TEAM_XG, settings)
    assert set(xg.columns) == set(TABLE_COLUMNS[Table.TEAM_XG])
    home = xg.filter(pl.col("team") == "Home").row(0, named=True)
    assert round(home["xg"], 4) == 0.5 and round(home["xga"], 4) == 0.1
    away = xg.filter(pl.col("team") == "Away").row(0, named=True)
    assert round(away["xg"], 4) == 0.1 and round(away["xga"], 4) == 0.5


def test_ingest_ppda_live_statsbomb(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    _patch_statsbomb(monkeypatch)
    n = ingest_ppda(settings, live=True)
    assert n == 2
    ppda = read_table(Table.TEAM_PPDA, settings)
    assert set(ppda.columns) == set(TABLE_COLUMNS[Table.TEAM_PPDA])
    assert ppda.filter(pl.col("team") == "Home").row(0, named=True)["ppda"] == 2.0
    assert ppda.filter(pl.col("team") == "Away").row(0, named=True)["ppda"] == 2.0


def test_ingest_team_xg_skips_without_live(tmp_path: Path) -> None:
    # No local CSV and live not requested -> clean skip, no StatsBomb pull, no table.
    settings = _settings(tmp_path)
    assert ingest_team_xg(settings, live=False) == 0
    assert not table_exists(Table.TEAM_XG, settings)


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


_MATCHES_SCHEMA = {
    "match_id": pl.Utf8,
    "date": pl.Date,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_goals": pl.Int64,
    "away_goals": pl.Int64,
    "competition": pl.Utf8,
    "is_knockout": pl.Boolean,
    "neutral_site": pl.Boolean,
    "group": pl.Utf8,
}


def test_ingest_squad_valuations_then_build_features_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    """Ingest squad valuations from raw, then build the core matrix: the Tier-1 squad
    value ratio must reach the written feature table, end to end."""

    settings = _settings(tmp_path)
    # A WC2018 fixture (window 2018-06-14..07-15). Team names are already canonical so they
    # line up with the valuation table after normalization (USA -> United States).
    matches = pl.DataFrame(
        {
            "match_id": ["wc18_1"],
            "date": [date(2018, 6, 17)],
            "home_team": ["Brazil"],
            "away_team": ["United States"],
            "home_goals": [2],
            "away_goals": [0],
            "competition": ["FIFA World Cup"],
            "is_knockout": [False],
            "neutral_site": [True],
            "group": ["E"],
        },
        schema=_MATCHES_SCHEMA,
    )
    write_table(Table.MATCHES, matches, settings=settings)
    (settings.raw_data_dir / "squad_valuations.csv").write_text(
        "team,tournament,total_value,median_value,player_count\n"
        "Brazil,WC2018,900000000,40000000,23\n"
        "USA,WC2018,150000000,5000000,23\n"
    )
    assert ingest_squad_valuations(settings) == 2

    # build_feature_matrix resolves its own Settings(); point it at tmp_path via the env.
    monkeypatch.setenv("POLYMBAPPE_DATA_DIR", str(tmp_path))
    build_feature_matrix()

    matrix = read_parquet(settings.processed_data_dir / "core_features.parquet")
    assert "squad_value_ratio" in matrix.columns
    row = matrix.row(0, named=True)
    assert row["home_log_total_value"] == math.log1p(900000000.0)
    assert row["away_log_total_value"] == math.log1p(150000000.0)
    # log(value_home / value_away) > 0: Brazil (home) is more valuable than the USA (away).
    assert row["squad_value_ratio"] == math.log1p(900000000.0) - math.log1p(150000000.0)
    assert row["squad_value_ratio"] > 0


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


def test_player_name_key_folds_accents_punct_case() -> None:
    from polymbappe.data.ingest import _player_name_key

    keys = (
        pl.DataFrame({"p": ["Kylian Mbappé", "N'Golo  Kanté", "SON Heung-min", "—"]})
        .select(_player_name_key("p").alias("k"))["k"]
        .to_list()
    )
    assert keys[0] == "kylian mbappe"
    assert keys[1] == "n golo kante"
    assert keys[2] == "son heung min"
    assert keys[3] == ""  # punctuation-only reduces to empty (dropped by the caller)


def test_value_squads_from_kaggle_point_in_time_and_citizenship(
    tmp_path: Path, monkeypatch
) -> None:
    """Kaggle valuations are joined onto rosters point-in-time (latest value on/before the
    tournament start) and disambiguated by citizenship; unmatched players count toward
    player_count only."""

    settings = _settings(tmp_path)
    (settings.raw_data_dir / "squad_valuations_kaggle.txt").write_text(
        "davidcariboo/player-scores\n"
    )
    write_table(
        Table.SQUADS,
        pl.DataFrame(
            {
                "team": ["Brazil", "Brazil", "Brazil"],
                "tournament": ["WC2018", "WC2018", "WC2018"],
                "player": ["Neymar", "Casemiro", "Ghost Player"],
                "club": ["PSG", "Madrid", "Nowhere"],
                "age": [26.0, 26.0, 30.0],
            }
        ).select(TABLE_COLUMNS[Table.SQUADS]),
        settings=settings,
    )

    def _fake_kaggle_values(*args, **kwargs):
        return pl.DataFrame(
            {
                "player": ["Neymar", "Neymar", "Neymar", "Casemiro", "Casemiro"],
                "country_of_citizenship": ["Brazil", "Brazil", "Brazil", "Brazil", "Spain"],
                "date": [
                    date(2017, 1, 1),
                    date(2018, 5, 1),  # latest on/before WC2018 start (2018-06-14)
                    date(2019, 1, 1),  # after the cutoff -> excluded
                    date(2018, 5, 1),
                    date(2018, 5, 1),  # wrong citizenship -> excluded by the team join
                ],
                "market_value_eur": [90e6, 100e6, 120e6, 40e6, 999e6],
            }
        )

    monkeypatch.setattr(
        ingest_mod.sources, "fetch_kaggle_player_valuations", _fake_kaggle_values
    )
    # The Transfermarkt scrape must NOT be reached when the Kaggle path yields rows.
    def _boom(*args, **kwargs):  # pragma: no cover - asserts it is never called
        raise AssertionError("Transfermarkt scrape should not run when Kaggle yields rows")

    monkeypatch.setattr(ingest_mod.sources, "fetch_transfermarkt_squad_valuation", _boom)

    assert ingest_squad_valuations(settings) == 1
    row = read_table(Table.SQUAD_VALUATIONS, settings).row(0, named=True)
    assert row["team"] == "Brazil"
    assert row["total_value"] == 140_000_000.0  # 100M (point-in-time) + 40M; Spain row excluded
    assert row["median_value"] == 70_000_000.0  # median(100M, 40M)
    assert row["player_count"] == 3  # Ghost Player matched nothing but still counts


def test_ingest_player_attributes_from_local(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.raw_data_dir / "player_attributes.csv").write_text(
        "team,player,overall\nUSA,Christian Pulisic,82\nBrazil,Vinicius Junior,89\n"
    )
    n = ingest_player_attributes(settings)
    assert n == 2
    attrs = read_table(Table.PLAYER_ATTRIBUTES, settings)
    assert tuple(attrs.columns) == TABLE_COLUMNS[Table.PLAYER_ATTRIBUTES]
    assert attrs.schema["overall"] == pl.Int64
    # team normalized via alias (USA -> United States).
    pulisic = attrs.filter(pl.col("player") == "Christian Pulisic").row(0, named=True)
    assert pulisic["team"] == "United States"
    assert "USA" not in set(attrs["team"].to_list())


def test_ingest_player_attributes_skips_when_absent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert ingest_player_attributes(settings) == 0
    assert not table_exists(Table.PLAYER_ATTRIBUTES, settings)


def test_ingest_player_attributes_from_kaggle(tmp_path: Path, monkeypatch) -> None:
    """The Kaggle config file triggers a fetch whose EA-FC columns are reconciled."""

    from polymbappe.data import ingest as ingest_mod

    settings = _settings(tmp_path)
    (settings.raw_data_dir / "player_attributes_kaggle.txt").write_text(
        "stefanoleone992/fc-24\nfile=male_players.csv\n"
    )

    def _fake_kaggle(dataset, *, file=None):
        assert dataset == "stefanoleone992/fc-24"
        assert file == "male_players.csv"
        # Raw EA FC schema: short_name / nationality_name / overall (+ noise columns).
        return pl.DataFrame(
            {
                "short_name": ["L. Messi", "K. Mbappé"],
                "nationality_name": ["Argentina", "France"],
                "overall": [90, 91],
                "club_name": ["Inter Miami", "Real Madrid"],
            }
        )

    monkeypatch.setattr(ingest_mod.sources, "fetch_kaggle_player_attributes", _fake_kaggle)

    n = ingest_player_attributes(settings)
    assert n == 2
    attrs = read_table(Table.PLAYER_ATTRIBUTES, settings)
    assert tuple(attrs.columns) == TABLE_COLUMNS[Table.PLAYER_ATTRIBUTES]
    assert set(attrs["player"].to_list()) == {"L. Messi", "K. Mbappé"}
    mbappe = attrs.filter(pl.col("player") == "K. Mbappé").row(0, named=True)
    assert mbappe["team"] == "France"
    assert mbappe["overall"] == 91


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
    report = ingest_all_sources(settings=settings)
    assert report["results"] == 3
    assert report["elo"] == 6
    assert report["market_odds"] == 1
    assert report["team_xg"] == 0  # optional, no file -> skipped cleanly
    assert report["team_ppda"] == 0  # optional, no file -> skipped cleanly
    assert report["squads"] == 0  # optional, no file/scraper -> skipped cleanly
    assert report["manager_records"] == 0  # optional, no file/scraper -> skipped cleanly
    for table in (Table.MATCHES, Table.ELO_SNAPSHOTS, Table.MARKET_ODDS):
        assert table_exists(table, settings)
