"""Tests for player-importance tiering (agent input, not model features)."""

from __future__ import annotations

import polars as pl

from polymbappe.data.normalize import normalize_player_attributes
from polymbappe.features.players import build_player_tiers, player_tier_map


def _attrs() -> pl.DataFrame:
    # Two teams; ratings chosen so tier boundaries are unambiguous with small tier sizes.
    return pl.DataFrame(
        {
            "team": ["France"] * 4 + ["Brazil"] * 3,
            "player": [
                "Mbappe",
                "Griezmann",
                "Tchouameni",
                "Backup",
                "Vinicius",
                "Rodrygo",
                "Sub",
            ],
            "overall": [91, 86, 84, 70, 89, 85, 68],
        }
    )


def test_build_player_tiers_ranks_within_team() -> None:
    tiers = build_player_tiers(_attrs(), tier1_size=1, tier2_size=1)
    assert tuple(tiers.columns) == ("team", "player", "overall", "tier")

    by_player = {r["player"]: r["tier"] for r in tiers.iter_rows(named=True)}
    # tier1_size=1 -> only the top player per team is tier 1.
    assert by_player["Mbappe"] == 1
    assert by_player["Vinicius"] == 1
    # tier2_size=1 -> the second-best per team is tier 2.
    assert by_player["Griezmann"] == 2
    assert by_player["Rodrygo"] == 2
    # everyone else is tier 3.
    assert by_player["Tchouameni"] == 3
    assert by_player["Backup"] == 3
    assert by_player["Sub"] == 3


def test_build_player_tiers_collapses_duplicate_editions() -> None:
    # Same player twice (two FIFA editions); the higher rating should win, one row out.
    dupes = pl.DataFrame(
        {"team": ["France", "France"], "player": ["Mbappe", "Mbappe"], "overall": [88, 91]}
    )
    tiers = build_player_tiers(dupes)
    assert tiers.height == 1
    assert tiers.row(0, named=True)["overall"] == 91


def test_build_player_tiers_empty() -> None:
    out = build_player_tiers(
        pl.DataFrame(schema={"team": pl.Utf8, "player": pl.Utf8, "overall": pl.Int64})
    )
    assert out.height == 0
    assert tuple(out.columns) == ("team", "player", "overall", "tier")


def test_player_tier_map_flattens_to_dict() -> None:
    mapping = player_tier_map(_attrs(), tier1_size=1, tier2_size=1)
    assert mapping["Mbappe"] == 1
    assert mapping["Rodrygo"] == 2
    assert mapping["Sub"] == 3


def test_player_tier_map_keeps_most_important_on_name_collision() -> None:
    # Same name on two teams at different tiers -> the lower (more important) tier wins.
    collide = pl.DataFrame(
        {
            "team": ["A", "A", "B", "B"],
            "player": ["Star", "Filler", "Star", "Filler2"],
            "overall": [90, 80, 60, 50],
        }
    )
    mapping = player_tier_map(collide, tier1_size=1, tier2_size=0)
    # "Star" is tier 1 for team A but tier 3 for team B (tier2_size=0) -> keep tier 1.
    assert mapping["Star"] == 1


def test_normalize_player_attributes_reconciles_fm_columns() -> None:
    # Football Manager-style column names differ from EA FC; the reconciler resolves both.
    raw = pl.DataFrame(
        {"Name": ["Player A", "  "], "Nation": ["Spain", "Italy"], "CA": [180, None]}
    )
    out = normalize_player_attributes(raw)
    assert tuple(out.columns) == ("team", "player", "overall")
    # Blank name and null overall rows are dropped.
    assert out.height == 1
    assert out.row(0, named=True) == {"team": "Spain", "player": "Player A", "overall": 180}
