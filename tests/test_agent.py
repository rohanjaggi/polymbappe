"""Tests for the live agent: state persistence, node logic, and the cycle."""

from __future__ import annotations

from datetime import datetime, timedelta

from polymbappe.agent.nodes import (
    AgentConfig,
    assess_node,
    cross_reference_node,
    heuristic_classifier,
    reflect_node,
    run_cycle,
    severity_at_least,
)
from polymbappe.agent.sources import NewsItem, scan_sources
from polymbappe.agent.state import AgentState
from polymbappe.config import Settings

NOW = datetime(2026, 6, 1, 12, 0, 0)


def _state(tmp_path) -> AgentState:
    return AgentState(Settings(data_dir=tmp_path))


def _config() -> AgentConfig:
    return AgentConfig(player_tiers={"Mbappe": 1, "Bench Warmer": 3})


def _item(title: str, team: str | None = "France") -> NewsItem:
    return NewsItem(source="bbc_rss", timestamp=NOW, title=title, snippet="", team=team)


# -- state ---------------------------------------------------------------------


def test_state_upsert_and_cooling(tmp_path) -> None:
    with _state(tmp_path) as state:
        assert state.get_player_status("Mbappe") is None
        state.upsert_player_status("Mbappe", "France", "doubt", NOW, "bbc_rss", 0.9)
        row = state.get_player_status("Mbappe")
        assert row is not None and row["status"] == "doubt"
        assert state.recently_assessed("Mbappe", NOW + timedelta(hours=2))
        assert not state.recently_assessed("Mbappe", NOW + timedelta(hours=20))


def test_state_changelog_export(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    with AgentState(settings) as state:
        state.append_changelog(NOW, "France", "Mbappe", "injury: out", "confirmed", -0.01)
        out = state.export_changelog_parquet()
    assert out.exists()


# -- config / player tiers -----------------------------------------------------


def test_load_agent_config_empty_without_attributes(tmp_path) -> None:
    from polymbappe.agent.scheduler import load_agent_config

    config = load_agent_config(Settings(data_dir=tmp_path))
    assert config.player_tiers == {}


def test_load_agent_config_loads_tiers_from_table(tmp_path) -> None:
    import polars as pl

    from polymbappe.agent.scheduler import load_agent_config
    from polymbappe.data.store import write_table
    from polymbappe.data.tables import Table

    settings = Settings(data_dir=tmp_path)
    write_table(
        Table.PLAYER_ATTRIBUTES,
        pl.DataFrame(
            {"team": ["France", "France"], "player": ["Mbappe", "Sub"], "overall": [91, 65]}
        ),
        settings=settings,
    )
    config = load_agent_config(settings)
    # Default tier sizes (top 3 = tier 1); both of two players fall in tier 1.
    assert config.player_tiers["Mbappe"] == 1
    assert config.player_tiers["Sub"] == 1


# -- sources -------------------------------------------------------------------


def test_scan_tags_team() -> None:
    items = scan_sources(
        ["France", "Brazil"],
        injected=[_item("Mbappe injured for France", team=None)],
    )
    assert items[0].team == "France"


# -- assess --------------------------------------------------------------------


def test_severity_ordering() -> None:
    assert severity_at_least("out", "doubt")
    assert severity_at_least("doubt", "doubt")
    assert not severity_at_least("minor", "doubt")


def test_heuristic_classifier_and_assess_filter() -> None:
    config = _config()
    out_item = _item("Mbappe ruled out for the tournament with injury")
    rumor_item = _item("Mbappe reportedly carrying a minor knock")
    bench_item = _item("Bench Warmer ruled out for the tournament")

    a = heuristic_classifier(out_item, config)
    assert a is not None and a.confidence == "confirmed" and a.severity == "out" and a.tier == 1

    # Assess keeps the tier-1 confirmed out; drops the rumor (low severity/confidence)
    # and the tier-3 player.
    passed = assess_node([out_item, rumor_item, bench_item], config)
    assert [p.player for p in passed] == ["Mbappe"]


# -- cross-reference -----------------------------------------------------------


def test_cross_reference_dedup_and_cooling(tmp_path) -> None:
    config = _config()
    with _state(tmp_path) as state:
        a = assess_node([_item("Mbappe ruled out for the tournament")], config)
        net_new = cross_reference_node(a, state, NOW, config)
        assert len(net_new) == 1
        # Apply it, then the same finding is no longer net-new (already known + cooling).
        state.upsert_player_status("Mbappe", "France", "out", NOW, "bbc_rss", 0.9)
        again = cross_reference_node(a, state, NOW + timedelta(hours=1), config)
        assert again == []


# -- reflect -------------------------------------------------------------------


def test_reflect_flags_significant_shift() -> None:
    config = _config()
    sig = reflect_node({"France": 0.12}, {"France": 0.13, "Brazil": 0.10}, config)
    teams = {s["team"] for s in sig}
    assert "France" in teams  # 1pp shift > 0.5pp threshold
    assert "Brazil" not in teams  # no prior -> zero shift


# -- full cycle ----------------------------------------------------------------


def test_run_cycle_end_to_end(tmp_path) -> None:
    config = _config()
    triggered: list[bool] = []
    with _state(tmp_path) as state:
        summary = run_cycle(
            state,
            teams=["France"],
            config=config,
            now=NOW,
            injected_items=[_item("Mbappe ruled out for the tournament")],
            simulate_fn=lambda: triggered.append(True),
            prev_probs={"France": 0.12},
            new_probs={"France": 0.10},
        )
        assert summary["acted"] == 1
        assert triggered == [True]
        assert state.get_player_status("Mbappe")["status"] == "out"
        assert state.changelog_df().height == 1
        assert state.decisions_df().height == 5  # one per node
        assert len(summary["significant_shifts"]) == 1
