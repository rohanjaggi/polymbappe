"""Tests for the Monte Carlo tournament engine."""

from __future__ import annotations

import numpy as np
import polars as pl

from polymbappe.simulate.structure import (
    build_structure,
    placeholder_structure_2026,
    pot_seed_groups,
    structure_from_strengths,
    team_strengths,
)
from polymbappe.simulate.tournament import (
    STAGES,
    StalenessMonitor,
    StrengthModel,
    compute_match_predictions,
    simulate_tournament,
    surprise_increment,
)


def _model(teams: list[str]) -> StrengthModel:
    # Descending attack strength so favorites are well-defined.
    attack = {t: 0.6 - 0.025 * i for i, t in enumerate(teams)}
    defense = {t: -0.3 + 0.012 * i for i, t in enumerate(teams)}
    return StrengthModel(attack=attack, defense=defense, home_advantage=0.0, rho=-0.03)


def test_simulation_runs_and_probabilities_normalize() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)
    result = simulate_tournament(structure, model, n_sims=200, rng=np.random.default_rng(0))

    stage = result.stage_probabilities()
    assert stage.height == 48
    # Exactly 32 teams reach R32 each sim -> mean R32 prob across teams = 32/48.
    assert abs(stage["R32"].sum() - 32.0) < 1e-6
    # Exactly one champion per sim.
    assert abs(stage["champion"].sum() - 1.0) < 1e-6
    # Stage probabilities are monotonically non-increasing R32 >= R16 >= ... per team.
    row = stage.row(0, named=True)
    chain = [row[s] for s in STAGES]
    assert all(chain[i] >= chain[i + 1] - 1e-9 for i in range(len(chain) - 1))


def test_host_bonus_lifts_host_rate_at_neutral_venue() -> None:
    # Two equal teams; only the host membership differs. The host should out-score.
    model = StrengthModel(
        attack={"USA": 0.0, "Brazil": 0.0},
        defense={"USA": 0.0, "Brazil": 0.0},
        home_advantage=0.25,
        hosts=frozenset({"USA"}),
        host_bonus=0.15,
    )
    # Host as the scheduled "away" team still gets the bonus (group order is arbitrary).
    lam, mu = model.rates("Brazil", "USA", neutral=True)
    assert mu > lam
    assert abs(mu - np.exp(0.15)) < 1e-9  # away host carries exactly the bonus
    assert abs(lam - 1.0) < 1e-9  # non-host baseline unchanged
    # Disabling the bonus collapses the gap (proves it is the host_bonus doing the work).
    flat = StrengthModel(
        attack={"USA": 0.0, "Brazil": 0.0},
        defense={"USA": 0.0, "Brazil": 0.0},
        hosts=frozenset({"USA"}),
        host_bonus=0.0,
    )
    assert flat.rates("Brazil", "USA", neutral=True) == (1.0, 1.0)


def test_hosts_reach_knockout_more_with_bonus() -> None:
    structure = placeholder_structure_2026()
    teams = structure.teams
    base = _model(teams)  # uniform-ish strengths, no hosts
    # Promote a mid-pack team to host with a sizeable bonus.
    host = teams[24]
    hosted = StrengthModel(
        attack=base.attack, defense=base.defense, home_advantage=0.0, rho=base.rho,
        hosts=frozenset({host}), host_bonus=0.5,
    )
    r_base = simulate_tournament(structure, base, n_sims=600, rng=np.random.default_rng(7))
    r_host = simulate_tournament(structure, hosted, n_sims=600, rng=np.random.default_rng(7))
    p_base = {r["team"]: r["R16"] for r in r_base.stage_probabilities().iter_rows(named=True)}
    p_host = {r["team"]: r["R16"] for r in r_host.stage_probabilities().iter_rows(named=True)}
    assert p_host[host] > p_base[host]


def test_group_finish_probabilities_sum_to_one() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)
    result = simulate_tournament(structure, model, n_sims=150, rng=np.random.default_rng(2))
    gp = result.group_probabilities()
    totals = gp.select(["finish_1", "finish_2", "finish_3", "finish_4"]).sum_horizontal()
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-6)


def test_favorites_win_more_than_minnows() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)
    result = simulate_tournament(structure, model, n_sims=400, rng=np.random.default_rng(5))
    champ = {r["team"]: r["champion"] for r in result.stage_probabilities().iter_rows(named=True)}
    # Strongest team (Team01) wins more often than the weakest (Team48).
    assert champ["Team01"] > champ["Team48"]


def test_build_structure_validates() -> None:
    try:
        build_structure({"A": ["x", "y", "z"]})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for malformed structure")


class _FakeDC:
    """Minimal stand-in for a fitted DixonColesModel (attack/defense by team index)."""

    def __init__(self, n: int) -> None:
        self.index_to_team = [f"Nat{i:02d}" for i in range(n)]
        self.team_to_index = {t: i for i, t in enumerate(self.index_to_team)}
        # Descending strength: team 0 strongest (high attack, low defense).
        self.attack = np.array([1.0 - 0.02 * i for i in range(n)])
        self.defense = np.array([-0.5 + 0.015 * i for i in range(n)])


def test_team_strengths_ordering() -> None:
    s = team_strengths(_FakeDC(5))
    assert s["Nat00"] > s["Nat04"]  # strongest first


def test_pot_seed_groups_balanced() -> None:
    ranked = [f"T{i:02d}" for i in range(48)]
    groups = pot_seed_groups(ranked)
    assert len(groups) == 12 and all(len(v) == 4 for v in groups.values())
    # Each group spans all four pots (one from each strength twelfth).
    g = groups["A"]
    assert g == ["T00", "T12", "T24", "T36"]
    # All 48 teams used exactly once.
    flat = [t for v in groups.values() for t in v]
    assert sorted(flat) == ranked


def test_structure_from_strengths_uses_real_teams_and_elo_seeds() -> None:
    dc = _FakeDC(60)  # more than 48 -> top 48 selected
    structure = structure_from_strengths(dc)
    assert len(structure.teams) == 48
    assert "Nat00" in structure.teams  # strongest team qualifies
    assert "Nat59" not in structure.teams  # weakest 12 dropped
    assert len(structure.elo) == 48  # pseudo-Elo attached for the upset floor

    # With real Elo, ranking follows Elo, not model strength.
    elo = {t: float(i) for i, t in enumerate(dc.index_to_team)}  # Nat59 highest
    by_elo = structure_from_strengths(dc, elo=elo)
    assert "Nat59" in by_elo.teams


def test_structure_from_strengths_requires_48() -> None:
    try:
        structure_from_strengths(_FakeDC(20))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError with fewer than 48 teams")


def test_compute_match_predictions_schema_and_simplex() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)
    preds = compute_match_predictions(structure, model)
    assert preds.height == 12 * 6  # 12 groups x 6 round-robin fixtures
    hda = preds.select("model_home", "model_draw", "model_away").to_numpy()
    assert np.allclose(hda.sum(axis=1), 1.0, atol=1e-9)
    assert preds["match_id"].to_list()[0].startswith("2026__")
    assert (preds["exp_home_goals"] > 0).all()


def test_context_hook_shifts_predictions() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)
    base = compute_match_predictions(structure, model)

    # Hook that pushes 3pp of mass from away onto home for every match.
    def hook(home, away, base_hda):
        adj = base_hda + np.array([0.03, 0.0, -0.03])
        return adj / adj.sum()

    ctx = compute_match_predictions(structure, model, context_hook=hook)
    # Home probability should rise on average once context nudges it up.
    assert ctx["model_home"].mean() > base["model_home"].mean()
    hda = ctx.select("model_home", "model_draw", "model_away").to_numpy()
    assert np.allclose(hda.sum(axis=1), 1.0, atol=1e-9)


def test_simulate_with_context_hook_runs() -> None:
    structure = placeholder_structure_2026()
    model = _model(structure.teams)

    def hook(home, away, base_hda):
        return base_hda  # identity hook: must not change invariants

    result = simulate_tournament(
        structure, model, n_sims=100, rng=np.random.default_rng(3), context_hook=hook
    )
    sp = result.stage_probabilities()
    assert abs(sp["champion"].sum() - 1.0) < 1e-6


def test_write_edges_graceful_without_market(tmp_path) -> None:
    import structlog

    from polymbappe.config import Settings
    from polymbappe.simulate.tournament import _write_edges

    preds = pl.DataFrame({
        "match_id": ["2026__A__B"], "model_home": [0.6], "model_draw": [0.25], "model_away": [0.15],
    })
    settings = Settings(data_dir=tmp_path)
    edges = _write_edges(preds, settings, structlog.get_logger())
    assert edges.is_empty()  # no market_odds table -> empty (right schema)
    assert "edge" in edges.columns


def test_write_edges_with_market(tmp_path) -> None:
    import structlog

    from polymbappe.config import Settings
    from polymbappe.data.store import write_table
    from polymbappe.data.tables import Table
    from polymbappe.simulate.tournament import _write_edges

    settings = Settings(data_dir=tmp_path)
    preds = pl.DataFrame({
        "match_id": ["2026__A__B"],
        "model_home": [0.62], "model_draw": [0.23], "model_away": [0.15],
    })
    market = pl.DataFrame({
        "match_id": ["2026__A__B"], "source": ["x"],
        "home_win_prob": [0.50], "draw_prob": [0.27], "away_win_prob": [0.23],
        "timestamp": [None],
    }).with_columns(pl.col("timestamp").cast(pl.Datetime))
    write_table(Table.MARKET_ODDS, market, settings=settings)
    edges = _write_edges(preds, settings, structlog.get_logger())
    assert "H" in edges["outcome"].to_list()  # 12pp home edge flagged


def test_refresh_market_odds_pulls_local(tmp_path) -> None:
    import structlog

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table
    from polymbappe.simulate.tournament import refresh_market_odds

    settings = Settings(data_dir=tmp_path)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    (settings.raw_data_dir / "odds.csv").write_text(
        "date,home_team,away_team,home_odds,draw_odds,away_odds\n"
        "2026-06-14,Spain,Brazil,2.0,3.3,3.7\n"
    )
    n = refresh_market_odds(settings, structlog.get_logger())
    assert n == 1
    assert table_exists(Table.MARKET_ODDS, settings)
    assert read_table(Table.MARKET_ODDS, settings).height == 1


def test_refresh_market_odds_no_sources_is_zero(tmp_path) -> None:
    import structlog

    from polymbappe.config import Settings
    from polymbappe.simulate.tournament import refresh_market_odds

    settings = Settings(data_dir=tmp_path)
    assert refresh_market_odds(settings, structlog.get_logger()) == 0


def test_staleness_monitor_levels() -> None:
    assert surprise_increment(0.1, True) == 0.9
    mon = StalenessMonitor(yellow=1.0, red=2.0)
    assert mon.observe(0.9, occurred=False) == "green"  # surprise 0.9
    assert mon.observe(0.95, occurred=False) == "yellow"  # cumulative 1.85
    assert mon.observe(0.95, occurred=False) == "red"  # cumulative 2.8


# -- live context hook (cohesion + manager feed) -------------------------------

from datetime import date, timedelta  # noqa: E402

from polymbappe.config import Settings  # noqa: E402
from polymbappe.context.runtime import SIM_CONTEXT_FEATURES  # noqa: E402
from polymbappe.data.store import write_table  # noqa: E402
from polymbappe.data.tables import Table  # noqa: E402
from polymbappe.simulate.tournament import (  # noqa: E402
    _context_feature_frame,
    _live_fixture_context,
)


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _live_matches() -> pl.DataFrame:
    """Pre-2026 friendly history between A and B (so overperf/elo have data)."""

    rows = []
    for i in range(6):
        rows.append(
            {
                "match_id": f"h{i}", "date": date(2024, 1, 1) + timedelta(days=i * 30),
                "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0,
                "competition": "Friendly", "is_knockout": False,
                "neutral_site": False, "group": None,
            }
        )
    return pl.DataFrame(rows)


def _write_live_tables(settings: Settings) -> None:
    """matches + 2026 squads + 2026 manager identity rows (team A only)."""

    write_table(Table.MATCHES, _live_matches(), settings=settings)
    squads = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "tournament": ["WC2026"] * 3,
            "player": ["p1", "p2", "p3"],
            "club": ["City", "City", "Madrid"],
            "age": [27.0, 29.0, 31.0],
        }
    )
    write_table(Table.SQUADS, squads, settings=settings)
    records = pl.DataFrame(
        {
            # Historical pedigree (WC2022) + a 2026 identity row naming the current manager.
            "manager": ["mgrA", "mgrA"],
            "team": ["A", "A"],
            "tournament": ["WC2022", "WC2026"],
            "stage_reached": ["FINAL", "GROUP"],
            "knockout_matches": [6, 0],
            "knockout_wins": [5, 0],
            "tournament_order": [1, 2],
        }
    )
    write_table(Table.MANAGER_RECORDS, records, settings=settings)


def test_live_context_frame_column_identical(tmp_path) -> None:
    """The live per-pair frame columns equal SIM_CONTEXT_FEATURES (col-identical to fit)."""

    settings = Settings(data_dir=tmp_path)
    _write_live_tables(settings)
    ctx = _live_fixture_context(settings, elo={"A": 1600.0, "B": 1500.0}, logger=_NullLogger())
    _pairs, frame = _context_feature_frame(["A", "B"], ctx)
    assert frame.columns == list(SIM_CONTEXT_FEATURES)

    # Cohesion/manager populated for A (2026 snapshot + pre-2026 FINAL pedigree).
    a_home = frame.row(0, named=True)  # first pair is (A, B)
    assert a_home["home_club_cluster_index"] == 1.0  # City pair
    assert a_home["home_median_age"] == 29.0
    assert a_home["home_knockout_win_rate"] > 0.0
    assert a_home["home_deepest_run_weighted"] > 0.0
    # B is absent from both tables -> away cohesion/manager 0-filled.
    assert a_home["away_club_cluster_index"] == 0.0
    assert a_home["away_knockout_win_rate"] == 0.0


def test_live_context_frame_no_tables(tmp_path) -> None:
    """Without squads/manager tables: same columns, new ones 0-filled."""

    settings = Settings(data_dir=tmp_path)
    write_table(Table.MATCHES, _live_matches(), settings=settings)
    ctx = _live_fixture_context(settings, elo={"A": 1600.0, "B": 1500.0}, logger=_NullLogger())
    _pairs, frame = _context_feature_frame(["A", "B"], ctx)
    assert frame.columns == list(SIM_CONTEXT_FEATURES)
    row = frame.row(0, named=True)
    for col in SIM_CONTEXT_FEATURES:
        if col not in ("home_xg_overperf", "away_xg_overperf", "draw_pressure"):
            assert row[col] == 0.0


def test_live_context_frame_matches_fit_frame_columns(tmp_path) -> None:
    """Parity: live frame and the fit-path builder produce the identical column set."""

    from polymbappe.context.runtime import build_tournament_context_features
    from polymbappe.eval.backtest import Tournament

    settings = Settings(data_dir=tmp_path)
    _write_live_tables(settings)
    ctx = _live_fixture_context(settings, elo={"A": 1600.0, "B": 1500.0}, logger=_NullLogger())
    _pairs, live_frame = _context_feature_frame(["A", "B"], ctx)

    tournaments = (Tournament("WC2026", "FIFA World Cup", date(2026, 6, 11), date(2026, 7, 19)),)
    fit_frame = build_tournament_context_features(_live_matches(), tournaments, settings)
    fit_feature_cols = [c for c in fit_frame.columns if c != "match_id"]
    assert live_frame.columns == fit_feature_cols == list(SIM_CONTEXT_FEATURES)
