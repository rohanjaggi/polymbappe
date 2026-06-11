"""Tests for the Monte Carlo tournament engine."""

from __future__ import annotations

import numpy as np

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


def test_staleness_monitor_levels() -> None:
    assert surprise_increment(0.1, True) == 0.9
    mon = StalenessMonitor(yellow=1.0, red=2.0)
    assert mon.observe(0.9, occurred=False) == "green"  # surprise 0.9
    assert mon.observe(0.95, occurred=False) == "yellow"  # cumulative 1.85
    assert mon.observe(0.95, occurred=False) == "red"  # cumulative 2.8
