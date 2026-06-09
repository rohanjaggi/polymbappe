"""Tests for the Monte Carlo tournament engine."""

from __future__ import annotations

import numpy as np

from polymbappe.simulate.structure import build_structure, placeholder_structure_2026
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


def test_staleness_monitor_levels() -> None:
    assert surprise_increment(0.1, True) == 0.9
    mon = StalenessMonitor(yellow=1.0, red=2.0)
    assert mon.observe(0.9, occurred=False) == "green"  # surprise 0.9
    assert mon.observe(0.95, occurred=False) == "yellow"  # cumulative 1.85
    assert mon.observe(0.95, occurred=False) == "red"  # cumulative 2.8
