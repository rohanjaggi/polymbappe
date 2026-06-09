"""Tests for the full-stack training orchestration."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from polymbappe.eval.backtest import Tournament
from polymbappe.models.train import (
    assemble_stacked_frame,
    load_artifact,
    persist_artifacts,
    train_full_stack,
)

TEAMS = ["A", "B", "C", "D"]
_ATTACK = {"A": 1.7, "B": 1.3, "C": 1.0, "D": 0.7}

_TOURNAMENTS = (
    Tournament("WC2016", "FIFA World Cup", date(2016, 6, 1), date(2016, 7, 31)),
    Tournament("EU2018", "UEFA Euro", date(2018, 6, 1), date(2018, 7, 31)),
    Tournament("CA2020", "Copa América", date(2020, 6, 1), date(2020, 7, 31)),
)


def _make_matches() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    idx = 0

    def add(d: date, home: str, away: str, comp: str, neutral: bool) -> None:
        nonlocal idx
        rows.append(
            {
                "match_id": f"m{idx}", "date": d, "home_team": home, "away_team": away,
                "home_goals": int(rng.poisson(_ATTACK[home] + (0 if neutral else 0.25))),
                "away_goals": int(rng.poisson(_ATTACK[away])),
                "competition": comp, "is_knockout": False, "neutral_site": neutral,
                "group": None,
            }
        )
        idx += 1

    day = date(2008, 1, 1)
    for _ in range(20):
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(day, h, a, "Friendly", False)
                    day += timedelta(days=7)
    for comp, year in (("FIFA World Cup", 2016), ("UEFA Euro", 2018), ("Copa América", 2020)):
        td = date(year, 6, 10)
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    add(td, h, a, comp, True)
                    td += timedelta(days=1)
    return pl.DataFrame(rows)


def test_assemble_stacked_frame() -> None:
    frame = assemble_stacked_frame(_make_matches(), _TOURNAMENTS)
    assert frame.height == 36  # 12 fixtures x 3 tournaments
    for col in ("dc_home", "dc_draw", "dc_away", "elo_home", "label"):
        assert col in frame.columns


def test_train_full_stack_and_persist(tmp_path) -> None:
    from polymbappe.config import Settings

    artifacts = train_full_stack(_make_matches(), tournaments=_TOURNAMENTS)
    proba = artifacts.calibration.predict_proba(artifacts.stacked_frame)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # Edge pipeline is market-blind (no market features anyway here).
    assert not any("mkt" in c for c in artifacts.edge.meta_features)

    settings = Settings(data_dir=tmp_path)
    persist_artifacts(artifacts, settings)
    loaded = load_artifact("dixon_coles", settings)
    assert loaded.predict_match("A", "D")["home_win"] > 0.0
