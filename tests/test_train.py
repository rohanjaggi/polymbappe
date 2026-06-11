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


def test_gbm_stacker_wired_into_ensembles() -> None:
    import pytest

    pytest.importorskip("lightgbm")

    matches = _make_matches()
    # Synthetic market odds keyed by match_id so the calibration GBM can see them.
    odds = matches.select(
        "match_id",
        pl.lit(0.45).alias("home_win_prob"),
        pl.lit(0.27).alias("draw_prob"),
        pl.lit(0.28).alias("away_win_prob"),
    )
    artifacts = train_full_stack(matches, tournaments=_TOURNAMENTS, market_odds=odds)

    # The GBM out-of-fold columns reach the meta-learner in both pipelines...
    for ens in (artifacts.calibration, artifacts.edge):
        assert {"gbm_home", "gbm_draw", "gbm_away"}.issubset(ens.meta_features)
        # ...and the GBM was fed real core features, not just base probabilities.
        assert "elo_diff" in ens.gbm_feature_columns

    # The edge GBM stays market-blind; the calibration GBM may use market columns.
    assert not any("mkt" in c for c in artifacts.edge._gbm_columns())
    assert any("mkt" in c for c in artifacts.calibration._gbm_columns())


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


def _write_context_tables(settings, tournaments=_TOURNAMENTS) -> None:
    """Squads + manager records for every tournament in ``tournaments`` (all teams).

    Mirrors the data contract the assembly path consumes (Phase C): per-player ``squads``
    rows (``team/tournament/player/club/age``) and per-team ``manager_records``
    (``manager/team/tournament/stage_reached/knockout_matches/knockout_wins/
    tournament_order``). Populated densely so cohesion has data for every tournament and
    manager has pre-cutoff pedigree for all but the earliest one.
    """

    from polymbappe.data.store import write_table
    from polymbappe.data.tables import Table

    squad_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    for order, t in enumerate(tournaments, start=1):
        for team in TEAMS:
            # Two club-mates (a cohesion pair) + one outlier -> club_cluster_index == 1.
            for p, club in enumerate(("City", "City", "Madrid")):
                squad_rows.append(
                    {
                        "team": team, "tournament": t.name, "player": f"{team}p{p}",
                        "club": club, "age": 27.0 + p,
                    }
                )
            record_rows.append(
                {
                    "manager": f"mgr{team}", "team": team, "tournament": t.name,
                    "stage_reached": "FINAL" if team == "A" else "QF",
                    "knockout_matches": 6, "knockout_wins": 5 if team == "A" else 2,
                    "tournament_order": order,
                }
            )
    write_table(Table.SQUADS, pl.DataFrame(squad_rows), settings=settings)
    write_table(Table.MANAGER_RECORDS, pl.DataFrame(record_rows), settings=settings)


def test_train_to_adjuster_integration_with_context_tables(tmp_path) -> None:
    """End-to-end fit smoke: synthetic matches + squads/manager tables -> fitted adjuster.

    Drives the real contextual FIT seam (``build_tournament_context_features`` ->
    ``_fit_contextual_adjuster``) over a tiny-but-LightGBM-fittable dataset and asserts:
    (a) it does not crash and produces an adjuster; (b) the fitted adjuster's
    ``active_features`` include the cohesion and manager columns (toggles on by default);
    (c) the assembled context frame carries all 11 ``SIM_CONTEXT_FEATURES`` *and* the
    cohesion/manager columns carry real non-zero data (coverage is sufficient, not 0-fill).
    """

    import importlib.util

    from polymbappe.config import Settings
    from polymbappe.context.runtime import (
        FEATURE_GROUPS,
        SIM_CONTEXT_FEATURES,
        build_tournament_context_features,
    )
    from polymbappe.models.ensemble import BASE_GROUPS, EnsembleConfig, build_dual_pipelines
    from polymbappe.models.train import (
        _attach_core_features,
        _fit_contextual_adjuster,
        assemble_stacked_frame,
    )

    settings = Settings(data_dir=tmp_path)
    _write_context_tables(settings)
    matches = _make_matches()

    # (c) The assembled fit frame carries all 11 columns with real cohesion/manager data.
    context = build_tournament_context_features(
        matches, _TOURNAMENTS, settings, include_tournament=True
    )
    assert all(c in context.columns for c in SIM_CONTEXT_FEATURES)
    cohesion_tournaments = (
        context.filter(pl.col("home_club_cluster_index") != 0.0)["tournament"].n_unique()
    )
    manager_tournaments = (
        context.filter(pl.col("home_knockout_win_rate") != 0.0)["tournament"].n_unique()
    )
    assert cohesion_tournaments == 3  # cohesion has data for every tournament
    assert manager_tournaments >= 2  # earliest tournament has no pre-cutoff pedigree

    # Build the stacked frame + calibration ensemble the adjuster is fit against.
    frame = assemble_stacked_frame(matches, _TOURNAMENTS)
    frame, core_cols = _attach_core_features(frame, matches, None)
    gbm_available = importlib.util.find_spec("lightgbm") is not None
    gbm_cols = [c for c in BASE_GROUPS["dc"] + BASE_GROUPS["elo"] if c in frame.columns] + core_cols
    cfg = EnsembleConfig(base_groups=("dc", "elo"), use_gbm=gbm_available and bool(gbm_cols))
    calibration, _edge = build_dual_pipelines(cfg, gbm_feature_columns=gbm_cols)
    calibration.fit(frame)

    # (a) The real fit seam produces an adjuster without crashing.
    adjuster = _fit_contextual_adjuster(frame, calibration, context.drop("tournament"))
    assert adjuster is not None

    # (b) Cohesion + manager columns are in the fitted feature set (toggles default on).
    for group in ("cohesion", "manager"):
        for col in FEATURE_GROUPS[group]:
            assert col in adjuster.active_features


def test_contextual_adjuster_fit_and_persisted(tmp_path) -> None:
    import numpy as np

    from polymbappe.config import Settings

    artifacts = train_full_stack(_make_matches(), tournaments=_TOURNAMENTS)
    # The adjuster is fit on the per-fixture contextual features (36 rows >= 20 threshold).
    assert artifacts.adjuster is not None

    settings = Settings(data_dir=tmp_path)
    persist_artifacts(artifacts, settings)
    adj = load_artifact("contextual_adjuster", settings)
    # It applies as a capped, simplex-preserving adjustment.
    base = np.tile([0.4, 0.3, 0.3], (artifacts.stacked_frame.height, 1))
    # Build the same contextual feature columns the adjuster expects.
    from polymbappe.context.runtime import SIM_CONTEXT_FEATURES

    feat = artifacts.stacked_frame.with_columns(
        [__import__("polars").lit(0.0).alias(c) for c in SIM_CONTEXT_FEATURES]
    )
    out = adj.adjust(feat, base)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.abs(out - base) <= 0.03 + 1e-9)
