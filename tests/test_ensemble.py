"""Tests for the dual-pipeline ensemble and credible-interval edge detection."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polymbappe.eval.market import compute_credible_edges

pytest.importorskip("lightgbm")

from polymbappe.models.ensemble import Ensemble, EnsembleConfig, build_dual_pipelines


def _frame(n: int = 150) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    elo_diff = rng.normal(0, 200, n)
    p_home = 1.0 / (1.0 + np.exp(-elo_diff / 150.0))
    rows = []
    for i in range(n):
        ph = float(np.clip(p_home[i], 0.05, 0.95))
        draw = 0.26
        home, away = ph * (1 - draw), (1 - ph) * (1 - draw)
        # market roughly tracks dc but noisier
        mh = float(np.clip(home + rng.normal(0, 0.03), 0.02, 0.96))
        u = rng.random()
        label = "H" if u < home else ("D" if u < home + draw else "A")
        rows.append(
            {
                "dc_home": home, "dc_draw": draw, "dc_away": away,
                "elo_home": home, "elo_draw": draw, "elo_away": away,
                "mkt_home": mh, "mkt_draw": draw, "mkt_away": 1 - mh - draw,
                "elo_diff": float(elo_diff[i]),
                "label": label,
            }
        )
    return pl.DataFrame(rows)


def test_calibration_pipeline_uses_market() -> None:
    df = _frame()
    ens = Ensemble(
        EnsembleConfig(base_groups=("dc", "elo", "mkt"), market_blind=False),
        gbm_feature_columns=["elo_diff", "mkt_home"],
    ).fit(df)
    assert "mkt_home" in ens.meta_features
    proba = ens.predict_proba(df)
    assert proba.shape == (df.height, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_edge_pipeline_is_market_blind() -> None:
    df = _frame()
    _, edge = build_dual_pipelines(
        EnsembleConfig(base_groups=("dc", "elo", "mkt")),
        gbm_feature_columns=["elo_diff", "mkt_home"],
    )
    edge.fit(df)
    # No market group columns and no market GBM column should reach the meta-learner.
    assert not any("mkt" in c for c in edge.meta_features)
    proba = edge.predict_proba(df)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_credible_edges_require_ci_exclusion() -> None:
    model = pl.DataFrame(
        {
            "match_id": ["a", "b"],
            "model_home": [0.62, 0.62],
            "model_draw": [0.20, 0.20],
            "model_away": [0.18, 0.18],
            "ci_home_low": [0.55, 0.45],  # a excludes 0.50, b contains 0.50
            "ci_home_high": [0.69, 0.70],
            "ci_draw_low": [0.15, 0.15],  # contains market draw 0.27
            "ci_draw_high": [0.30, 0.30],
            "ci_away_low": [0.12, 0.12],  # contains market away 0.23
            "ci_away_high": [0.26, 0.26],
        }
    )
    market = pl.DataFrame(
        {
            "match_id": ["a", "b"],
            "home_win_prob": [0.50, 0.50],  # 12pp edge both, but b's CI overlaps
            "draw_prob": [0.27, 0.27],
            "away_win_prob": [0.23, 0.23],
        }
    )
    edges = compute_credible_edges(model, market, threshold=0.05)
    # Only match 'a' (home) qualifies: edge>5pp AND CI excludes market.
    assert edges["match_id"].to_list() == ["a"]
    assert edges["outcome"].to_list() == ["H"]
