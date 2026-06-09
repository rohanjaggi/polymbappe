"""Tests for the LightGBM stacked base model."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from polymbappe.models.gbm import GBMConfig, GBMStackedModel

_FEATURES = ["dc_home", "dc_draw", "dc_away", "elo_diff"]


def _frame(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(5)
    elo_diff = rng.normal(0, 200, n)
    p_home = 1.0 / (1.0 + np.exp(-elo_diff / 150.0))
    rows = []
    for i in range(n):
        ph = float(np.clip(p_home[i], 0.05, 0.95))
        draw = 0.26
        home = ph * (1 - draw)
        away = (1 - ph) * (1 - draw)
        u = rng.random()
        label = "H" if u < home else ("D" if u < home + draw else "A")
        rows.append(
            {
                "dc_home": home,
                "dc_draw": draw,
                "dc_away": away,
                "elo_diff": float(elo_diff[i]),
                "label": label,
            }
        )
    return pl.DataFrame(rows)


def test_fit_predict_simplex() -> None:
    df = _frame()
    model = GBMStackedModel(_FEATURES, GBMConfig(n_estimators=50)).fit(df)
    proba = model.predict_proba(df)
    assert proba.shape == (df.height, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)


def test_oof_predict_is_simplex_and_aligned() -> None:
    df = _frame()
    model = GBMStackedModel(_FEATURES, GBMConfig(n_estimators=40, n_splits=4))
    oof = model.oof_predict(df)
    assert oof.shape == (df.height, 3)
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-6)
    # Strong positive elo_diff rows should lean home on average.
    strong = df.with_row_index().filter(pl.col("elo_diff") > 250)["index"].to_numpy()
    if len(strong) > 3:
        assert oof[strong, 0].mean() > oof[strong, 2].mean()
