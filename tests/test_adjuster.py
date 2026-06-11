"""Tests for the contextual residual adjuster and the ±3pp cap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polymbappe.context.adjuster import apply_adjustment

pytest.importorskip("lightgbm")

from polymbappe.context.adjuster import ContextualAdjuster, ContextualAdjusterConfig


def test_apply_adjustment_caps_shift_and_is_simplex() -> None:
    base = np.array([[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]])
    raw = np.array([[0.5, -0.5, 0.0], [-0.4, 0.4, 0.0]])  # huge raw adjustments
    out = apply_adjustment(base, raw, cap=0.03)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out >= 0.0)
    # No outcome shifts by more than the cap (+ tiny renorm slack).
    assert np.all(np.abs(out - base) <= 0.03 + 1e-9)


def _frame(n: int = 200) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(4)
    # Contextual feature that genuinely predicts extra home wins.
    signal = rng.normal(0, 1, n)
    base = np.tile([0.4, 0.3, 0.3], (n, 1))
    labels = []
    for i in range(n):
        p_home = 0.4 + 0.2 * (signal[i] > 0.5)
        u = rng.random()
        labels.append("H" if u < p_home else ("D" if u < p_home + 0.3 else "A"))
    df = pl.DataFrame({"ctx_signal": signal, "label": labels})
    return df, base


def test_adjuster_fits_and_respects_cap() -> None:
    df, base = _frame()
    adj = ContextualAdjuster({"ppda": ["ctx_signal"]}, ContextualAdjusterConfig())
    adj.fit(df, base)
    out = adj.adjust(df, base)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.abs(out - base) <= 0.03 + 1e-9)


def test_disabled_layer_returns_base_unchanged() -> None:
    df, base = _frame(50)
    adj = ContextualAdjuster(
        {"ppda": ["ctx_signal"]},
        ContextualAdjusterConfig(enable_contextual_layer=False),
    )
    adj.fit(df, base)
    out = adj.adjust(df, base)
    assert np.array_equal(out, base)


def test_group_toggle_drops_features() -> None:
    df, base = _frame(50)
    cfg = ContextualAdjusterConfig(toggle_ppda=False, toggle_manager=True)
    adj = ContextualAdjuster(
        {"ppda": ["ctx_signal"], "manager": []}, cfg
    )
    adj.fit(df, base)
    assert "ctx_signal" not in adj.active_features
