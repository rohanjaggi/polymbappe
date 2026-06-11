"""Evaluation metrics for probabilistic forecasts."""

from __future__ import annotations

import numpy as np
import polars as pl


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary Brier score."""

    if y_true.shape != y_prob.shape:
        raise ValueError("Shapes must match for Brier score.")
    return float(np.mean((y_prob - y_true) ** 2))


def multiclass_log_loss(y_true_idx: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Multiclass log loss."""

    probs = np.clip(y_prob, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(probs[np.arange(len(y_true_idx)), y_true_idx])))


def ranked_probability_score(y_true_idx: np.ndarray, y_prob: np.ndarray) -> float:
    """Ranked probability score for ordered categorical outcomes.

    Normalized by 1/(K-1) per the standard convention (Epstein 1969, Constantinou 2019).
    For 3 outcomes (H/D/A), divides by 2 so the scale matches literature benchmarks.
    """

    n_classes = y_prob.shape[1]
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    cdf_prob = np.cumsum(y_prob, axis=1)
    cdf_true = np.cumsum(one_hot, axis=1)
    raw = np.mean(np.sum((cdf_prob[:, : n_classes - 1] - cdf_true[:, : n_classes - 1]) ** 2, axis=1))
    return float(raw / (n_classes - 1))


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Return calibration table with mean predicted probability and empirical frequency."""

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bins, right=True) - 1, 0, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        mask = indices == i
        if not np.any(mask):
            rows.append(
                {
                    "bin": i,
                    "bin_lower": float(bins[i]),
                    "bin_upper": float(bins[i + 1]),
                    "mean_pred": float("nan"),
                    "empirical": float("nan"),
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "mean_pred": float(np.mean(y_prob[mask])),
                "empirical": float(np.mean(y_true[mask])),
                "count": int(mask.sum()),
            }
        )
    return pl.DataFrame(rows)
