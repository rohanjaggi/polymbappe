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
    sq = (cdf_prob[:, : n_classes - 1] - cdf_true[:, : n_classes - 1]) ** 2
    raw = np.mean(np.sum(sq, axis=1))
    return float(raw / (n_classes - 1))


def multiclass_brier_score(y_true_idx: np.ndarray, y_prob: np.ndarray) -> float:
    """Summed multiclass Brier score (0 best, 2 worst for 3 classes).

    Uses the *summed* convention (sum of squared errors over the K outcomes, averaged
    over matches) so it matches the dashboard scorecard and the classic (1/3,1/3,1/3)
    uniform baseline of ~0.667 for 3-way football. Do not compare against an *averaged*
    Brier (divided by K) — the scales differ.
    """

    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def per_match_rps(y_true_idx: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score (same normalization as :func:`ranked_probability_score`).

    Returns one RPS value per row so callers can run paired significance tests on the
    per-match loss series rather than on a single aggregate.
    """

    n_classes = y_prob.shape[1]
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    cdf_prob = np.cumsum(y_prob, axis=1)
    cdf_true = np.cumsum(one_hot, axis=1)
    per = np.sum((cdf_prob[:, : n_classes - 1] - cdf_true[:, : n_classes - 1]) ** 2, axis=1)
    return per / (n_classes - 1)


def skill_score(score: float, reference: float) -> float:
    """Skill score of a loss ``score`` against a ``reference`` loss: ``1 - score/reference``.

    Positive means the model beats the reference; 0 means parity; negative means worse.
    Returns ``nan`` for a non-positive reference (no meaningful benchmark).
    """

    if not reference > 0:
        return float("nan")
    return 1.0 - (score / reference)


def uniform_reference_scores(y_true_idx: np.ndarray, n_classes: int = 3) -> dict[str, float]:
    """Score a uniform (1/K each) forecast on the *same* realized outcomes.

    Scoring the uniform forecast on the actual outcome sequence (rather than using a
    closed-form constant) gives an honest, outcome-mix-matched baseline for skill scores.
    Returns ``rps`` / ``log_loss`` / ``brier`` under the same conventions as the model
    metrics. Returns zeros for an empty input.
    """

    n = len(y_true_idx)
    if n == 0:
        return {"rps": 0.0, "log_loss": 0.0, "brier": 0.0}
    uniform = np.full((n, n_classes), 1.0 / n_classes)
    return {
        "rps": ranked_probability_score(y_true_idx, uniform),
        "log_loss": multiclass_log_loss(y_true_idx, uniform),
        "brier": multiclass_brier_score(y_true_idx, uniform),
    }


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Expected and maximum calibration error of top-pick ``confidence`` vs ``correct``.

    Bins matches by the probability assigned to the model's favoured outcome into
    ``n_bins`` equal-width buckets over ``[0, 1]``, then measures the gap between mean
    confidence and observed hit rate in each. ECE is the count-weighted mean absolute
    gap; MCE is the worst single-bin gap. Both are 0 for a perfectly calibrated model
    and lower-is-better. Returns zeros for an empty input.
    """

    n = len(confidence)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0}
    correct = correct.astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        gap = abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
        ece += (cnt / n) * gap
        mce = max(mce, gap)
    return {"ece": ece, "mce": mce}


def calibration_slope_intercept(
    prob: np.ndarray, indicator: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> dict[str, float]:
    """Logistic calibration slope and intercept from pooled (probability, outcome) pairs.

    Fits ``logit(P(indicator=1)) = intercept + slope * logit(prob)`` by Newton-Raphson
    (IRLS), the standard recalibration diagnostic. A well-calibrated forecaster has
    ``slope ≈ 1`` and ``intercept ≈ 0``; ``slope < 1`` signals overconfidence and
    ``slope > 1`` underconfidence. Feed the full pooled set of per-class
    (predicted probability, realized 0/1) pairs across H/D/A.

    Returns ``nan`` slope/intercept when the fit is degenerate (empty input, all-equal
    probabilities, or perfectly separable outcomes that fail to converge).
    """

    nan = {"slope": float("nan"), "intercept": float("nan")}
    n = len(prob)
    if n == 0:
        return nan
    eps = 1e-12
    p = np.clip(prob, eps, 1.0 - eps)
    x = np.log(p / (1.0 - p))  # logit of the forecast
    y = indicator.astype(float)
    if np.allclose(x, x[0]) or y.min() == y.max():
        return nan

    beta = np.zeros(2)  # [intercept, slope]
    design = np.column_stack([np.ones(n), x])
    for _ in range(max_iter):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), eps, None)
        grad = design.T @ (y - mu)
        hess = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return nan
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


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
