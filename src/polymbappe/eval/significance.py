"""Paired significance tests for head-to-head forecast comparison.

Two forecasters predict the *same* fixtures, so their skill is best compared per-match
in pairs — this cancels the easy games everyone gets right and squeezes real signal out
of a small (one-tournament) sample. Accuracy is compared with McNemar's test; per-match
loss series (RPS / log-loss) with the Wilcoxon signed-rank test and a paired bootstrap.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_test(model_correct: np.ndarray, other_correct: np.ndarray) -> dict[str, float]:
    """McNemar's test on paired top-pick correctness (model vs. a competitor).

    Considers only the *discordant* fixtures — those one forecaster got right and the
    other wrong — and tests whether the model wins those disagreements more often than
    chance. ``b`` counts model-right/other-wrong, ``c`` model-wrong/other-right. Uses the
    exact two-sided binomial p-value (appropriate for the small samples here) rather than
    the chi-square approximation.

    Returns ``b``, ``c``, ``n_discordant``, and ``p_value`` (1.0 when there are no
    disagreements). Inputs must be equal-length boolean/0-1 arrays aligned by fixture.
    """

    model_correct = np.asarray(model_correct, dtype=bool)
    other_correct = np.asarray(other_correct, dtype=bool)
    if model_correct.shape != other_correct.shape:
        raise ValueError("model_correct and other_correct must be the same length.")

    b = int(np.sum(model_correct & ~other_correct))
    c = int(np.sum(~model_correct & other_correct))
    n = b + c
    if n == 0:
        return {"b": 0.0, "c": 0.0, "n_discordant": 0.0, "p_value": 1.0}
    p_value = float(stats.binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    return {"b": float(b), "c": float(c), "n_discordant": float(n), "p_value": p_value}


def wilcoxon_loss_diff(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test on per-match loss differences ``loss_a - loss_b``.

    Non-parametric (no normality assumption), operating on the paired per-match loss
    series (e.g. RPS or log-loss). A significant result with ``median_diff < 0`` means
    forecaster A has the lower loss (is better). Returns ``nan`` statistics when every
    difference is zero (test undefined) or the series is empty.
    """

    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss_a and loss_b must be the same length.")

    diff = loss_a - loss_b
    if diff.size == 0 or np.allclose(diff, 0.0):
        return {"statistic": float("nan"), "p_value": float("nan"),
                "median_diff": 0.0, "mean_diff": 0.0}
    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
    }


def paired_bootstrap_loss_diff(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20260611,
) -> dict[str, float]:
    """Paired bootstrap CI for the mean per-match loss gap ``loss_a - loss_b``.

    Resamples fixtures with replacement ``n_boot`` times, recomputing the mean loss
    difference each time, and reads a percentile confidence interval off the resulting
    distribution. Robust and assumption-light. A CI lying entirely below 0 means A beats
    B at the chosen level. Returns ``nan`` bounds for an empty input.
    """

    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss_a and loss_b must be the same length.")

    diff = loss_a - loss_b
    n = diff.size
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_two_sided": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    # Two-sided bootstrap p-value: twice the smaller tail mass around 0.
    frac_below = float(np.mean(boot_means < 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(frac_below, 1.0 - frac_below)))
    return {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
            "p_two_sided": p_two_sided}
