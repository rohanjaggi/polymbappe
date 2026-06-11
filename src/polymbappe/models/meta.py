"""Stacked meta-learner over base-model H/D/A probabilities (spec section 3.4).

Combines base-model H/D/A probabilities (Dixon-Coles, Elo, market, GBM) into a single
calibrated H/D/A distribution. With few features and a few hundred training samples, a
strongly-regularized parametric calibrator is the right default — but the calibrator
*family* is itself a tunable structural choice (``MetaConfig.learner``):

* ``logistic`` — L2-regularized multinomial logistic regression (the default).
* ``isotonic`` — per-outcome isotonic calibration of the averaged base probability; a
  non-parametric monotone map, robust when the logistic link is mis-specified.
* ``weighted_average`` — a learned convex blend of the base-model probability triples; no
  per-feature weights, just one non-negative weight per base model, so it cannot overfit.

Outcome order is fixed as Home, Draw, Away (index 0/1/2) — the ordering the ranked
probability score expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}
_SUFFIX_TO_IDX = {"home": 0, "draw": 1, "away": 2}
LEARNERS: tuple[str, ...] = ("logistic", "isotonic", "weighted_average")


@dataclass(slots=True)
class MetaConfig:
    """Meta-learner hyperparameters."""

    C: float = 1.0
    max_iter: int = 1000
    learner: str = "logistic"
    """Calibrator family: one of :data:`LEARNERS`."""


def _group_triples(feature_columns: list[str]) -> list[tuple[int, int, int]]:
    """Group ``<prefix>_{home,draw,away}`` columns into ordered H/D/A index triples.

    Returns one ``(home_idx, draw_idx, away_idx)`` tuple of positions into
    ``feature_columns`` per base model. Raises if the columns do not decompose into
    complete H/D/A triples (required by the isotonic / weighted-average learners, which
    operate on whole probability vectors rather than individual columns).
    """

    groups: dict[str, dict[int, int]] = {}
    for pos, col in enumerate(feature_columns):
        prefix, _, suffix = col.rpartition("_")
        if not prefix or suffix not in _SUFFIX_TO_IDX:
            raise ValueError(
                f"Column {col!r} is not a '<model>_{{home,draw,away}}' probability column; "
                "the isotonic and weighted_average learners require complete H/D/A triples."
            )
        groups.setdefault(prefix, {})[_SUFFIX_TO_IDX[suffix]] = pos
    triples: list[tuple[int, int, int]] = []
    for prefix, slots in groups.items():
        if set(slots) != {0, 1, 2}:
            raise ValueError(f"Base model {prefix!r} is missing one of its H/D/A columns.")
        triples.append((slots[0], slots[1], slots[2]))
    return triples


class MetaLearner:
    """Calibrator over base-model probability features (logistic / isotonic / blend)."""

    def __init__(self, feature_columns: list[str], config: MetaConfig | None = None) -> None:
        if not feature_columns:
            raise ValueError("MetaLearner requires at least one feature column.")
        self.feature_columns = list(feature_columns)
        self.config = config or MetaConfig()
        if self.config.learner not in LEARNERS:
            raise ValueError(
                f"Unknown meta-learner {self.config.learner!r}; expected one of {LEARNERS}."
            )
        self._model: LogisticRegression | None = None
        self._isotonic: list[IsotonicRegression] | None = None
        self._blend_weights: np.ndarray | None = None
        self._triples: list[tuple[int, int, int]] = []

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.feature_columns).to_numpy()

    def _stacked_triples(self, x: np.ndarray) -> np.ndarray:
        """Reshape the feature matrix to ``(n, n_models, 3)`` in H/D/A order."""

        cols = [x[:, list(triple)] for triple in self._triples]
        return np.stack(cols, axis=1)

    def fit(self, df: pl.DataFrame, label_column: str = "label") -> MetaLearner:
        """Fit the calibrator on a frame of base-prob features and H/D/A labels."""

        x = self._matrix(df)
        y = np.array([_LABEL_TO_IDX[label] for label in df[label_column].to_list()])
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two outcome classes to fit the meta-learner.")
        if self.config.learner == "logistic":
            self._fit_logistic(x, y)
        elif self.config.learner == "isotonic":
            self._fit_isotonic(x, y)
        else:
            self._fit_weighted_average(x, y)
        return self

    def _fit_logistic(self, x: np.ndarray, y: np.ndarray) -> None:
        # Default penalty is L2 (ridge) across supported sklearn versions; passing it
        # explicitly is deprecated in newer releases, so we rely on the default.
        self._model = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter)
        self._model.fit(x, y)

    def _fit_isotonic(self, x: np.ndarray, y: np.ndarray) -> None:
        self._triples = _group_triples(self.feature_columns)
        mean_prob = self._stacked_triples(x).mean(axis=1)  # (n, 3): avg base prob per outcome
        self._isotonic = []
        for k in range(len(OUTCOMES)):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(mean_prob[:, k], (y == k).astype(float))
            self._isotonic.append(iso)

    def _fit_weighted_average(self, x: np.ndarray, y: np.ndarray) -> None:
        from scipy.optimize import minimize

        self._triples = _group_triples(self.feature_columns)
        triples = self._stacked_triples(x)  # (n, n_models, 3)
        n_models = triples.shape[1]
        one_hot = np.zeros((len(y), len(OUTCOMES)))
        one_hot[np.arange(len(y)), y] = 1.0

        def neg_log_loss(z: np.ndarray) -> float:
            weights = _softmax(z)
            blended = np.tensordot(triples, weights, axes=([1], [0]))  # (n, 3)
            blended = blended / blended.sum(axis=1, keepdims=True)
            return float(-np.mean(np.sum(one_hot * np.log(np.clip(blended, 1e-12, None)), axis=1)))

        if n_models == 1:
            self._blend_weights = np.ones(1)
            return
        res = minimize(neg_log_loss, np.zeros(n_models), method="Nelder-Mead")
        self._blend_weights = _softmax(np.asarray(res.x, dtype=float))

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return an (n, 3) array of calibrated [home, draw, away] probabilities."""

        x = self._matrix(df)
        if self.config.learner == "logistic":
            return self._predict_logistic(x)
        if self.config.learner == "isotonic":
            return self._predict_isotonic(x)
        return self._predict_weighted_average(x)

    def _predict_logistic(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MetaLearner must be fit before predicting.")
        raw = self._model.predict_proba(x)
        # Reindex columns to the canonical H/D/A order regardless of class ordering.
        out = np.zeros((raw.shape[0], len(OUTCOMES)), dtype=float)
        for col, cls in enumerate(self._model.classes_):
            out[:, int(cls)] = raw[:, col]
        return out

    def _predict_isotonic(self, x: np.ndarray) -> np.ndarray:
        if self._isotonic is None:
            raise RuntimeError("MetaLearner must be fit before predicting.")
        mean_prob = self._stacked_triples(x).mean(axis=1)
        cal = np.stack(
            [self._isotonic[k].predict(mean_prob[:, k]) for k in range(len(OUTCOMES))], axis=1
        )
        cal = np.clip(cal, 1e-12, None)
        return np.asarray(cal / cal.sum(axis=1, keepdims=True), dtype=float)

    def _predict_weighted_average(self, x: np.ndarray) -> np.ndarray:
        if self._blend_weights is None:
            raise RuntimeError("MetaLearner must be fit before predicting.")
        blended = np.tensordot(self._stacked_triples(x), self._blend_weights, axes=([1], [0]))
        return np.asarray(blended / blended.sum(axis=1, keepdims=True), dtype=float)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return np.asarray(e / e.sum(), dtype=float)
