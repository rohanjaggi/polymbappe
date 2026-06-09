"""LightGBM stacked base model (spec section 3.3).

A 3-class (home / draw / away) gradient-boosted classifier over the Tier 1-3 core
features *plus* the other base models' H/D/A outputs (Dixon-Coles probabilities,
Bayesian posterior means). It captures non-linear feature interactions the Poisson
framework misses. Hyperparameters are deliberately conservative to limit overfitting on
sparse international data.

The model produces leakage-safe out-of-fold predictions via :meth:`oof_predict` so the
meta-learner (spec 3.4) can stack on it without seeing in-sample fits.

Outcome order is fixed Home, Draw, Away (index 0/1/2), matching the meta-learner and the
ranked probability score. LightGBM is an optional (``modeling``) dependency, imported
lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}


@dataclass(slots=True)
class GBMConfig:
    """LightGBM stacked-model hyperparameters (spec 3.3 defaults)."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 20
    random_state: int = 20260611
    n_splits: int = 5


class GBMStackedModel:
    """LightGBM 3-class classifier over core features + base-model probabilities."""

    def __init__(self, feature_columns: list[str], config: GBMConfig | None = None) -> None:
        if not feature_columns:
            raise ValueError("GBMStackedModel requires at least one feature column.")
        self.feature_columns = list(feature_columns)
        self.config = config or GBMConfig()
        self._model: object | None = None

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.feature_columns).to_numpy()

    @staticmethod
    def _labels(df: pl.DataFrame, label_column: str) -> np.ndarray:
        return np.array([_LABEL_TO_IDX[label] for label in df[label_column].to_list()])

    def _new_estimator(self) -> object:
        from lightgbm import LGBMClassifier  # lazy: optional dependency

        cfg = self.config
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            min_child_samples=cfg.min_child_samples,
            random_state=cfg.random_state,
            verbose=-1,
        )

    @staticmethod
    def _reindex(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Reorder a classifier's probability columns to canonical H/D/A order."""

        out = np.zeros((raw.shape[0], len(OUTCOMES)), dtype=float)
        for col, cls in enumerate(classes):
            out[:, int(cls)] = raw[:, col]
        return out

    def fit(self, df: pl.DataFrame, label_column: str = "label") -> GBMStackedModel:
        """Fit on the full frame of features and H/D/A labels."""

        x = self._matrix(df)
        y = self._labels(df, label_column)
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two outcome classes to fit the GBM.")
        model = self._new_estimator()
        model.fit(x, y)  # type: ignore[attr-defined]
        self._model = model
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` array of [home, draw, away] probabilities."""

        if self._model is None:
            raise RuntimeError("GBMStackedModel must be fit before predicting.")
        raw = self._model.predict_proba(self._matrix(df))  # type: ignore[attr-defined]
        return self._reindex(raw, self._model.classes_)  # type: ignore[attr-defined]

    def oof_predict(
        self, df: pl.DataFrame, label_column: str = "label", n_splits: int | None = None
    ) -> np.ndarray:
        """Leakage-safe out-of-fold H/D/A probabilities for stacking.

        Each row is predicted by a model trained only on the other folds. Stratified
        K-fold preserves the H/D/A class balance. Returns an ``(n, 3)`` array aligned to
        ``df`` row order.
        """

        from sklearn.model_selection import StratifiedKFold

        x = self._matrix(df)
        y = self._labels(df, label_column)
        splits = n_splits or self.config.n_splits
        splits = max(2, min(splits, int(np.min(np.bincount(y)))))
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.config.random_state)
        out = np.zeros((len(y), len(OUTCOMES)), dtype=float)
        for train_idx, test_idx in skf.split(x, y):
            model = self._new_estimator()
            model.fit(x[train_idx], y[train_idx])  # type: ignore[attr-defined]
            raw = model.predict_proba(x[test_idx])  # type: ignore[attr-defined]
            out[test_idx] = self._reindex(raw, model.classes_)  # type: ignore[attr-defined]
        return out
