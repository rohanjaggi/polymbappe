"""Stacked meta-learner: L2-regularized multinomial logistic regression.

Combines base-model H/D/A probabilities (Dixon-Coles, Elo, market) into a single
calibrated H/D/A distribution. With few features and a few hundred training samples, a
strongly-regularized parametric calibrator is the right default (spec section 3.4).

Outcome order is fixed as Home, Draw, Away (index 0/1/2) — the ordering the ranked
probability score expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}


@dataclass(slots=True)
class MetaConfig:
    """Meta-learner hyperparameters."""

    C: float = 1.0
    max_iter: int = 1000


class MetaLearner:
    """Multinomial logistic calibrator over base-model probability features."""

    def __init__(self, feature_columns: list[str], config: MetaConfig | None = None) -> None:
        if not feature_columns:
            raise ValueError("MetaLearner requires at least one feature column.")
        self.feature_columns = list(feature_columns)
        self.config = config or MetaConfig()
        self._model: LogisticRegression | None = None

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.feature_columns).to_numpy()

    def fit(self, df: pl.DataFrame, label_column: str = "label") -> MetaLearner:
        """Fit the calibrator on a frame of base-prob features and H/D/A labels."""

        x = self._matrix(df)
        y = np.array([_LABEL_TO_IDX[label] for label in df[label_column].to_list()])
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two outcome classes to fit the meta-learner.")
        # Default penalty is L2 (ridge) across supported sklearn versions; passing it
        # explicitly is deprecated in newer releases, so we rely on the default.
        self._model = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter)
        self._model.fit(x, y)
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return an (n, 3) array of calibrated [home, draw, away] probabilities."""

        if self._model is None:
            raise RuntimeError("MetaLearner must be fit before predicting.")
        raw = self._model.predict_proba(self._matrix(df))
        # Reindex columns to the canonical H/D/A order regardless of class ordering.
        out = np.zeros((raw.shape[0], len(OUTCOMES)), dtype=float)
        for col, cls in enumerate(self._model.classes_):
            out[:, int(cls)] = raw[:, col]
        return out
