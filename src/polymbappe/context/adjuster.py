"""Contextual adjustment layer (spec 3.5).

Sits between the calibrated base predictions and the final probabilities. Trained on the
*residuals* between calibrated base predictions and actual outcomes, it learns only what
the core model systematically misses: "when these contextual features are present, the base
model tends to under/over-predict in this direction."

* **Model:** a small LightGBM regressor per outcome (``num_leaves=15``, ``n_estimators=100``).
* **Target:** the signed 3-class residual ``one_hot(actual) - base_probs``.
* **Output:** an adjustment vector added to the base probabilities, projected back onto the
  simplex.
* **Hard cap (spec 7.7):** no contextual feature may shift any probability by more than
  ±3pp — :func:`apply_adjustment` clamps the per-outcome shift, bounding worst-case damage
  from untested Tier B features.

The whole layer and each feature group are independently toggleable (spec 7.5 kill
criteria): a disabled layer returns the base probabilities unchanged, and a disabled group
is dropped from the feature set so the autotuner can measure its marginal value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polymbappe.models._lgbm import silence_feature_name_warning

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}

#: Maps a feature group to the config toggle attribute that enables it.
_GROUP_TOGGLE = {
    "ppda": "toggle_ppda",
    "cohesion": "toggle_cohesion",
    "manager": "toggle_manager",
    "fatigue": "toggle_fatigue",
    "xg_overperformance": "toggle_xg_overperformance",
    "draw_pressure": "toggle_draw_pressure",
    "sentiment": "toggle_sentiment",
}


@dataclass(slots=True)
class ContextualAdjusterConfig:
    """Contextual-adjuster hyperparameters and toggles."""

    enable_contextual_layer: bool = True
    cap: float = 0.03
    num_leaves: int = 15
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_samples: int = 20
    random_state: int = 20260611
    toggle_ppda: bool = True
    toggle_cohesion: bool = True
    toggle_manager: bool = True
    toggle_fatigue: bool = True
    toggle_xg_overperformance: bool = True
    toggle_draw_pressure: bool = True
    toggle_sentiment: bool = True


def apply_adjustment(base: np.ndarray, raw_adj: np.ndarray, cap: float = 0.03) -> np.ndarray:
    """Apply a capped adjustment vector to base probabilities, returning a simplex.

    Each outcome's net shift from ``base`` is clamped to ``±cap`` (spec 7.7), then the
    result is re-projected onto the probability simplex. Re-centering keeps the adjustment
    zero-sum so the cap survives normalization.
    """

    base = np.asarray(base, dtype=float)
    raw_adj = np.asarray(raw_adj, dtype=float)
    # Re-center to zero-sum (preserves direction) so adding it keeps the row summing to 1.
    adj = raw_adj - raw_adj.mean(axis=1, keepdims=True)
    # Scale each row so its largest per-outcome shift is exactly the cap (spec 7.7).
    # Scaling preserves the zero-sum property, so the cap holds after the add.
    max_abs = np.max(np.abs(adj), axis=1, keepdims=True)
    scale = np.where(max_abs > cap, cap / np.maximum(max_abs, 1e-12), 1.0)
    adj = adj * scale
    out = base + adj
    if np.any(out < 1e-6):
        # Rare when a base probability sits below the cap; clamp and renormalize. The
        # per-outcome shift may then nudge marginally past the cap at that boundary.
        out = np.clip(out, 1e-6, None)
        out = out / out.sum(axis=1, keepdims=True)
    return np.asarray(out, dtype=float)


class ContextualAdjuster:
    """LightGBM residual adjuster over contextual feature groups."""

    def __init__(
        self,
        feature_groups: dict[str, list[str]],
        config: ContextualAdjusterConfig | None = None,
    ) -> None:
        self.feature_groups = {k: list(v) for k, v in feature_groups.items()}
        self.config = config or ContextualAdjusterConfig()
        self._models: list[object] = []
        self.active_features: list[str] = []

    def _enabled_features(self) -> list[str]:
        cols: list[str] = []
        for group, columns in self.feature_groups.items():
            toggle_attr = _GROUP_TOGGLE.get(group)
            enabled = getattr(self.config, toggle_attr) if toggle_attr else True
            if enabled:
                cols.extend(columns)
        return cols

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.active_features).fill_null(0.0).to_numpy()

    def _new_regressor(self) -> object:
        from lightgbm import LGBMRegressor  # lazy: optional dependency

        cfg = self.config
        return LGBMRegressor(
            num_leaves=cfg.num_leaves,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            min_child_samples=cfg.min_child_samples,
            random_state=cfg.random_state,
            verbose=-1,
        )

    def fit(
        self,
        df: pl.DataFrame,
        base_probs: np.ndarray,
        label_column: str = "label",
    ) -> ContextualAdjuster:
        """Fit one regressor per outcome on the signed residual.

        Args:
            df: Frame carrying the contextual feature columns and the ``label`` target.
            base_probs: ``(n, 3)`` calibrated base probabilities for the same rows.
            label_column: H/D/A label column.
        """

        self.active_features = self._enabled_features()
        if not self.config.enable_contextual_layer or not self.active_features:
            self._models = []
            return self

        idx = np.array([_LABEL_TO_IDX[label] for label in df[label_column].to_list()])
        one_hot = np.zeros_like(base_probs)
        one_hot[np.arange(len(idx)), idx] = 1.0
        residual = one_hot - base_probs

        x = self._matrix(df)
        self._models = []
        for k in range(len(OUTCOMES)):
            model = self._new_regressor()
            model.fit(x, residual[:, k])  # type: ignore[attr-defined]
            self._models.append(model)
        return self

    def predict_adjustment(self, df: pl.DataFrame) -> np.ndarray:
        """Predict the raw (un-capped) ``(n, 3)`` residual adjustment."""

        if not self._models:
            return np.zeros((df.height, len(OUTCOMES)))
        x = self._matrix(df)
        with silence_feature_name_warning():
            cols = [np.asarray(m.predict(x), dtype=float) for m in self._models]  # type: ignore[attr-defined]
        return np.stack(cols, axis=1)

    def adjust(self, df: pl.DataFrame, base_probs: np.ndarray) -> np.ndarray:
        """Return the final capped, simplex-projected probabilities.

        When the layer is disabled (or no group is active / unfit), returns ``base_probs``
        unchanged.
        """

        if not self.config.enable_contextual_layer or not self._models:
            return np.asarray(base_probs, dtype=float)
        raw = self.predict_adjustment(df)
        return apply_adjustment(base_probs, raw, self.config.cap)
