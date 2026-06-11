"""Stacked ensemble orchestration and the dual calibration/edge pipelines (spec 3.4, 3.6).

The ensemble stacks base-model H/D/A probabilities (Dixon-Coles, Bayesian, Elo, market)
through the logistic meta-learner, optionally adding a LightGBM base model whose
out-of-fold predictions enter the stack leakage-free, and an optional contextual
adjuster on top.

Two pipelines run from the same base models (spec 3.6):

* **Calibration** (``market_blind=False``) — includes market-implied probabilities; the
  primary pipeline feeding simulation and the dashboard. Optimizes raw RPS.
* **Edge** (``market_blind=True``) — excludes every market input, producing a
  "market-blind" assessment. Edges are flagged where the edge model and the market
  diverge *and* the edge model's uncertainty does not overlap the market (see
  :func:`polymbappe.eval.market.compute_credible_edges`).

The ensemble is frame-based (it consumes a feature frame with base-probability columns
already attached, mirroring :mod:`polymbappe.eval.base_probs`), keeping it composable and
testable without the full data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from polymbappe.models.gbm import GBMConfig, GBMStackedModel
from polymbappe.models.meta import MetaConfig, MetaLearner

#: Canonical base-probability column groups, in H/D/A order, keyed by source.
BASE_GROUPS: dict[str, tuple[str, str, str]] = {
    "dc": ("dc_home", "dc_draw", "dc_away"),
    "bay": ("bay_home", "bay_draw", "bay_away"),
    "elo": ("elo_home", "elo_draw", "elo_away"),
    "mkt": ("mkt_home", "mkt_draw", "mkt_away"),
}
_GBM_GROUP: tuple[str, str, str] = ("gbm_home", "gbm_draw", "gbm_away")

#: Groups that carry market information and must be dropped in the edge pipeline.
_MARKET_GROUPS = frozenset({"mkt"})


@dataclass(slots=True)
class EnsembleConfig:
    """Ensemble wiring and hyperparameters."""

    base_groups: tuple[str, ...] = ("dc", "bay", "elo", "mkt")
    """Which base-probability groups to stack (subset of :data:`BASE_GROUPS`)."""
    use_gbm: bool = True
    """Add a LightGBM base model whose OOF predictions enter the stack."""
    market_blind: bool = False
    """Edge pipeline: drop all market features (group + GBM market columns)."""
    meta: MetaConfig = field(default_factory=MetaConfig)
    gbm: GBMConfig = field(default_factory=GBMConfig)


class Ensemble:
    """Stacked ensemble: base probabilities -> GBM -> logistic meta-learner."""

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        gbm_feature_columns: list[str] | None = None,
    ) -> None:
        self.config = config or EnsembleConfig()
        self.gbm_feature_columns = list(gbm_feature_columns or [])
        self._meta: MetaLearner | None = None
        self._gbm: GBMStackedModel | None = None
        self.meta_features: list[str] = []

    # -- column planning -------------------------------------------------------

    def _active_groups(self) -> list[str]:
        groups = [g for g in self.config.base_groups if g in BASE_GROUPS]
        if self.config.market_blind:
            groups = [g for g in groups if g not in _MARKET_GROUPS]
        return groups

    def _gbm_columns(self) -> list[str]:
        cols = list(self.gbm_feature_columns)
        if self.config.market_blind:
            cols = [c for c in cols if "mkt" not in c and "market" not in c]
        return cols

    def _present_group_columns(self, df: pl.DataFrame) -> list[str]:
        cols: list[str] = []
        for group in self._active_groups():
            triple = BASE_GROUPS[group]
            if all(c in df.columns for c in triple):
                cols.extend(triple)
        return cols

    # -- fit / predict ---------------------------------------------------------

    def fit(self, df: pl.DataFrame, label_column: str = "label") -> Ensemble:
        """Fit the (optional) GBM and the meta-learner.

        The GBM contributes leakage-safe out-of-fold predictions to the meta training
        frame; a final GBM is fit on all rows for prediction time.
        """

        frame = df
        meta_cols = self._present_group_columns(frame)

        gbm_cols = self._gbm_columns()
        if self.config.use_gbm and gbm_cols:
            self._gbm = GBMStackedModel(gbm_cols, self.config.gbm)
            oof = self._gbm.oof_predict(frame, label_column)
            frame = frame.with_columns(
                pl.Series(_GBM_GROUP[0], oof[:, 0]),
                pl.Series(_GBM_GROUP[1], oof[:, 1]),
                pl.Series(_GBM_GROUP[2], oof[:, 2]),
            )
            self._gbm.fit(df, label_column)  # final model on all rows for predict-time
            meta_cols = meta_cols + list(_GBM_GROUP)

        if not meta_cols:
            raise ValueError("Ensemble has no base-probability features to stack.")
        self.meta_features = meta_cols
        self._meta = MetaLearner(meta_cols, self.config.meta).fit(frame, label_column)
        return self

    def _attach_gbm(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._gbm is None:
            return df
        probs = self._gbm.predict_proba(df)
        return df.with_columns(
            pl.Series(_GBM_GROUP[0], probs[:, 0]),
            pl.Series(_GBM_GROUP[1], probs[:, 1]),
            pl.Series(_GBM_GROUP[2], probs[:, 2]),
        )

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return calibrated ``(n, 3)`` [home, draw, away] probabilities."""

        if self._meta is None:
            raise RuntimeError("Ensemble must be fit before predicting.")
        frame = self._attach_gbm(df)
        return self._meta.predict_proba(frame)


def build_dual_pipelines(
    config: EnsembleConfig | None = None,
    gbm_feature_columns: list[str] | None = None,
) -> tuple[Ensemble, Ensemble]:
    """Construct the (calibration, edge) ensemble pair from one configuration.

    Both share architecture; the edge pipeline forces ``market_blind=True`` so it never
    sees market odds (spec 3.6 — genuine edges require a market-blind model).
    """

    base = config or EnsembleConfig()
    calibration = Ensemble(
        EnsembleConfig(
            base_groups=base.base_groups,
            use_gbm=base.use_gbm,
            market_blind=False,
            meta=base.meta,
            gbm=base.gbm,
        ),
        gbm_feature_columns,
    )
    edge = Ensemble(
        EnsembleConfig(
            base_groups=base.base_groups,
            use_gbm=base.use_gbm,
            market_blind=True,
            meta=base.meta,
            gbm=base.gbm,
        ),
        gbm_feature_columns,
    )
    return calibration, edge
