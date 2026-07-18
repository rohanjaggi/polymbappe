"""Adaptive online contextual weighting for live WC2026 monitoring.

During the tournament, this module tests whether each Tier-2 contextual feature group
has a statistically significant relationship with live WC2026 match outcomes. Groups that
pass the signal gate (p < 0.05 AND RPS improvement > 0.003) earn a non-zero weight that
gets baked into the simulation's context hook.

Protocol:
- Minimum 32 completed matches before any test runs (end of matchday 2).
- For each group: OLS regression of home-win residual on the group's collapsed scalar
  signal. The slope (coefficient) becomes the weight when both gate conditions are met.
- Weights are zero by default; they only earn non-zero values from live evidence.
- Running ``contextual-monitor --apply`` updates data/outputs/contextual_wc2026_weights.json
  and the next simulate call will pick up the new weights automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from polymbappe.context.adjuster import apply_adjustment
from polymbappe.context.runtime import FEATURE_GROUPS
from polymbappe.eval.metrics import ranked_probability_score
from polymbappe.models.meta import OUTCOMES

#: Minimum completed WC2026 matches before any signal test runs (end of matchday 2).
MIN_MATCHES = 32
#: Signal gate: p-value threshold for OLS slope on home-win residual.
SIGNAL_GATE_P = 0.05
#: Signal gate: minimum RPS improvement from applying the linear correction.
SIGNAL_GATE_RPS_DELTA = 0.003

#: Start of the 2026 World Cup — used to partition history from live.
WC2026_START = date(2026, 6, 11)

WEIGHT_FILE = "contextual_wc2026_weights.json"
ATTRIBUTION_FILE = "contextual_attribution.parquet"

_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}


@dataclass
class SignalTestResult:
    """Outcome of one feature group's signal test."""

    feature_group: str
    p_value: float
    rps_delta: float
    weight: float
    active: bool
    n_matches: int


@dataclass
class AdaptiveWeightState:
    """Current adaptive weights and metadata, persisted to JSON."""

    weights: dict[str, float] = field(default_factory=dict)
    n_matches: int = 0
    last_updated: str = ""

    def is_active(self) -> bool:
        return any(w != 0.0 for w in self.weights.values())

    @classmethod
    def zero(cls, n_matches: int = 0) -> AdaptiveWeightState:
        return cls(weights={g: 0.0 for g in FEATURE_GROUPS}, n_matches=n_matches)


def load_adaptive_weights(settings: Any | None = None) -> AdaptiveWeightState:
    """Load adaptive weights from disk; returns zero-weight state when absent."""

    from polymbappe.config import Settings

    _s = settings if isinstance(settings, Settings) else Settings()
    path = Path(_s.outputs_data_dir) / WEIGHT_FILE
    if not path.exists():
        return AdaptiveWeightState()
    with path.open() as fh:
        data = json.load(fh)
    return AdaptiveWeightState(
        weights=data.get("weights", {}),
        n_matches=data.get("n_matches", 0),
        last_updated=data.get("last_updated", ""),
    )


def save_adaptive_weights(state: AdaptiveWeightState, settings: Any | None = None) -> Path:
    """Persist adaptive weights to data/outputs/contextual_wc2026_weights.json."""

    from polymbappe.config import Settings

    _s = settings if isinstance(settings, Settings) else Settings()
    path = Path(_s.outputs_data_dir) / WEIGHT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(
            {
                "weights": state.weights,
                "n_matches": state.n_matches,
                "last_updated": state.last_updated,
            },
            fh,
            indent=2,
        )
    return path


def _group_scalar_signal(context_df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Collapse a feature group to one scalar signal per match.

    For home/away column pairs: signal = mean(home_cols) - mean(away_cols) (positive
    means the home team has the advantage). For match-level scalars (e.g. draw_pressure):
    use the value directly. Returns zeros when no columns are present.
    """

    present = [c for c in cols if c in context_df.columns]
    if not present:
        return np.zeros(context_df.height)

    home_cols = [c for c in present if c.startswith("home_")]
    away_cols = [c for c in present if c.startswith("away_")]
    other_cols = [c for c in present if not c.startswith("home_") and not c.startswith("away_")]

    parts: list[np.ndarray] = []
    if home_cols and away_cols:
        home_mat = context_df.select(home_cols).fill_null(0.0).to_numpy()
        away_mat = context_df.select(away_cols).fill_null(0.0).to_numpy()
        parts.append(home_mat.mean(axis=1) - away_mat.mean(axis=1))
    elif home_cols:
        parts.append(context_df.select(home_cols).fill_null(0.0).to_numpy().mean(axis=1))
    elif away_cols:
        parts.append(-context_df.select(away_cols).fill_null(0.0).to_numpy().mean(axis=1))
    if other_cols:
        parts.append(context_df.select(other_cols).fill_null(0.0).to_numpy().mean(axis=1))

    return np.mean(np.stack(parts, axis=0), axis=0)


def run_signal_test(
    labels: list[str],
    base_predictions: np.ndarray,
    context_df: pl.DataFrame,
    feature_group: str,
    group_cols: list[str],
) -> SignalTestResult:
    """Test whether one feature group explains residuals beyond the base model.

    Method: OLS regression of the home-win residual on the group's collapsed scalar signal.
    The gate requires both:
    - p-value of the slope < SIGNAL_GATE_P (statistical significance)
    - RPS improvement from applying the linear correction > SIGNAL_GATE_RPS_DELTA (practical)

    The stored weight is the OLS slope: how many probability points to shift per unit of
    signal. Applied at simulation time via a linear H/D/A adjustment capped at ±3pp.
    """

    from scipy import stats  # optional; already in requirements via autotune

    n = len(labels)
    idx = np.array([_LABEL_TO_IDX[lab] for lab in labels])
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), idx] = 1.0
    residuals = one_hot - base_predictions

    signal = _group_scalar_signal(context_df, group_cols)
    # Require at least 10 matches with a non-zero signal to avoid spurious results.
    if np.count_nonzero(signal) < 10:
        return SignalTestResult(
            feature_group=feature_group, p_value=1.0, rps_delta=0.0,
            weight=0.0, active=False, n_matches=n,
        )

    fit = stats.linregress(signal, residuals[:, 0])
    slope = float(fit.slope)  # type: ignore[attr-defined]
    p_value = float(fit.pvalue)  # type: ignore[attr-defined]

    # Estimate RPS improvement with the linear correction applied.
    base_rps = float(ranked_probability_score(idx, base_predictions))
    raw_adj = np.column_stack([slope * signal, np.zeros(n), -slope * signal])
    adjusted = apply_adjustment(base_predictions, raw_adj)
    adj_rps = float(ranked_probability_score(idx, adjusted))
    rps_delta = base_rps - adj_rps  # positive = improvement

    active = (p_value < SIGNAL_GATE_P) and (rps_delta > SIGNAL_GATE_RPS_DELTA)
    weight = slope if active else 0.0

    return SignalTestResult(
        feature_group=feature_group,
        p_value=p_value,
        rps_delta=rps_delta,
        weight=weight,
        active=active,
        n_matches=n,
    )


def run_all_signal_tests(
    labels: list[str],
    base_predictions: np.ndarray,
    context_df: pl.DataFrame,
) -> list[SignalTestResult]:
    """Run signal tests for every contextual feature group."""

    return [
        run_signal_test(labels, base_predictions, context_df, group, cols)
        for group, cols in FEATURE_GROUPS.items()
    ]


def append_attribution(
    results: list[SignalTestResult],
    settings: Any | None = None,
) -> None:
    """Append test results to the rolling attribution parquet."""

    from polymbappe.config import Settings

    _s = settings if isinstance(settings, Settings) else Settings()
    path = Path(_s.outputs_data_dir) / ATTRIBUTION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().isoformat()
    new_df = pl.DataFrame(
        [
            {
                "timestamp": now,
                "feature_group": r.feature_group,
                "n_matches": r.n_matches,
                "p_value": r.p_value,
                "rps_delta": r.rps_delta,
                "weight": r.weight,
                "active": r.active,
            }
            for r in results
        ]
    )
    if path.exists():
        combined = pl.concat([pl.read_parquet(path), new_df], how="diagonal_relaxed")
    else:
        combined = new_df
    combined.write_parquet(path)


def load_live_wc2026_matches(
    matches: pl.DataFrame, settings: Any | None = None
) -> pl.DataFrame:
    """Return completed WC2026 matches from the matches table by joining with schedule.

    Uses the SCHEDULE table's match_ids (which are keyed to WC2026 fixtures) to find
    completed matches. Filters to matches on or after WC2026_START that appear in both
    the schedule and the matches table (meaning they have results).
    """

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = settings or Settings()
    if not table_exists(Table.SCHEDULE, settings):
        return pl.DataFrame(schema=matches.schema)

    schedule = read_table(Table.SCHEDULE, settings)
    schedule_ids = set(schedule["match_id"].to_list())

    return matches.filter(
        pl.col("match_id").is_in(schedule_ids)
        & (pl.col("date") >= pl.lit(WC2026_START))
    )


def labels_from_matches(live: pl.DataFrame) -> list[str]:
    """H/D/A outcome labels for completed matches, row-aligned to ``live``.

    ``live`` must already be filtered to rows with non-null goals; the returned list
    matches its row order, so it aligns with base predictions and context features
    computed from the same frame.
    """

    from polymbappe.features.pipeline import result_label

    return [
        result_label(int(h), int(a))
        for h, a in zip(live["home_goals"].to_list(), live["away_goals"].to_list(), strict=True)
    ]


def compute_wc2026_base_predictions(
    live_matches: pl.DataFrame, all_matches: pl.DataFrame, settings: Any | None = None
) -> np.ndarray:
    """Compute ensemble base predictions for live WC2026 matches.

    Uses pre-WC2026 history to fit the base model (leakage-safe), then runs the
    saved ensemble_calibration artifact to produce H/D/A probabilities. Falls back
    to simple DC+Elo average when the artifact is unavailable.
    """

    from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs

    history = all_matches.filter(pl.col("date") < WC2026_START)
    probs_df = compute_tournament_base_probs(
        history, live_matches, tournament="WC2026", config=BaseProbConfig()
    )

    try:
        from polymbappe.config import Settings
        from polymbappe.models.train import load_artifact

        settings = settings or Settings()
        calibration = load_artifact("ensemble_calibration", settings)
        return calibration.predict_proba(probs_df)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: equal-weight average of DC and Elo probs.
        dc = probs_df.select(["dc_home", "dc_draw", "dc_away"]).to_numpy()
        elo = probs_df.select(["elo_home", "elo_draw", "elo_away"]).to_numpy()
        avg = (dc + elo) / 2.0
        return avg / avg.sum(axis=1, keepdims=True)
