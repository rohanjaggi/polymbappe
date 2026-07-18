"""WC2026 adaptive context hook (Phase 2).

Translates the live-earned adaptive weights (loaded from
data/outputs/contextual_wc2026_weights.json) into a per-match context hook that the
simulator can drop in as a replacement for the historically-trained ContextualAdjuster.

The hook applies a linear weighted adjustment per feature group:
  signal = mean(home_cols) - mean(away_cols)   (same as _group_scalar_signal)
  Δhome  = Σ weight[group] * signal[group]
  raw_adj = [Δhome, 0, -Δhome]
  final  = apply_adjustment(base_hda, raw_adj, cap=0.03)

Weights start at zero (no effect) and earn non-zero values only after passing the
signal gate in ``contextual-monitor``. When all weights are zero the hook returns the
base probabilities unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from polymbappe.context.adaptive import AdaptiveWeightState, _group_scalar_signal
from polymbappe.context.adjuster import apply_adjustment
from polymbappe.context.runtime import FEATURE_GROUPS, FixtureContext, fixture_feature_row

#: Callable type alias matching ContextHook without the circular import.
_ContextHook = Callable[[str, str, np.ndarray], np.ndarray]


def build_adaptive_hook(
    state: AdaptiveWeightState,
    teams: list[str],
    ctx: FixtureContext,
    team_travel: dict[str, float] | None = None,
) -> _ContextHook | None:
    """Build a precomputed adaptive hook from the current weight state.

    Precomputes contextual feature values and group signals for every ordered (home, away)
    pair in one batch, then returns an O(1)-lookup ContextHook. Returns None when no
    weights are active (all groups inactive → base probs unchanged).

    Args:
        state: Adaptive weight state loaded from the weight file.
        teams: All WC2026 teams in the simulation.
        ctx: FixtureContext bundle (overperf, elo, cohesion, manager) for WC2026.
        team_travel: Optional team → mean travel km mapping from the schedule.
    """

    if not state.is_active():
        return None

    active_groups = {g: w for g, w in state.weights.items() if w != 0.0}
    travel = team_travel or {}

    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = [
        fixture_feature_row(
            h, a, ctx,
            home_travel_km=travel.get(h, 0.0),
            away_travel_km=travel.get(a, 0.0),
        )
        for h, a in pairs
    ]
    frame = pl.DataFrame(rows)

    # Precompute the net adjustment scalar for every pair.
    pair_adjustment: dict[tuple[str, str], float] = {}
    for i, pair in enumerate(pairs):
        net = 0.0
        for group, weight in active_groups.items():
            cols = FEATURE_GROUPS.get(group, [])
            row_frame = frame.slice(i, 1)
            signal = _group_scalar_signal(row_frame, cols)
            net += float(weight) * float(signal[0])
        pair_adjustment[pair] = net

    def hook(home: str, away: str, base_hda: np.ndarray) -> np.ndarray:
        net = pair_adjustment.get((home, away), 0.0)
        if net == 0.0:
            return base_hda
        raw_adj = np.array([[net, 0.0, -net]])
        return apply_adjustment(base_hda.reshape(1, 3), raw_adj)[0]

    return hook
