"""Contextual intelligence layer (spec section 2.2, 3.5).

Feature builders for signals the core quantitative models structurally miss — pressing
intensity, squad cohesion, manager pedigree, fatigue, draw pressure, and sentiment — plus
the LightGBM residual adjuster that applies them on top of the calibrated base
predictions under a hard ±3pp influence cap.

Each feature group is independently toggleable so the autotuner can measure its marginal
RPS contribution and disable any group that fails its kill criterion (spec 7.5).
"""

from polymbappe.context.adjuster import (
    ContextualAdjuster,
    ContextualAdjusterConfig,
    apply_adjustment,
)

__all__ = [
    "ContextualAdjuster",
    "ContextualAdjusterConfig",
    "apply_adjustment",
]
