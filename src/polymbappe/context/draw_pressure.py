"""Draw-pressure indicators (spec 2.2 Group F).

Draws are the hardest outcome to predict (~25-28% of group matches) and are systematically
under-predicted by Poisson models. These match-pair features specifically target draw
probability:

* **Mutual qualification incentive** — would a draw qualify *both* teams? (final group
  matchday only). Historically lifts draw probability 5-8pp.
* **PPDA similarity** — similar pressing styles draw more (see :mod:`.ppda`).
* **Low-scoring probability** — ``P(total goals <= 1)`` from the Dixon-Coles score matrix;
  low-scoring matches are mechanically more draw-prone.
* **Stage x Elo-gap interaction** — group stage with a small Elo gap draws more; knockout
  resolves via extra time so draws (in 90') are less decisive.

Match-pair features need a known opponent: for the group stage they are pre-computed, and
for knockout matches the simulator computes them dynamically per path (spec 2.4).
"""

from __future__ import annotations

import numpy as np

from polymbappe.context.ppda import ppda_similarity

__all__ = [
    "mutual_qualification_incentive",
    "low_scoring_probability",
    "stage_elo_interaction",
    "ppda_similarity",
    "draw_pressure_features",
]


def mutual_qualification_incentive(
    is_final_matchday: bool,
    draw_qualifies_home: bool,
    draw_qualifies_away: bool,
) -> int:
    """1 when it is the final group matchday and a draw qualifies *both* teams."""

    return int(is_final_matchday and draw_qualifies_home and draw_qualifies_away)


def low_scoring_probability(score_matrix: np.ndarray) -> float:
    """``P(total goals <= 1)`` from a normalized home-vs-away score matrix."""

    total = 0.0
    rows, cols = score_matrix.shape
    for x in range(rows):
        for y in range(cols):
            if x + y <= 1:
                total += float(score_matrix[x, y])
    return total


def stage_elo_interaction(
    is_knockout: bool, elo_gap: float, small_gap: float = 100.0
) -> float:
    """Signed draw-pressure from stage and Elo gap.

    Positive in the group stage when the Elo gap is small (draw-prone); negative in the
    knockout stage (extra time resolves). Magnitude shrinks as the gap widens.
    """

    closeness = max(0.0, 1.0 - abs(elo_gap) / small_gap) if small_gap > 0 else 0.0
    return (-1.0 if is_knockout else 1.0) * closeness


def draw_pressure_features(
    *,
    is_final_matchday: bool,
    draw_qualifies_home: bool,
    draw_qualifies_away: bool,
    home_ppda: float | None,
    away_ppda: float | None,
    score_matrix: np.ndarray,
    is_knockout: bool,
    elo_gap: float,
) -> dict[str, float]:
    """Assemble all draw-pressure indicators for one fixture into a feature dict."""

    sim = ppda_similarity(home_ppda, away_ppda)
    return {
        "mutual_qual_incentive": float(
            mutual_qualification_incentive(
                is_final_matchday, draw_qualifies_home, draw_qualifies_away
            )
        ),
        "ppda_similarity": float(sim) if sim is not None else 0.0,
        "low_scoring_prob": low_scoring_probability(score_matrix),
        "stage_elo_interaction": stage_elo_interaction(is_knockout, elo_gap),
    }
