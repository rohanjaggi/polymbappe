"""R32 probable matchup predictions from simulation matchup frequency (spec 4.3)."""

from __future__ import annotations

import numpy as np
import polars as pl

from polymbappe.simulate.match import hda_marginals


def compute_knockout_predictions(
    r32_matchup_counts: dict[tuple[str, str], int],
    n_sims: int,
    model: object,
) -> pl.DataFrame:
    """Ranked R32 probable matchups with model H/D/A, weighted by simulation frequency.

    Sorts all observed R32 pairings by how often they occurred across ``n_sims`` runs,
    then attaches point-estimate H/D/A from the strength model at neutral venue.
    The bracket is partially random (only top-4 ranked group winners have protected slots),
    so matchup frequency is the ground truth for pre-tournament knockout predictions.

    Args:
        r32_matchup_counts: {(home_team, away_team): count} from SimulationResult.
        n_sims: total simulation count for normalising to probabilities.
        model: StrengthModel with .score_matrix() and .max_goals.

    Returns:
        DataFrame sorted by matchup_prob descending with columns:
        rank, home_team, away_team, matchup_prob,
        model_home, model_draw, model_away, exp_home_goals, exp_away_goals.
    """
    grid = np.arange(model.max_goals + 1)  # type: ignore[attr-defined]
    rows = []
    for (home, away), count in sorted(r32_matchup_counts.items(), key=lambda x: -x[1]):
        matrix = model.score_matrix(home, away, neutral=True)  # type: ignore[attr-defined]
        h, d, a = hda_marginals(matrix)
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "matchup_prob": count / n_sims,
                "model_home": h,
                "model_draw": d,
                "model_away": a,
                "exp_home_goals": float((matrix.sum(axis=1) * grid).sum()),
                "exp_away_goals": float((matrix.sum(axis=0) * grid).sum()),
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "rank": pl.Int32,
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
                "matchup_prob": pl.Float64,
                "model_home": pl.Float64,
                "model_draw": pl.Float64,
                "model_away": pl.Float64,
                "exp_home_goals": pl.Float64,
                "exp_away_goals": pl.Float64,
            }
        )

    return (
        pl.DataFrame(rows)
        .with_columns(pl.Series("rank", list(range(1, len(rows) + 1)), dtype=pl.Int32))
        .select(
            "rank",
            "home_team",
            "away_team",
            "matchup_prob",
            "model_home",
            "model_draw",
            "model_away",
            "exp_home_goals",
            "exp_away_goals",
        )
    )
