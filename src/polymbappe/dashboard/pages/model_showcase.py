"""Page 6 — Model Showcase.

Showcases the model's sophistication: backtest results, autotuner journey,
pipeline overview, and per-tournament performance.
"""

from __future__ import annotations

import json

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Model Showcase page."""

    import streamlit as st

    st.header("Model Showcase")

    _render_headline(st, settings)
    st.divider()
    _render_pipeline_overview(st)
    st.divider()
    _render_autotuner(st, settings)
    st.divider()
    _render_backtest(st, settings)


def _render_headline(st: object, settings: Settings) -> None:
    """Headline metrics that establish credibility."""

    leaderboard = data.load_autotune_leaderboard(settings)

    cols = st.columns(3)

    if not leaderboard.is_empty() and "mean_rps" in leaderboard.columns:
        best_rps = float(leaderboard["mean_rps"].min())
        cols[0].metric(
            "Best RPS (11 tournaments)",
            f"{best_rps:.4f}",
            help="Ranked Probability Score across WC 2010-2022, Euro 2016-2024, Copa 2016-2024. Lower is better.",
        )
        cols[1].metric("Experiments Tested", leaderboard.height)
    else:
        cols[0].metric("Backtest Coverage", "11 tournaments")
        cols[1].metric("Experiments", "—")

    cols[2].metric("Simulations per Forecast", "100,000")


def _render_pipeline_overview(st: object) -> None:
    """Static description of the forecasting pipeline."""

    st.subheader("How It Works")

    st.markdown("""
**Data**: 49,000+ international football matches from 1872 to present, including
friendlies, qualifiers, and major tournaments.

**Model Stack**:
- **Dixon-Coles** (bivariate Poisson) with exponential time decay, L2 regularization,
  and altitude/AFC corrections
- **Bayesian Dixon-Coles** (PyMC) with confederation-level hierarchical pooling
- **LightGBM** stacker over base model outputs + engineered features
- **Meta-learner** (calibrated logistic regression) for final H/D/A probabilities

**Features**: Elo ratings, rolling form (5/10-match windows), head-to-head records,
squad market valuations (Transfermarkt), manager knockout pedigree, expected goals (xG),
pressing intensity (PPDA), travel fatigue, draw pressure, and contextual adjustments.

**Simulation**: 100,000 Monte Carlo runs of the full 48-team bracket per forecast cycle,
with correlated team-strength updates and FIFA tiebreaker rules.

**Calibration**: Dual pipelines — one including market odds (for predictions), one excluding
them (for genuine edge detection against betting markets).
""")


def _render_autotuner(st: object, settings: Settings) -> None:
    """Autotuner optimization journey."""

    leaderboard = data.load_autotune_leaderboard(settings)
    if leaderboard.is_empty():
        st.info("No autotuner data available.")
        return

    st.subheader("Hyperparameter Optimization")
    st.caption(
        "Two-phase automated tuning: Phase 1 explores structural changes "
        "(feature inclusion, meta-learner family). Phase 2 optimizes numeric "
        "hyperparameters via Optuna TPE."
    )

    st.plotly_chart(charts.autotuner_chart(leaderboard), use_container_width=True)

    # Top 5 experiments
    top = leaderboard.sort("mean_rps").head(5)
    st.subheader("Top 5 Configurations")
    st.dataframe(
        top.select(["experiment_id", "phase", "mean_rps"]).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )


def _render_backtest(st: object, settings: Settings) -> None:
    """Per-tournament backtest results."""

    leaderboard = data.load_autotune_leaderboard(settings)
    if leaderboard.is_empty():
        return

    best = leaderboard.sort("mean_rps").head(1)
    if best.is_empty():
        return

    per_tournament_str = best.row(0, named=True).get("per_tournament")
    if not per_tournament_str:
        return

    try:
        per_tournament = json.loads(str(per_tournament_str))
    except (json.JSONDecodeError, TypeError):
        return

    if not isinstance(per_tournament, dict) or not per_tournament:
        return

    st.subheader("Leave-One-Tournament-Out Backtest")
    st.caption(
        "RPS for each held-out tournament when the model is trained on all other data. "
        "Lower is better — 0.21 is a common benchmark for 3-way international football prediction."
    )
    st.plotly_chart(charts.backtest_bar(per_tournament), use_container_width=True)
