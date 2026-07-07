"""Page 6 — Model Showcase.

Showcases the forecasting pipeline: live tournament performance, backtest
results, autotuner journey, and pipeline architecture.
"""

from __future__ import annotations

import json

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Model Showcase page."""

    import streamlit as st

    st.header("Model Showcase")

    _render_live_performance(st, settings)
    st.divider()
    _render_backtest(st, settings)
    st.divider()
    _render_pipeline_overview(st)
    st.divider()
    _render_autotuner(st, settings)


def _render_live_performance(st: object, settings: Settings) -> None:
    """How the model is performing on the live 2026 World Cup."""

    st.subheader("Live Tournament Performance")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("No predictions yet.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)
    if finished.is_empty():
        st.info("No finished matches yet.")
        return

    scorecard = data.prediction_scorecard(finished)
    n = int(scorecard["n"])
    accuracy = scorecard["accuracy"]

    # Compute live RPS
    rps_scores = []
    for r in finished.iter_rows(named=True):
        probs = [float(r["model_home"]), float(r["model_draw"]), float(r["model_away"])]
        actual_idx = {"home": 0, "draw": 1, "away": 2}[str(r["actual_outcome"])]
        actual = [0.0, 0.0, 0.0]
        actual[actual_idx] = 1.0
        rps = sum(
            (sum(probs[:k + 1]) - sum(actual[:k + 1])) ** 2 for k in range(3)
        ) / 2
        rps_scores.append(rps)
    live_rps = sum(rps_scores) / len(rps_scores)

    # KO accuracy
    gs = finished.filter(pl.col("group") != "KO") if "group" in finished.columns else finished
    ko = finished.filter(pl.col("group") == "KO") if "group" in finished.columns else pl.DataFrame()
    ko_sc = data.prediction_scorecard(ko) if not ko.is_empty() else None

    # Group points MAE
    predicted = data.predicted_group_points(match_df)
    standings = data.compute_group_standings(match_df, results)
    pts_joined = predicted.join(standings.select(["team", "points"]), on="team", how="inner")
    pts_mae = float(
        (pts_joined["predicted_points"] - pts_joined["points"].cast(pl.Float64)).abs().mean()
    ) if not pts_joined.is_empty() else 0.0

    row1 = st.columns(5)
    row1[0].metric(
        "Match prediction accuracy",
        f"{accuracy:.0%}",
        help=f"Model's top-pick result was correct in {int(accuracy * n)}/{n} matches.",
    )
    row1[1].metric(
        "Ranked Probability Score",
        f"{live_rps:.4f}",
        delta=f"{(1 - live_rps / 0.2222) * 100:.0f}% better than random",
        delta_color="normal",
        help="RPS measures probabilistic calibration, not just the top pick. Lower is better. A random model scores 0.222.",
    )
    row1[2].metric(
        "Brier Score",
        f"{scorecard['brier_score']:.3f}",
        delta=f"{(1 - scorecard['brier_score'] / 0.6667) * 100:.0f}% better than random",
        delta_color="normal",
        help="Mean squared error over H/D/A probabilities. Lower is better. Random = 0.667.",
    )
    if ko_sc:
        row1[3].metric(
            "Knockout accuracy",
            f"{ko_sc['accuracy']:.0%}",
            help=f"{int(ko_sc['accuracy'] * ko_sc['n'])}/{int(ko_sc['n'])} knockout matches correctly predicted.",
        )
    else:
        row1[3].metric("Knockout accuracy", "—")
    row1[4].metric(
        "Group points MAE",
        f"{pts_mae:.1f} pts",
        help="Average error in predicted group-stage points per team. Lower is better.",
    )

    st.caption(
        f"Performance across {n} matches of the 2026 FIFA World Cup. "
        "RPS evaluates the full probability distribution, not just the top pick — "
        f"our {live_rps:.4f} is {(1 - live_rps / 0.2222) * 100:.0f}% better than "
        "a naive equal-probability baseline (0.222)."
    )


def _render_backtest(st: object, settings: Settings) -> None:
    """Per-tournament backtest results — the model's pre-tournament validation."""

    leaderboard = data.load_autotune_leaderboard(settings)
    if leaderboard.is_empty():
        return

    best = leaderboard.sort("mean_rps").head(1)
    if best.is_empty():
        return

    best_row = best.row(0, named=True)
    best_rps = float(best_row["mean_rps"])

    per_tournament_str = best_row.get("per_tournament")
    per_tournament: dict[str, float] = {}
    if per_tournament_str:
        try:
            per_tournament = json.loads(str(per_tournament_str))
        except (json.JSONDecodeError, TypeError):
            pass

    st.subheader("Backtest: 11-Tournament Cross-Validation")

    if per_tournament:
        beaten = sum(1 for rps in per_tournament.values() if rps < 0.21)
        best_tourney = min(per_tournament, key=per_tournament.get)
        best_tourney_rps = per_tournament[best_tourney]

        row1 = st.columns(4)
        row1[0].metric(
            "Mean RPS (backtest)",
            f"{best_rps:.4f}",
            delta=f"{(1 - best_rps / 0.2222) * 100:.0f}% better than random",
            delta_color="normal",
        )
        row1[1].metric("Tournaments tested", len(per_tournament))
        row1[2].metric("Beat 0.21 benchmark", f"{beaten}/{len(per_tournament)}")
        row1[3].metric(
            "Best tournament",
            f"{_tournament_label(best_tourney)}",
            delta=f"RPS {best_tourney_rps:.4f}",
            delta_color="off",
        )

        st.caption(
            "Leave-one-tournament-out cross-validation: the model is trained on 10 tournaments "
            "and evaluated on the held-out one. This tests generalization, not memorization. "
            "The 0.21 benchmark is a typical baseline for 3-way international football prediction."
        )
        st.plotly_chart(charts.backtest_bar(per_tournament), use_container_width=True)
    else:
        st.metric("Mean RPS", f"{best_rps:.4f}")


def _render_pipeline_overview(st: object) -> None:
    """Accurate description of the forecasting pipeline."""

    st.subheader("How It Works")

    st.markdown("""
**Data**: 49,500+ international matches (1872 -- present), covering friendlies,
qualifiers, and major tournaments.

**Core Model**: Dixon-Coles bivariate Poisson — the gold standard for football
match prediction. Models each team's attack and defense strength, with:
- Exponential time-decay weighting (recent matches matter more)
- Low-score correlation correction (the original Dixon-Coles innovation)
- Contextual adjustments via a LightGBM residual layer (draw pressure, rest days)

**Features**: Elo ratings, squad market valuations (Transfermarkt),
rest-day effects, and draw-pressure dynamics.

**Ensemble**: A calibrated logistic meta-learner stacks Dixon-Coles probabilities
with market-implied odds and feature-derived signals. A separate market-blind
pipeline runs in parallel for genuine edge detection.

**Simulation**: 100,000 Monte Carlo runs of the full 48-team bracket per forecast
cycle, with FIFA tiebreaker rules and Bayesian penalty-shootout modelling.

**Optimization**: 231 hyperparameter configurations tested via two-phase automated
tuning, evaluated on leave-one-tournament-out cross-validation across 11
international tournaments (World Cups 2010--2022, Euros 2016--2024, Copa America
2016--2024).
""")


def _render_autotuner(st: object, settings: Settings) -> None:
    """Autotuner optimization journey."""

    leaderboard = data.load_autotune_leaderboard(settings)
    if leaderboard.is_empty():
        st.info("No autotuner data available.")
        return

    st.subheader("Hyperparameter Optimization Journey")

    phase1 = leaderboard.filter(pl.col("phase") == "phase1")
    phase2 = leaderboard.filter(pl.col("phase") == "phase2")

    cols = st.columns(3)
    cols[0].metric("Total experiments", leaderboard.height)
    cols[1].metric(
        "Phase 1 (structural)",
        f"{phase1.height} runs",
        help="Explores structural choices: which features to include, meta-learner family, GBM toggle.",
    )
    cols[2].metric(
        "Phase 2 (numeric)",
        f"{phase2.height} runs",
        help="Optimizes continuous hyperparameters (decay rate, regularization, Elo K-factor) via Optuna TPE.",
    )

    st.caption(
        "Each dot is one experiment. Phase 1 explores structural decisions "
        "(which features to enable, meta-learner type). Phase 2 fine-tunes "
        "numeric hyperparameters using Bayesian optimization (Optuna TPE). "
        "Lower RPS is better."
    )
    st.plotly_chart(charts.autotuner_chart(leaderboard), use_container_width=True)

    st.subheader("Simulations per Forecast")
    st.metric("Monte Carlo runs", "100,000",
              help="Each forecast cycle simulates the full 48-team bracket 100,000 times to compute stage-reaching probabilities.")


def _tournament_label(code: str) -> str:
    labels = {
        "WC2010": "World Cup 2010", "WC2014": "World Cup 2014",
        "WC2018": "World Cup 2018", "WC2022": "World Cup 2022",
        "EU2016": "Euro 2016", "EU2020": "Euro 2020", "EU2024": "Euro 2024",
        "CA2016": "Copa 2016", "CA2019": "Copa 2019",
        "CA2021": "Copa 2021", "CA2024": "Copa 2024",
    }
    return labels.get(code, code)
