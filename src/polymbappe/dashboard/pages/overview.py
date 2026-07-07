"""Page 1 — Tournament Overview.

Model scorecard hero, trophy probability leaderboard, biggest surprises,
and group-stage predictions vs actuals.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Tournament Overview page."""

    import streamlit as st

    st.header("Tournament Overview")

    stage_df = data.load_stage_probabilities(settings)
    match_df = data.load_match_predictions(settings)

    if stage_df.is_empty():
        st.info("No simulation results yet. Run `polymbappe simulate` to populate the dashboard.")
        return

    _render_scorecard(st, settings, match_df)
    st.divider()
    _render_trophy_leaderboard(st, stage_df)
    st.divider()
    _render_biggest_surprises(st, settings, match_df)
    st.divider()
    _render_group_comparison(st, settings, match_df)


def _render_scorecard(st: object, settings: Settings, match_df: pl.DataFrame) -> None:
    """Model performance headline metrics."""

    if match_df.is_empty():
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)

    if finished.is_empty():
        return

    scorecard = data.prediction_scorecard(finished)
    n = int(scorecard["n"])

    st.subheader("Model Scorecard")
    cols = st.columns(4)
    cols[0].metric("Matches Predicted", n)
    cols[1].metric("Top-Pick Accuracy", f"{scorecard['accuracy']:.0%}")
    cols[2].metric(
        "Brier Score",
        f"{scorecard['brier_score']:.3f}",
        help="Mean squared error over H/D/A — lower is better (0 best, 2 worst).",
    )
    cols[3].metric(
        "Log Loss",
        f"{scorecard['log_loss']:.3f}",
        help="Mean negative log-probability of the realized outcome — lower is better.",
    )


def _render_trophy_leaderboard(st: object, stage_df: pl.DataFrame) -> None:
    """Trophy probability bar chart + top contenders table."""

    st.subheader("Trophy Probability Leaderboard")
    st.plotly_chart(charts.trophy_bar(stage_df, n=10), use_container_width=True)
    st.dataframe(
        data.top_contenders(stage_df, n=10).to_pandas(),
        use_container_width=True,
    )


def _render_biggest_surprises(
    st: object, settings: Settings, match_df: pl.DataFrame
) -> None:
    """Matches where the model was most wrong."""

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)

    if finished.is_empty():
        return

    st.subheader("Biggest Surprises")
    st.caption(
        "Matches where the model assigned the lowest probability to what actually happened."
    )
    surprises = data.biggest_surprises(finished, n=5)
    if not surprises.is_empty():
        st.dataframe(surprises.to_pandas(), use_container_width=True, hide_index=True)


def _render_group_comparison(
    st: object, settings: Settings, match_df: pl.DataFrame
) -> None:
    """Predicted vs actual group standings."""

    results = data.tournament_results(data.load_recorded_results(settings))
    actual = data.compute_group_standings(match_df, results)
    predicted = data.predicted_group_points(match_df)

    if actual.is_empty() or predicted.is_empty():
        return

    st.subheader("Group Stage: Predicted vs Actual Points")
    st.caption(
        "How well the model predicted each team's group-stage points. "
        "Predicted points = 3·P(win) + P(draw) summed across group fixtures."
    )

    # Overall MAE
    merged = predicted.join(
        actual.select(["team", "points"]),
        on="team",
        how="inner",
    )
    if not merged.is_empty():
        mae = float(
            (merged["predicted_points"] - merged["points"].cast(pl.Float64)).abs().mean()
        )
        st.metric("Mean Absolute Error (points)", f"{mae:.2f}")

    groups = sorted(actual["group"].unique().to_list())
    left, right = st.columns(2)
    for i, group in enumerate(groups):
        col = left if i % 2 == 0 else right
        with col:
            st.plotly_chart(
                charts.group_standings_chart(predicted, actual, group),
                use_container_width=True,
            )
