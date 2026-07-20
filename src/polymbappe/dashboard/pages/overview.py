"""Tournament Overview.

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

    champion = data.champion_team(stage_df)
    if champion is not None:
        st.markdown(f"## 🏆 {champion} — 2026 World Cup champions")
        
    updated = data.last_updated(settings)
    if updated is not None:
        st.caption(
            f"Forecasts last updated {updated:%d %b %Y, %H:%M} UTC · "
            "[source & methodology](https://github.com/pastchum/polymbappe)"
        )

    if stage_df.is_empty():
        st.info(
            "Forecasts haven't been published yet — the leaderboard, scorecard and "
            "group-stage comparisons appear here once the first simulation results land."
        )
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
    cols[0].metric("Matches scored", n)
    cols[1].metric(
        "Top-pick accuracy",
        f"{scorecard['accuracy']:.1%}",
        help=(
            "Share of matches where the model's most likely outcome happened. "
            "Picking home/draw/away at random ≈ 33%."
        ),
    )
    cols[2].metric(
        "RPS",
        f"{scorecard['rps']:.3f}",
        delta=f"{scorecard['rps_skill']:.0%} better than a uniform guess",
        delta_color="normal",
        help=(
            "Ranked probability score — grades the full home/draw/away probabilities, "
            "not just the top pick. Lower is better; the benchmark is a uniform "
            "⅓/⅓/⅓ forecast scored on the same matches."
        ),
    )
    cols[3].metric(
        "Brier score",
        f"{scorecard['brier_score']:.3f}",
        delta=f"{scorecard['brier_skill']:.0%} better than a uniform guess",
        delta_color="normal",
        help=(
            "Mean squared error of the home/draw/away probabilities. Lower is "
            "better; the benchmark is the same uniform forecast."
        ),
    )


def _render_trophy_leaderboard(st: object, stage_df: pl.DataFrame) -> None:
    """Title race while it's live; how the field finished once it's decided."""

    if data.champion_team(stage_df) is not None:
        st.subheader("How the Field Finished")
        standings = data.final_standings(stage_df)
        early = ("Round of 32", "Group stage")
        deep_runs = standings.filter(~pl.col("result").is_in(early))
        early_exits = standings.filter(pl.col("result").is_in(early))
        st.dataframe(deep_runs.to_pandas(), width="stretch", hide_index=True)
        if not early_exits.is_empty():
            with st.expander(f"Earlier exits ({early_exits.height} teams)"):
                st.dataframe(
                    early_exits.to_pandas(), width="stretch", hide_index=True
                )
        return

    st.subheader("Trophy Probability Leaderboard")
    alive = stage_df.filter(pl.col("champion") > 0)
    st.caption(
        f"{alive.height} of {stage_df.height} teams can still win the title. "
        "Eliminated teams are excluded."
    )
    st.plotly_chart(charts.trophy_bar(alive, n=10), width="stretch")
    st.dataframe(
        data.top_contenders(alive, n=10).to_pandas(),
        width="stretch",
        hide_index=True,
        column_config={
            "team": st.column_config.TextColumn("Team"),
            **{
                stage: st.column_config.NumberColumn(stage, format="percent")
                for stage in data.STAGE_COLUMNS
            },
        },
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
        st.dataframe(surprises.to_pandas(), width="stretch", hide_index=True)


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
                width="stretch",
            )
