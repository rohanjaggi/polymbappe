"""Tournament Retrospective page: the completed-tournament story (spec 6.1 addendum).

Everything here is *summary* — per-match scoring detail lives on Predictions vs
Actuals. Sections: headline scorecard, per-round accuracy, the honest replay
trajectory (``polymbappe trajectory``), the bookmaker head-to-head, and the
champion-market P&L when Polymarket history was available.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Tournament Retrospective page."""

    import streamlit as st

    st.header("Tournament Retrospective")

    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))
    if match_df.is_empty():
        st.info("The retrospective appears once predictions are published and results are in.")
        return
    _, finished = data.split_fixtures(match_df, results)
    if finished.is_empty():
        st.info("No finished matches recorded yet — the retrospective needs results.")
        return

    _render_headline(st, finished)
    st.divider()
    _render_per_round(st, match_df, results, settings)
    st.divider()
    _render_trajectory(st, settings)
    st.divider()
    _render_bookmaker(st, finished, settings)
    st.divider()
    _render_market_pnl(st, settings)


def _render_headline(st: object, finished: pl.DataFrame) -> None:
    card = data.prediction_scorecard(finished)
    st.subheader(f"Final scorecard — {int(card['n'])} matches")
    cols = st.columns(4)
    cols[0].metric(
        "Top-pick accuracy",
        f"{card['accuracy']:.1%}",
        help="Picking home/draw/away at random ≈ 33%.",
    )
    cols[1].metric("RPS", f"{card['rps']:.3f}", help=f"Skill vs uniform: {card['rps_skill']:.1%}")
    cols[2].metric("Brier", f"{card['brier_score']:.3f}", help=f"Skill: {card['brier_skill']:.1%}")
    cols[3].metric(
        "Log loss", f"{card['log_loss']:.3f}", help=f"Skill: {card['log_loss_skill']:.1%}"
    )


def _render_per_round(
    st: object, match_df: pl.DataFrame, results: pl.DataFrame, settings: Settings
) -> None:
    st.subheader("Accuracy by round")
    table = data.per_round_accuracy(match_df, results, data.load_schedule(settings))
    if table.is_empty():
        st.info("No completed rounds yet.")
        return
    st.dataframe(
        table.with_columns(
            (pl.col("accuracy") * 100).round(1).alias("accuracy %"),
        ).drop("accuracy", "avg_p_actual"),
        hide_index=True,
    )


def _render_trajectory(st: object, settings: Settings) -> None:
    st.subheader("Championship probability through the tournament")
    trajectory = data.load_champion_trajectory(settings)
    if trajectory.is_empty():
        st.info(
            "The day-by-day title-race replay hasn't been computed yet. When it is, "
            "this chart reconstructs how each contender's championship odds evolved "
            "using only information available on each date — no hindsight."
        )
        return
    st.caption(
        "Each point re-simulates the tournament using only information available at that "
        "date — the Dixon-Coles model is refit on the pre-date history, played results "
        "are locked, and the real bracket is walked. No hindsight."
    )
    market = _market_history_frame(settings)
    market_team = _champion_team(trajectory)
    st.plotly_chart(
        charts.trajectory_lines(trajectory, market=market, market_team=market_team),
        width="stretch",
    )


def _market_history_frame(settings: Settings) -> pl.DataFrame | None:
    """Daily market prices reconstructed from the P&L frame (its date x team x price rows)."""

    pnl = data.load_market_pnl(settings)
    if pnl.is_empty():
        return None
    return pnl.select("date", "team", pl.col("market_price").alias("price"))


def _champion_team(trajectory: pl.DataFrame) -> str | None:
    last = trajectory.filter(pl.col("date") == trajectory["date"].max())
    decided = last.filter(pl.col("champion") >= 0.999)
    if decided.is_empty():
        return None
    return str(decided.row(0, named=True)["team"])


def _render_bookmaker(st: object, finished: pl.DataFrame, settings: Settings) -> None:
    st.subheader("Model vs bookmaker favorites")
    comparison = data.bookmaker_comparison(finished, settings)
    if not comparison.get("available"):
        st.info(f"Bookmaker workbook comparison unavailable: {comparison.get('reason', '—')}")
        return
    card = data.prediction_scorecard(finished)
    cols = st.columns(3)
    cols[0].metric(
        "Model accuracy (head-to-head)",
        f"{comparison['model_accuracy']:.1%}",
        help="Model top-pick accuracy on the matches both the model and the "
        "bookmaker workbook graded.",
    )
    cols[1].metric("Bookmaker accuracy", f"{comparison['book_accuracy']:.1%}")
    cols[2].metric(
        "Model accuracy (all matches)",
        f"{card['accuracy']:.1%}",
        help=f"Across all {int(card['n'])} scored matches of the tournament.",
    )


def _render_market_pnl(st: object, settings: Settings) -> None:
    st.subheader("Champion-market P&L (Polymarket)")
    pnl = data.load_market_pnl(settings)
    if pnl.is_empty():
        st.info(
            "The champion-market P&L appears once the Polymarket champion market "
            "resolves and its price history has been settled against the model's edges."
        )
        return
    total_staked = float(pnl["stake"].sum())
    total_profit = float(pnl["profit"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    cols = st.columns(3)
    cols[0].metric("Bets placed", f"{pnl.height}")
    cols[1].metric("Total staked", f"{total_staked:.3f} u")
    cols[2].metric("P&L", f"{total_profit:+.3f} u", f"{roi:+.1%} ROI")
    st.caption(
        "Quarter-Kelly stakes on positive model-vs-market edges (>3pp) at each replay "
        "date, settled at resolution. Long-Yes positions only."
    )
    cumulative = (
        pnl.sort("date")
        .group_by("date", maintain_order=True)
        .agg(pl.col("profit").sum())
        .with_columns(pl.col("profit").cum_sum().alias("cumulative"))
    )
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=cumulative["date"].to_list(),
            y=cumulative["cumulative"].to_list(),
            mode="lines+markers",
            fill="tozeroy",
            line={"color": charts.BRAND},
            fillcolor="rgba(38, 82, 249, 0.15)",
        )
    )
    fig.update_layout(
        title="Cumulative P&L by bet date (units)",
        xaxis_title="Bet date",
        yaxis_title="Cumulative profit (u)",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    st.plotly_chart(fig, width="stretch")
