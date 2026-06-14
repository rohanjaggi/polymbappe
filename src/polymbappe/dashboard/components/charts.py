"""Plotly chart builders for the dashboard (spec section 6.1).

Each function takes a Polars DataFrame (or a numeric matrix) and returns a Plotly
``Figure``. ``plotly`` is an optional dependency (spec section 13, ``dashboard``
extra) so it is imported lazily inside each builder, keeping this module importable
without it. Builders never call Streamlit — pages pass the returned figures to
``st.plotly_chart``.

Builders tolerate empty inputs by returning an empty (but valid) figure with an
explanatory annotation, so pages render a graceful "no data yet" state before the
first ``polymbappe simulate`` run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from polymbappe.dashboard.data import STAGE_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing plotly at module load
    from plotly.graph_objects import Figure


def _empty_figure(message: str) -> Figure:
    """A blank figure carrying an explanatory annotation for missing data."""

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16})
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def trophy_bar(df: pl.DataFrame, *, prob_col: str = "champion", n: int = 10) -> Figure:
    """Horizontal bar chart of trophy probability per team (spec 6.1, page 1).

    Expects a stage-probabilities frame with ``team`` and ``champion`` columns.
    """

    import plotly.graph_objects as go

    if df.is_empty() or "team" not in df.columns or prob_col not in df.columns:
        return _empty_figure("No simulation results yet — run `polymbappe simulate`.")

    top = df.sort(prob_col, descending=True).head(n)
    teams = top["team"].to_list()
    probs = top[prob_col].to_list()
    fig = go.Figure(
        go.Bar(
            x=probs[::-1],
            y=teams[::-1],
            orientation="h",
            marker={"color": "seagreen"},
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Trophy probability — top {len(teams)}",
        xaxis_title="P(champion)",
        xaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def advancement_heatmap(df: pl.DataFrame, *, teams: Sequence[str] | None = None) -> Figure:
    """Heatmap of group-finish probabilities, team x finish position (spec 6.1, page 1).

    Expects a group-probabilities frame with ``team`` and ``finish_1..finish_4``.
    """

    import plotly.express as px

    finish_cols = ["finish_1", "finish_2", "finish_3", "finish_4"]
    if df.is_empty() or "team" not in df.columns or not all(c in df.columns for c in finish_cols):
        return _empty_figure("No group probabilities yet — run `polymbappe simulate`.")

    frame = df
    if teams is not None:
        frame = frame.filter(pl.col("team").is_in(list(teams)))
    if frame.is_empty():
        return _empty_figure("No teams selected.")

    frame = frame.sort("finish_1", descending=True)
    matrix = frame.select(finish_cols).to_numpy()
    fig = px.imshow(
        matrix,
        x=["1st", "2nd", "3rd", "4th"],
        y=frame["team"].to_list(),
        color_continuous_scale="Greens",
        aspect="auto",
        labels={"x": "Group finish", "y": "Team", "color": "Probability"},
    )
    fig.update_layout(
        title="Group advancement probabilities",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def score_heatmap(matrix: Sequence[Sequence[float]], *, max_goals: int | None = None) -> Figure:
    """Heatmap of a scoreline probability matrix, home goals x away goals (spec 6.1, page 3).

    ``matrix[i][j]`` is P(home scores ``i``, away scores ``j``).
    """

    import numpy as np
    import plotly.express as px

    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0 or arr.ndim != 2:
        return _empty_figure("No scoreline matrix available.")

    if max_goals is not None:
        arr = arr[: max_goals + 1, : max_goals + 1]
    rows, cols = arr.shape
    fig = px.imshow(
        arr,
        x=[str(j) for j in range(cols)],
        y=[str(i) for i in range(rows)],
        color_continuous_scale="Blues",
        aspect="auto",
        labels={"x": "Away goals", "y": "Home goals", "color": "Probability"},
    )
    fig.update_layout(
        title="Score distribution",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def hda_bar(
    home_prob: float, draw_prob: float, away_prob: float, *, home: str, away: str
) -> Figure:
    """Bar chart of home / draw / away probabilities for one fixture (spec 6.1, page 3)."""

    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=[f"{home} win", "Draw", f"{away} win"],
            y=[home_prob, draw_prob, away_prob],
            marker={"color": ["seagreen", "goldenrod", "indianred"]},
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{home} vs {away}",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def stage_waterfall(stage_probs: dict[str, float], *, team: str) -> Figure:
    """Stage-reaching probability waterfall for one team (spec 6.1, page 2).

    ``stage_probs`` maps stage keys (``R32..champion``) to probabilities.
    """

    import plotly.graph_objects as go

    if not stage_probs:
        return _empty_figure(f"No stage probabilities for {team}.")

    stages = [s for s in STAGE_COLUMNS if s in stage_probs]
    values = [stage_probs[s] for s in stages]
    fig = go.Figure(
        go.Bar(
            x=stages,
            y=values,
            marker={"color": "steelblue"},
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{team} — stage-reaching probabilities",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def outcome_accuracy_bar(df: pl.DataFrame) -> Figure:
    """Bar chart of top-pick accuracy per realized outcome (spec 6.1, page 7).

    Expects the frame from :func:`polymbappe.dashboard.data.accuracy_by_outcome`
    (``actual_outcome``, ``n``, ``accuracy``).
    """

    import plotly.graph_objects as go

    if df.is_empty() or "accuracy" not in df.columns:
        return _empty_figure("No finished matches yet — accuracy needs recorded results.")

    labels = {"home": "Home win", "draw": "Draw", "away": "Away win"}
    cats = [labels.get(o, o) for o in df["actual_outcome"].to_list()]
    fig = go.Figure(
        go.Bar(
            x=cats,
            y=df["accuracy"].to_list(),
            marker={"color": ["seagreen", "goldenrod", "indianred"][: len(cats)]},
            customdata=df["n"].to_list(),
            hovertemplate="%{x}: %{y:.0%} (n=%{customdata})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top-pick accuracy by actual outcome",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def calibration_curve(df: pl.DataFrame) -> Figure:
    """Reliability diagram of forecast confidence vs. observed hit rate (spec 6.1, page 7).

    Expects the frame from :func:`polymbappe.dashboard.data.calibration_bins`
    (``mean_confidence``, ``hit_rate``, ``count``). Plots the model's points against the
    perfect-calibration diagonal; marker size scales with the number of matches per bin.
    """

    import plotly.graph_objects as go

    needed = {"mean_confidence", "hit_rate"}
    if df.is_empty() or not needed.issubset(df.columns):
        return _empty_figure("No finished matches yet — calibration needs recorded results.")

    counts = df["count"].to_list() if "count" in df.columns else [1] * df.height
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["mean_confidence"].to_list(),
            y=df["hit_rate"].to_list(),
            mode="markers+lines",
            marker={"size": [min(40, 10 + 5 * c) for c in counts], "color": "steelblue"},
            line={"color": "steelblue"},
            customdata=counts,
            name="Model",
            hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<br>n=%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Calibration — predicted confidence vs. observed hit rate",
        xaxis_title="Predicted confidence (favourite)",
        yaxis_title="Observed hit rate",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def edge_scatter(df: pl.DataFrame) -> Figure:
    """Scatter of edge magnitude vs. Kelly stake for flagged edges (spec 6.1, page 4).

    Expects an edges frame with ``edge_bps``, ``kelly_fraction``, ``match_id``,
    ``outcome``.
    """

    import plotly.express as px

    needed = {"edge_bps", "kelly_fraction"}
    if df.is_empty() or not needed.issubset(df.columns):
        return _empty_figure("No market edges flagged — run `polymbappe edges`.")

    pdf = df.to_pandas()
    pdf["abs_edge_bps"] = pdf["edge_bps"].abs()
    hover = [c for c in ("match_id", "outcome") if c in pdf.columns]
    fig = px.scatter(
        pdf,
        x="abs_edge_bps",
        y="kelly_fraction",
        hover_data=hover or None,
        labels={"abs_edge_bps": "|edge| (bps)", "kelly_fraction": "Kelly fraction"},
    )
    fig.update_layout(
        title="Market edges — magnitude vs. stake",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig
