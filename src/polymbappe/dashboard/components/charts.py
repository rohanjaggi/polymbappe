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

from typing import TYPE_CHECKING

import polars as pl

from polymbappe.dashboard.data import STAGE_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing plotly at module load
    from plotly.graph_objects import Figure

# -- palette -------------------------------------------------------------------
#
# One palette for every chart, anchored on the polymbappe logo blue and
# validated (dataviz six-checks) for colorblind-safe adjacency and ≥3:1 contrast
# on the light surface; contrast also holds on the dark theme surface. Trace
# colors are static across Streamlit's light/dark modes, so they were chosen to
# work on both.

#: Brand blue (the logo color): primary accent, single-series marks, "model".
BRAND = "#2652f9"
#: Light step of the brand ramp: the "predicted/forecast" half of an ordered pair.
BRAND_LIGHT = "#97adfc"
#: Warm counterpoint: away side, actual-xG overlay.
ORANGE = "#eb6834"
#: Aqua: the draw slot between blue (home) and orange (away).
AQUA = "#1baf7a"
#: Muted mode-invariant gray for reference lines and market overlays — never a series.
NEUTRAL = "#898781"

#: home / draw / away — used everywhere those three outcomes appear, in this order.
HDA_COLORS: dict[str, str] = {"home": BRAND, "draw": AQUA, "away": ORANGE}

#: Fixed categorical order for multi-series charts (trajectory lines). Adjacent
#: pairs pass CVD separation; assign in order, never cycled.
CATEGORICAL: tuple[str, ...] = (
    BRAND, "#008300", "#e87ba4", "#eda100", AQUA, ORANGE, "#7b6ee0", "#e34948",
)


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
        return _empty_figure("Forecasts not published yet.")

    top = df.sort(prob_col, descending=True).head(n)
    teams = top["team"].to_list()
    probs = top[prob_col].to_list()
    fig = go.Figure(
        go.Bar(
            x=probs[::-1],
            y=teams[::-1],
            orientation="h",
            marker={"color": BRAND},
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="P(champion)",
        xaxis_tickformat=".0%",
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
            marker={"color": [HDA_COLORS["home"], HDA_COLORS["draw"], HDA_COLORS["away"]]},
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
            marker={"color": BRAND},
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
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
    outcomes = df["actual_outcome"].to_list()
    cats = [labels.get(o, o) for o in outcomes]
    fig = go.Figure(
        go.Bar(
            x=cats,
            y=df["accuracy"].to_list(),
            # Color follows the outcome entity (same mapping as the H/D/A bar),
            # not the row position — the frame arrives sorted alphabetically.
            marker={"color": [HDA_COLORS.get(o, NEUTRAL) for o in outcomes]},
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
            line={"dash": "dash", "color": NEUTRAL},
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["mean_confidence"].to_list(),
            y=df["hit_rate"].to_list(),
            mode="markers+lines",
            marker={"size": [min(40, 10 + 5 * c) for c in counts], "color": BRAND},
            line={"color": BRAND},
            customdata=counts,
            name="Model",
            hovertemplate=(
                "Predicted %{x:.0%}<br>Observed %{y:.0%}<br>n=%{customdata}<extra></extra>"
            ),
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


def xg_scatter(finished: pl.DataFrame, match_xg: pl.DataFrame | None = None) -> Figure:
    """Scatter of model predicted xG vs actual goals (and actual xG when available).

    Plots one point per team per match. When ``match_xg`` is supplied, adds a second
    series showing actual FotMob xG vs actual goals so you can separate model error
    from finishing-luck variance. The diagonal represents perfect prediction.
    """

    import plotly.graph_objects as go

    needed = {"exp_home_goals", "exp_away_goals", "home_goals", "away_goals"}
    if finished.is_empty() or not needed.issubset(finished.columns):
        return _empty_figure("No xG data yet — needs finished matches with predictions.")

    x_pred, y_goals, labels = [], [], []
    for r in finished.iter_rows(named=True):
        fixture = f"{r['home_team']} vs {r['away_team']}"
        x_pred += [float(r["exp_home_goals"]), float(r["exp_away_goals"])]
        y_goals += [float(r["home_goals"]), float(r["away_goals"])]
        labels += [f"{fixture} (H)", f"{fixture} (A)"]

    all_vals = x_pred + y_goals
    max_val = max(max(all_vals, default=0), 4.0) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line={"dash": "dash", "color": NEUTRAL},
        name="Perfect prediction",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_pred, y=y_goals,
        mode="markers",
        marker={"color": BRAND, "size": 9, "opacity": 0.75},
        text=labels,
        hovertemplate="<b>%{text}</b><br>Model xG: %{x:.2f}<br>Actual goals: %{y}<extra></extra>",
        name="Model xG vs goals",
    ))

    # Overlay actual FotMob xG vs goals when available.
    if match_xg is not None and not match_xg.is_empty():
        xg_slim = match_xg.select(["home_team", "away_team", "home_xg", "away_xg"])
        joined = finished.join(xg_slim, on=["home_team", "away_team"], how="inner")
        unmatched = finished.join(xg_slim, on=["home_team", "away_team"], how="anti")
        if not unmatched.is_empty():
            xg_rev = xg_slim.rename(
                {"home_team": "away_team", "away_team": "home_team",
                 "home_xg": "away_xg", "away_xg": "home_xg"}
            )
            rev_joined = unmatched.join(xg_rev, on=["home_team", "away_team"], how="inner")
            if not rev_joined.is_empty():
                joined = pl.concat([joined, rev_joined], how="diagonal_relaxed")
        if not joined.is_empty():
            x_actual_xg, y_actual_goals, xg_labels = [], [], []
            for r in joined.iter_rows(named=True):
                fixture = f"{r['home_team']} vs {r['away_team']}"
                x_actual_xg += [float(r["home_xg"]), float(r["away_xg"])]
                y_actual_goals += [float(r["home_goals"]), float(r["away_goals"])]
                xg_labels += [f"{fixture} (H)", f"{fixture} (A)"]
            max_val = max(max_val, max(x_actual_xg, default=0) + 0.5)
            fig.add_trace(go.Scatter(
                x=x_actual_xg, y=y_actual_goals,
                mode="markers",
                marker={"color": ORANGE, "size": 9, "opacity": 0.75, "symbol": "diamond"},
                text=xg_labels,
                hovertemplate=(
                    "<b>%{text}</b><br>FotMob xG: %{x:.2f}<br>Actual goals: %{y}<extra></extra>"
                ),
                name="Actual xG vs goals (luck)",
            ))

    fig.update_layout(
        title="xG error decomposition — model vs actual vs goals",
        xaxis_title="xG value",
        yaxis_title="Actual goals scored",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def group_standings_chart(
    predicted_df: pl.DataFrame, actual_df: pl.DataFrame, group: str
) -> Figure:
    """Grouped horizontal bar comparing predicted vs actual points for one group."""

    import plotly.graph_objects as go

    pred_g = predicted_df.filter(pl.col("group") == group).sort("predicted_points", descending=True)
    act_g = actual_df.filter(pl.col("group") == group)

    if pred_g.is_empty():
        return _empty_figure(f"No data for Group {group}.")

    teams = pred_g["team"].to_list()
    pred_pts = pred_g["predicted_points"].to_list()

    act_map = {}
    if not act_g.is_empty():
        for r in act_g.iter_rows(named=True):
            act_map[r["team"]] = r["points"]
    act_pts = [act_map.get(t, 0) for t in teams]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=teams[::-1], x=pred_pts[::-1], orientation="h",
        name="Predicted", marker={"color": BRAND_LIGHT},
        hovertemplate="%{y}: %{x:.1f} pts<extra>Predicted</extra>",
    ))
    fig.add_trace(go.Bar(
        y=teams[::-1], x=act_pts[::-1], orientation="h",
        name="Actual", marker={"color": BRAND},
        hovertemplate="%{y}: %{x} pts<extra>Actual</extra>",
    ))
    fig.update_layout(
        title=f"Group {group} — Predicted vs Actual Points",
        xaxis_title="Points",
        barmode="group",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def backtest_bar(per_tournament: dict[str, float]) -> Figure:
    """Bar chart of RPS per tournament from backtest results."""

    import plotly.graph_objects as go

    if not per_tournament:
        return _empty_figure("No backtest data available.")

    tournaments = sorted(per_tournament.keys())
    values = [per_tournament[t] for t in tournaments]
    mean_rps = sum(values) / len(values) if values else 0

    fig = go.Figure(go.Bar(
        x=tournaments, y=values,
        marker={"color": BRAND},
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=mean_rps, line_dash="dash", line_color=NEUTRAL,
                  annotation_text=f"Mean: {mean_rps:.3f}")
    fig.add_hline(y=0.21, line_dash="dot", line_color=NEUTRAL,
                  annotation_text="0.21 benchmark")
    fig.update_layout(
        title="RPS by Tournament (Leave-One-Out Backtest)",
        xaxis_title="Tournament",
        yaxis_title="Ranked Probability Score",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
    )
    return fig


def trajectory_lines(
    df: pl.DataFrame,
    *,
    n_teams: int = 8,
    market: pl.DataFrame | None = None,
    market_team: str | None = None,
) -> Figure:
    """Championship-probability trajectories over the tournament (retrospective page).

    ``df`` is the replay frame (``date, team, ..., champion``); the ``n_teams`` teams with
    the highest peak champion probability get a line each. When ``market``
    (``date, team, price``) and ``market_team`` are given, that team's Polymarket price is
    overlaid as a dashed line for a model-vs-market comparison.
    """

    import plotly.graph_objects as go

    if df.is_empty() or "champion" not in df.columns:
        return _empty_figure("The title-race replay has not been computed yet.")

    top = (
        df.group_by("team")
        .agg(pl.col("champion").max().alias("peak"))
        .sort("peak", descending=True)
        .head(n_teams)["team"]
        .to_list()
    )
    fig = go.Figure()
    for team in top:
        series = df.filter(pl.col("team") == team).sort("date")
        fig.add_trace(
            go.Scatter(
                x=series["date"].to_list(),
                y=series["champion"].to_list(),
                mode="lines+markers",
                name=team,
                hovertemplate=f"{team}: %{{y:.1%}} (%{{x}})<extra></extra>",
            )
        )
    if market is not None and market_team and not market.is_empty():
        series = market.filter(pl.col("team") == market_team).sort("date")
        if not series.is_empty():
            fig.add_trace(
                go.Scatter(
                    x=series["date"].to_list(),
                    y=series["price"].to_list(),
                    mode="lines",
                    name=f"{market_team} (Polymarket)",
                    line={"dash": "dash", "color": NEUTRAL},
                    hovertemplate=f"{market_team} market: %{{y:.1%}} (%{{x}})<extra></extra>",
                )
            )
    fig.update_layout(
        xaxis_title="Information cutoff (end of day)",
        yaxis_title="P(champion)",
        yaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "y": -0.2},
        colorway=list(CATEGORICAL),
    )
    return fig
