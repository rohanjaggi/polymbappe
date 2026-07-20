"""Model Showcase.

The method page: how the forecasting pipeline works (including the LangGraph
live-news agent and its nodes) and the leave-one-tournament-out backtest that
validated it.
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

    _render_pipeline_overview(st)
    st.divider()
    _render_agent_graph(st)
    st.divider()
    _render_backtest(st, settings)


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


#: The agent's LangGraph topology (mirrors ``agent/graph.py``): five nodes with
#: conditional short-circuits to END when a stage yields nothing material.
_AGENT_GRAPH_DOT = """
digraph agent {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#2652f9", color="#2652f9",
          fontcolor="white", fontname="Helvetica", fontsize=12];
    edge [color="#898781", fontcolor="#898781", fontname="Helvetica", fontsize=10];

    start [label="START", shape=circle, fillcolor="#898781", color="#898781", fontsize=10];
    finish [label="END", shape=doublecircle, fillcolor="#898781", color="#898781", fontsize=10];

    scan [label="Scan\\npull squad / injury news"];
    assess [label="Assess\\nclassify materiality"];
    xref [label="Cross-Reference\\nkeep net-new only"];
    act [label="Act\\napply changes, re-simulate"];
    reflect [label="Reflect\\nflag trophy-odds shifts"];

    start -> scan;
    scan -> assess;
    assess -> xref [label="material"];
    assess -> finish [label="nothing material", style=dashed];
    xref -> act [label="net-new"];
    xref -> finish [label="already known", style=dashed];
    act -> reflect;
    reflect -> finish;
}
"""


def _render_agent_graph(st: object) -> None:
    """The LangGraph live-news agent: the state machine and what each node does."""

    st.subheader("Live-News Agent (LangGraph)")
    st.markdown(
        "Between simulation cycles, a LangGraph state machine keeps the forecast "
        "current. Each cycle threads shared state through five nodes — **Scan** "
        "pulls raw news items across configured sources, **Assess** classifies "
        "them and keeps only material findings, **Cross-Reference** drops "
        "anything already reflected in the ratings, **Act** applies the changes "
        "(status updates, changelog, optional re-simulation), and **Reflect** "
        "flags teams whose trophy probability moved beyond the significance "
        "threshold. Assess and Cross-Reference short-circuit straight to END "
        "when nothing material or net-new is found, so quiet news days cost "
        "nothing."
    )
    st.graphviz_chart(_AGENT_GRAPH_DOT)


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
            f"{best_rps:.3f}",
            delta=f"{(1 - best_rps / 0.21) * 100:.0f}% better than the 0.21 benchmark",
            delta_color="normal",
        )
        row1[1].metric("Tournaments tested", len(per_tournament))
        row1[2].metric("Beat 0.21 benchmark", f"{beaten}/{len(per_tournament)}")
        row1[3].metric(
            "Best tournament",
            f"{_tournament_label(best_tourney)}",
            delta=f"RPS {best_tourney_rps:.3f}",
            delta_color="off",
        )

        st.caption(
            "Leave-one-tournament-out cross-validation: the model is trained on 10 tournaments "
            "and evaluated on the held-out one. This tests generalization, not memorization. "
            "The 0.21 benchmark is a typical baseline for 3-way international football prediction."
        )
        st.plotly_chart(charts.backtest_bar(per_tournament), width="stretch")
    else:
        st.metric("Mean RPS", f"{best_rps:.3f}")


def _tournament_label(code: str) -> str:
    labels = {
        "WC2010": "World Cup 2010", "WC2014": "World Cup 2014",
        "WC2018": "World Cup 2018", "WC2022": "World Cup 2022",
        "EU2016": "Euro 2016", "EU2020": "Euro 2020", "EU2024": "Euro 2024",
        "CA2016": "Copa 2016", "CA2019": "Copa 2019",
        "CA2021": "Copa 2021", "CA2024": "Copa 2024",
    }
    return labels.get(code, code)
