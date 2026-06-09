"""Page 4 — Market Edges (spec section 6.1).

Table of model vs. market divergences (spec 3.6) sorted by edge magnitude weighted
by conviction (Kelly stake), with a magnitude-vs-stake scatter and a stage/direction
filter. ``streamlit`` is imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Market Edges page (spec 6.1, page 4)."""

    import streamlit as st

    st.header("Market Edges")

    edges_df = data.load_edges(settings)
    if edges_df.is_empty():
        st.info("No market edges yet. Run `polymbappe edges` to populate the dashboard.")
        return

    direction = st.radio(
        "Edge direction",
        ("All", "Model higher (value)", "Model lower (fade)"),
        horizontal=True,
    )
    filtered = edges_df
    if direction == "Model higher (value)":
        filtered = filtered.filter(pl.col("edge") > 0)
    elif direction == "Model lower (fade)":
        filtered = filtered.filter(pl.col("edge") < 0)

    prioritized = data.edges_by_priority(filtered)

    st.subheader("Flagged edges (sorted by |edge| x Kelly)")
    st.dataframe(prioritized.to_pandas(), use_container_width=True)

    st.subheader("Edge magnitude vs. stake")
    st.plotly_chart(charts.edge_scatter(prioritized), use_container_width=True)

    st.caption(
        "Edges where the model diverges from the market by more than the threshold "
        "(spec 3.6). Genuine edges come from the market-blind edge pipeline."
    )
