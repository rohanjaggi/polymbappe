"""Page 1 — Tournament Overview (spec section 6.1).

Trophy probability leaderboard, group-by-group advancement heatmap, and key
metrics (last simulation timestamp / data freshness). ``streamlit`` is imported
lazily so the module imports without the optional ``dashboard`` extra.
"""

from __future__ import annotations

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Tournament Overview page (spec 6.1, page 1)."""

    import streamlit as st

    st.header("Tournament Overview")

    stage_df = data.load_stage_probabilities(settings)
    group_df = data.load_group_probabilities(settings)

    if stage_df.is_empty():
        st.info("No simulation results yet. Run `polymbappe simulate` to populate the dashboard.")
        return

    st.subheader("Trophy probability leaderboard")
    contenders = data.top_contenders(stage_df, n=10)
    st.plotly_chart(charts.trophy_bar(stage_df, n=10), use_container_width=True)
    st.dataframe(contenders.to_pandas(), use_container_width=True)

    st.subheader("Group advancement probabilities")
    teams = data.top_contenders(stage_df, n=16)["team"].to_list()
    st.plotly_chart(
        charts.advancement_heatmap(group_df, teams=teams or None),
        use_container_width=True,
    )

    st.subheader("Key metrics & data freshness")
    freshness = data.data_freshness(settings)
    cols = st.columns(3)
    cols[0].metric("Teams simulated", stage_df.height)
    cols[1].metric("Last simulation", freshness.get("stage_probabilities.parquet", "missing"))
    cols[2].metric("Edges artifact", freshness.get("edges.parquet", "missing"))
    st.caption("Output artifact freshness (UTC):")
    st.json(freshness)
