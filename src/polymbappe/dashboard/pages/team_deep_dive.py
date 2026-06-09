"""Page 2 — Team Deep Dive (spec section 6.1).

Team selector, stage-reaching probability waterfall, and group-finish breakdown for
a single team. ``streamlit`` is imported lazily.
"""

from __future__ import annotations

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Team Deep Dive page (spec 6.1, page 2)."""

    import streamlit as st

    st.header("Team Deep Dive")

    stage_df = data.load_stage_probabilities(settings)
    group_df = data.load_group_probabilities(settings)

    teams = data.available_teams(stage_df)
    if not teams:
        st.info("No simulation results yet. Run `polymbappe simulate` to populate the dashboard.")
        return

    team = st.selectbox("Select a team", teams)

    st.subheader("Stage-reaching probability waterfall")
    stage_probs = data.team_stage_row(stage_df, team)
    st.plotly_chart(charts.stage_waterfall(stage_probs, team=team), use_container_width=True)

    st.subheader("Stage-reaching probabilities")
    st.dataframe(
        stage_df.filter(stage_df["team"] == team).to_pandas(),
        use_container_width=True,
    )

    if not group_df.is_empty() and "team" in group_df.columns:
        st.subheader("Group-finish probabilities")
        team_group = group_df.filter(group_df["team"] == team)
        if team_group.is_empty():
            st.caption("No group-finish data for this team.")
        else:
            st.dataframe(team_group.to_pandas(), use_container_width=True)
