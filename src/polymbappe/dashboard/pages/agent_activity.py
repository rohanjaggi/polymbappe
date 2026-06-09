"""Page 6 — Agent Activity (spec sections 5.3 & 6.1).

Live feed of the LangGraph monitoring agent: changelog entries, player status board,
and significant-shift notifications (spec 5.2 Reflect node). ``streamlit`` is
imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data


def render(settings: Settings) -> None:
    """Render the Agent Activity page (spec 6.1, page 6)."""

    import streamlit as st

    st.header("Agent Activity")

    changelog = data.load_agent_changelog(settings)
    if changelog.is_empty():
        st.info(
            "No agent activity yet. The LangGraph agent writes `agent_changelog.parquet` "
            "on each cycle (spec 5.3)."
        )
        return

    st.subheader("Changelog feed")
    feed = changelog
    if "timestamp" in feed.columns:
        feed = feed.sort("timestamp", descending=True)
    st.dataframe(feed.to_pandas(), use_container_width=True)

    if "prob_shift" in changelog.columns:
        st.subheader("Significant shifts (> 0.5pp)")
        significant = changelog.filter(pl.col("prob_shift").abs() > 0.005)
        if significant.is_empty():
            st.caption("No shifts above the 0.5pp notification threshold (spec 5.2).")
        else:
            st.dataframe(significant.to_pandas(), use_container_width=True)

    if "player" in changelog.columns and "team" in changelog.columns:
        st.subheader("Player status board")
        candidate_cols = ("player", "team", "change", "timestamp")
        status_cols = [c for c in candidate_cols if c in changelog.columns]
        latest = changelog.select(status_cols)
        if "timestamp" in latest.columns:
            latest = latest.sort("timestamp", descending=True)
        latest = latest.unique(subset=["player"], keep="first")
        st.dataframe(latest.to_pandas(), use_container_width=True)
