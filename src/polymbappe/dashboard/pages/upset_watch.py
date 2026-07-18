"""Page 5 — Upset Watch.

Two sections: retrospective upsets that happened, and forward-looking dark
horses still alive in the tournament.
"""

from __future__ import annotations

from polymbappe.config import Settings
from polymbappe.dashboard import data


def render(settings: Settings) -> None:
    """Render the Upset Watch page."""

    import streamlit as st

    st.header("Upset Watch")

    _render_upsets_that_happened(st, settings)
    st.divider()
    _render_dark_horses(st, settings)


def _render_upsets_that_happened(st: object, settings: Settings) -> None:
    """Matches where the underdog won against model expectations."""

    st.subheader("Upsets That Happened")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("No match predictions yet.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)

    if finished.is_empty():
        st.info("No finished matches yet.")
        return

    upsets = data.actual_upsets(finished, threshold=0.35)

    if upsets.is_empty() or (
        upsets.height == 1
        and "Fixture" in upsets.columns
        and upsets["Fixture"].to_list() == [None]
    ):
        st.caption("No upsets so far — the model's favoured outcome has won every match!")
        return

    total = finished.height
    upset_count = upsets.height
    st.metric(
        "Upsets",
        f"{upset_count} out of {total} matches ({upset_count / total:.0%})",
    )
    st.caption(
        "Matches where the model's favoured outcome lost and the actual result had "
        "less than 35% predicted probability. Sorted by upset magnitude."
    )
    st.dataframe(upsets.to_pandas(), use_container_width=True, hide_index=True)


def _render_dark_horses(st: object, settings: Settings) -> None:
    """Teams punching above their weight in advancement odds."""

    st.subheader("Dark Horses Still Standing")

    stage_df = data.load_stage_probabilities(settings)
    if stage_df.is_empty():
        st.info("No simulation results yet.")
        return

    horses = data.dark_horses(stage_df, n=10)
    if horses.is_empty():
        st.caption("No dark horse candidates — all remaining teams are favourites.")
        return

    st.caption(
        "Teams with championship odds under 5% but disproportionately high advancement "
        "probabilities. Sorted by overperformance score (QF probability relative to "
        "champion probability)."
    )
    st.dataframe(horses.to_pandas(), use_container_width=True, hide_index=True)
