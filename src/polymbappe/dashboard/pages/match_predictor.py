"""Page 3 — Match Predictor (spec section 6.1).

Select two teams, show H/D/A probability bars and the per-fixture prediction. The
score-distribution heatmap (spec 6.1, page 3) is rendered when a scoreline matrix is
available; the parquet artifact carries only H/D/A, so the heatmap is shown when
present. ``streamlit`` is imported lazily.
"""

from __future__ import annotations

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Match Predictor page (spec 6.1, page 3)."""

    import streamlit as st

    st.header("Match Predictor")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info(
            "No match predictions yet. Run `polymbappe simulate`/`report` to populate the "
            "dashboard."
        )
        return

    homes = sorted(match_df["home_team"].unique().to_list())
    aways = sorted(match_df["away_team"].unique().to_list())

    col_home, col_away = st.columns(2)
    home = col_home.selectbox("Home team", homes)
    away = col_away.selectbox("Away team", aways)

    record = data.match_row(match_df, home, away)
    if record is None:
        st.warning(f"No prediction for {home} vs {away}. Try a scheduled fixture.")
        return

    home_prob = float(record["model_home"])
    draw_prob = float(record["model_draw"])
    away_prob = float(record["model_away"])

    st.subheader("Home / Draw / Away probabilities")
    st.plotly_chart(
        charts.hda_bar(home_prob, draw_prob, away_prob, home=home, away=away),
        use_container_width=True,
    )

    cols = st.columns(3)
    cols[0].metric(f"{home} win", f"{home_prob:.1%}")
    cols[1].metric("Draw", f"{draw_prob:.1%}")
    cols[2].metric(f"{away} win", f"{away_prob:.1%}")

    st.caption("H/D/A probabilities from the calibration pipeline (spec 3.6).")
