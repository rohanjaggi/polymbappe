"""Streamlit dashboard entry point.

Six-page sidebar navigation dispatching to the page renderers in
:mod:`polymbappe.dashboard.pages`. ``streamlit`` is imported lazily inside
:func:`main` so the module imports without the optional ``dashboard`` extra
installed.

Run with ``streamlit run -m polymbappe.dashboard.app`` or via the
``polymbappe dashboard`` CLI command (wired separately in ``cli.py``).
"""

from __future__ import annotations

from collections.abc import Callable

from polymbappe.config import Settings
from polymbappe.dashboard.pages import (
    match_predictor,
    model_showcase,
    overview,
    predictions_vs_actuals,
    team_deep_dive,
    upset_watch,
)

PAGES: dict[str, Callable[[Settings], None]] = {
    "Tournament Overview": overview.render,
    "Match Predictor": match_predictor.render,
    "Predictions vs Actuals": predictions_vs_actuals.render,
    "Team Deep Dive": team_deep_dive.render,
    "Upset Watch": upset_watch.render,
    "Model Showcase": model_showcase.render,
}


def main() -> None:
    """Launch the Streamlit dashboard with six-page sidebar navigation."""

    import streamlit as st

    st.set_page_config(page_title="Polymbappe — 2026 World Cup Forecast", layout="wide")
    st.sidebar.title("Polymbappe")
    st.sidebar.caption("2026 FIFA World Cup forecasting")

    settings = Settings()
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[choice](settings)


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes main() directly
    main()
