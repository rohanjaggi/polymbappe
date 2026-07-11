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
    knockout_bracket,
    match_predictor,
    model_showcase,
    overview,
    predictions_vs_actuals,
    team_deep_dive,
    upset_watch,
)

PAGES: dict[str, Callable[[Settings], None]] = {
    "Tournament Overview": overview.render,
    "Team Deep Dive": team_deep_dive.render,
    "Match Predictor": match_predictor.render,
    "Knockout Bracket": knockout_bracket.render,
    "Predictions vs Actuals": predictions_vs_actuals.render,
    "Upset Watch": upset_watch.render,
    "Model Showcase": model_showcase.render,
}


def main() -> None:
    """Launch the Streamlit dashboard with six-page sidebar navigation."""

    import streamlit as st

    st.set_page_config(page_title="Polymbappe — 2026 World Cup Forecast", layout="wide")

    from pathlib import Path

    logo_path = Path(Settings().data_dir).parent / "data" / "polymbappe_logo.jpg"
    if not logo_path.exists():
        logo_path = Path(__file__).resolve().parents[3] / "data" / "polymbappe_logo.jpg"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.title("Polymbappe")

    st.sidebar.caption("2026 FIFA World Cup forecasting")

    settings = Settings()
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[choice](settings)


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes main() directly
    main()
