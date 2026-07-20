"""Streamlit dashboard entry point.

Seven-page app built on ``st.navigation``/``st.Page`` so every page owns a
stable URL (``/overview``, ``/retrospective``, …) that can be shared and deep-linked.
``streamlit`` is imported lazily inside :func:`main` so the module imports
without the optional ``dashboard`` extra installed.

Run with ``streamlit run -m polymbappe.dashboard.app`` or via the
``polymbappe dashboard`` CLI command (wired separately in ``cli.py``).
"""

from __future__ import annotations

from collections.abc import Callable

from polymbappe.config import Settings
from polymbappe.dashboard import cache
from polymbappe.dashboard.pages import (
    match_predictor,
    model_showcase,
    overview,
    predictions_vs_actuals,
    retrospective,
    team_deep_dive,
    upset_watch,
)

REPO_URL = "https://github.com/pastchum/polymbappe"

#: ``(title, url_path, icon, renderer)`` per page, in sidebar order. ``url_path``
#: is a public URL — changing one breaks existing shared links.
PAGES: tuple[tuple[str, str, str, Callable[[Settings], None]], ...] = (
    ("Tournament Overview", "overview", ":material/leaderboard:", overview.render),
    # The final has resolved, so the completed-tournament story ranks right
    # after the landing page.
    ("Tournament Retrospective", "retrospective", ":material/history:", retrospective.render),
    ("Predictions vs Actuals", "results", ":material/fact_check:", predictions_vs_actuals.render),
    ("Match Predictor", "matches", ":material/sports_soccer:", match_predictor.render),
    ("Team Deep Dive", "teams", ":material/travel_explore:", team_deep_dive.render),
    ("Upset Watch", "upsets", ":material/bolt:", upset_watch.render),
    ("Model Showcase", "model", ":material/insights:", model_showcase.render),
)


def main() -> None:
    """Launch the Streamlit dashboard with URL-addressable multipage navigation."""

    import functools

    import streamlit as st

    st.set_page_config(
        page_title="Polymbappe — 2026 World Cup Forecast",
        page_icon="⚽",
        layout="wide",
    )

    cache.install()
    settings = Settings()

    from pathlib import Path

    logo_path = Path(settings.data_dir).parent / "data" / "polymbappe_logo.jpg"
    if not logo_path.exists():
        logo_path = Path(__file__).resolve().parents[3] / "data" / "polymbappe_logo.jpg"
    if logo_path.exists():
        st.logo(str(logo_path), size="large")
        # st.logo caps at ~32px even on "large"; render it half again bigger.
        st.html(
            """
            <style>
            img[data-testid="stLogo"] {
                height: 3.25rem;
                width: auto;
                max-width: 100%;
            }
            </style>
            """
        )

    nav = st.navigation(
        [
            st.Page(
                functools.partial(render, settings),
                title=title,
                icon=icon,
                url_path=url_path,
                default=(url_path == "overview"),
            )
            for title, url_path, icon, render in PAGES
        ]
    )

    nav.run()

    with st.sidebar:
        st.divider()
        st.markdown(f"[Source & methodology on GitHub]({REPO_URL})")


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes main() directly
    main()
