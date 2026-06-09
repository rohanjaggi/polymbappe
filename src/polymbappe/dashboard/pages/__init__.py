"""Streamlit dashboard pages (spec section 6.1).

Each page module exposes ``render(settings: Settings) -> None`` which lazily imports
``streamlit``, reads artifacts through :mod:`polymbappe.dashboard.data`, and renders
its page. The six pages mirror spec section 6.1:

1. :mod:`polymbappe.dashboard.pages.overview` — Tournament Overview
2. :mod:`polymbappe.dashboard.pages.team_deep_dive` — Team Deep Dive
3. :mod:`polymbappe.dashboard.pages.match_predictor` — Match Predictor
4. :mod:`polymbappe.dashboard.pages.market_edges` — Market Edges
5. :mod:`polymbappe.dashboard.pages.upset_watch` — Upset Watch
6. :mod:`polymbappe.dashboard.pages.agent_activity` — Agent Activity
"""

from __future__ import annotations
