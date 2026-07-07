"""Streamlit dashboard pages.

Each page module exposes ``render(settings: Settings) -> None`` which lazily imports
``streamlit``, reads artifacts through :mod:`polymbappe.dashboard.data`, and renders
its page.

1. :mod:`polymbappe.dashboard.pages.overview` — Tournament Overview
2. :mod:`polymbappe.dashboard.pages.match_predictor` — Match Predictor
3. :mod:`polymbappe.dashboard.pages.predictions_vs_actuals` — Predictions vs Actuals
4. :mod:`polymbappe.dashboard.pages.team_deep_dive` — Team Deep Dive
5. :mod:`polymbappe.dashboard.pages.upset_watch` — Upset Watch
6. :mod:`polymbappe.dashboard.pages.model_showcase` — Model Showcase
"""

from __future__ import annotations
