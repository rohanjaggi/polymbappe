"""Streamlit dashboard pages.

Each page module exposes ``render(settings: Settings) -> None`` which lazily imports
``streamlit``, reads artifacts through :mod:`polymbappe.dashboard.data`, and renders
its page. Nav order and URL paths live in ``dashboard/app.py`` (``PAGES``); in that
order:

1. :mod:`polymbappe.dashboard.pages.overview` — Tournament Overview (``/overview``)
2. :mod:`polymbappe.dashboard.pages.retrospective` — Tournament Retrospective (``/retrospective``)
3. :mod:`polymbappe.dashboard.pages.predictions_vs_actuals` — Predictions vs Actuals (``/results``)
4. :mod:`polymbappe.dashboard.pages.match_predictor` — Match Predictor (``/matches``)
5. :mod:`polymbappe.dashboard.pages.team_deep_dive` — Team Deep Dive (``/teams``)
6. :mod:`polymbappe.dashboard.pages.upset_watch` — Upset Watch (``/upsets``)
7. :mod:`polymbappe.dashboard.pages.model_showcase` — Model Showcase (``/model``)
"""

from __future__ import annotations
