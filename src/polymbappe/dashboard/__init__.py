"""Streamlit dashboard package (spec section 6).

Surfaces the forecasting engine's parquet outputs (spec section 11) through a
six-page Streamlit app. ``streamlit`` and ``plotly`` are optional dependencies
(spec section 13, ``dashboard`` extra) and are imported lazily inside functions so
this package imports cleanly without them.

The pure, testable data-access layer lives in :mod:`polymbappe.dashboard.data`;
chart builders in :mod:`polymbappe.dashboard.components.charts`; page renderers in
:mod:`polymbappe.dashboard.pages`; the entry point in
:mod:`polymbappe.dashboard.app`.
"""

from __future__ import annotations
