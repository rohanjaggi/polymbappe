"""Shared dashboard UI components (spec section 6).

Chart builders live in :mod:`polymbappe.dashboard.components.charts`. Each builder
takes a Polars DataFrame and returns a Plotly figure, lazily importing ``plotly``
so the package imports without the optional ``dashboard`` extra installed.
"""

from __future__ import annotations
