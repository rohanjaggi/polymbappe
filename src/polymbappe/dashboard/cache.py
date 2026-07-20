"""Streamlit caching shims over the pure data layer.

:mod:`polymbappe.dashboard.data` is deliberately free of any ``streamlit``
import so it stays unit-testable in isolation. This module wraps its parquet
loaders with ``st.cache_data`` at app startup, so a page interaction re-reads
each artifact at most once per TTL instead of once per widget event.

Idempotent across Streamlit reruns: wrapping happens once per server process.
"""

from __future__ import annotations

#: Data-layer functions that read artifacts from disk. Everything downstream of
#: these is cheap in-memory Polars work, so caching the reads is sufficient.
_LOADERS: tuple[str, ...] = (
    "load_stage_probabilities",
    "load_group_probabilities",
    "load_match_predictions",
    "load_edges",
    "load_agent_changelog",
    "load_knockout_predictions",
    "load_knockout_bracket",
    "load_match_xg",
    "load_recorded_results",
    "load_schedule",
    "load_autotune_leaderboard",
    "load_champion_trajectory",
    "load_market_pnl",
)

_installed = False


def install(ttl: int = 300) -> None:
    """Wrap each data-layer loader with ``st.cache_data(ttl=ttl)`` in place.

    Pages keep calling ``data.load_*`` unchanged; the module attribute is
    swapped for the cached wrapper. ``Settings`` is hashed by its JSON dump so
    two equivalent instances share cache entries.
    """

    global _installed
    if _installed:
        return

    import streamlit as st

    from polymbappe.config import Settings
    from polymbappe.dashboard import data

    hash_funcs = {Settings: lambda s: s.model_dump_json()}
    for name in _LOADERS:
        loader = getattr(data, name)
        setattr(
            data,
            name,
            st.cache_data(ttl=ttl, show_spinner=False, hash_funcs=hash_funcs)(loader),
        )
    _installed = True
