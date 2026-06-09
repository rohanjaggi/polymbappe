"""Page 5 — Upset Watch (spec section 6.1).

Surfaces underdogs whose advancement probabilities look high relative to their Elo
gap (spec 4.2 / 6.1, page 5). ``streamlit`` is imported lazily.
"""

from __future__ import annotations

from polymbappe.config import Settings
from polymbappe.dashboard import data


def render(settings: Settings) -> None:
    """Render the Upset Watch page (spec 6.1, page 5)."""

    import streamlit as st

    st.header("Upset Watch")

    stage_df = data.load_stage_probabilities(settings)
    if stage_df.is_empty():
        st.info("No simulation results yet. Run `polymbappe simulate` to populate the dashboard.")
        return

    st.caption(
        "Teams the Monte Carlo engine gives an unusually strong run to. With Elo data "
        "available, ranking weights advancement by Elo deficit vs. the field (spec 4.2)."
    )

    min_gap = st.slider("Minimum Elo deficit vs. field max", 0.0, 600.0, 300.0, step=50.0)
    elo = _load_elo(settings)

    candidates = data.upset_candidates(stage_df, elo=elo or None, min_elo_gap=min_gap)
    if candidates.is_empty():
        st.warning("No upset candidates at this Elo-deficit threshold.")
        return

    st.dataframe(candidates.to_pandas(), use_container_width=True)


def _load_elo(settings: Settings) -> dict[str, float]:
    """Best-effort load of latest per-team Elo ratings from the processed store.

    Returns an empty mapping when the Elo table is absent, in which case Upset Watch
    falls back to ranking purely by advancement probability.
    """

    try:
        import polars as pl

        from polymbappe.data.store import read_table, table_exists
        from polymbappe.data.tables import Table

        if not table_exists(Table.ELO_SNAPSHOTS, settings):
            return {}
        elo_df = read_table(Table.ELO_SNAPSHOTS, settings)
        if elo_df.is_empty():
            return {}
        latest = (
            elo_df.sort("date", descending=True)
            .group_by("team")
            .agg(pl.col("rating").first())
        )
        return {row["team"]: float(row["rating"]) for row in latest.iter_rows(named=True)}
    except Exception:  # pragma: no cover - resilience against missing/odd data
        return {}
