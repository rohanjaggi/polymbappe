"""Player-importance tiers for the live agent (not model features).

Derived from the ``player_attributes`` table (EA FC / FM overall ratings). Unlike the
other ``features`` modules — which feed the prediction model — these tiers are consumed
only by the LangGraph agent's Assess node to decide *which player's* injury/suspension
news is material (unified spec, "Player attribute data strategy"). Tier 1 = a team's most
important players, tier 2 = important squad members, tier 3 = the rest; the agent's
``min_tier`` then filters news about tier-3 players out.
"""

from __future__ import annotations

import polars as pl

#: Default per-team tier sizes: the top ``TIER1_SIZE`` players by overall rating are tier 1,
#: the next ``TIER2_SIZE`` are tier 2, everyone else is tier 3.
TIER1_SIZE = 3
TIER2_SIZE = 8


def build_player_tiers(
    attributes: pl.DataFrame,
    *,
    tier1_size: int = TIER1_SIZE,
    tier2_size: int = TIER2_SIZE,
) -> pl.DataFrame:
    """Rank each national team's players by overall rating into importance tiers.

    Args:
        attributes: Frame with columns ``[team, player, overall]`` (the
            ``player_attributes`` table). A player appearing multiple times for the same
            team (e.g. across FIFA editions) is collapsed to their highest ``overall``.
        tier1_size: How many top-rated players per team are tier 1 (most important).
        tier2_size: How many of the next-rated players per team are tier 2.

    Returns:
        Frame ``[team, player, overall, tier]`` with one row per ``(team, player)``, where
        ``tier`` is 1/2/3. Empty input yields an empty, correctly-typed frame.
    """

    if attributes.height == 0:
        return pl.DataFrame(
            schema={"team": pl.Utf8, "player": pl.Utf8, "overall": pl.Int64, "tier": pl.Int64}
        )

    # Collapse duplicate (team, player) rows (multiple editions) to the best rating, then
    # rank within each team. ``method="ordinal"`` breaks ties deterministically so the
    # tier-1/2 boundaries are stable.
    deduped = attributes.group_by(["team", "player"]).agg(pl.col("overall").max())
    ranked = deduped.with_columns(
        pl.col("overall").rank(method="ordinal", descending=True).over("team").alias("_rank")
    )
    return (
        ranked.with_columns(
            pl.when(pl.col("_rank") <= tier1_size)
            .then(1)
            .when(pl.col("_rank") <= tier1_size + tier2_size)
            .then(2)
            .otherwise(3)
            .cast(pl.Int64)
            .alias("tier")
        )
        .drop("_rank")
        .select("team", "player", "overall", "tier")
        .sort(["team", "tier", "overall"], descending=[False, False, True])
    )


def player_tier_map(
    attributes: pl.DataFrame,
    *,
    tier1_size: int = TIER1_SIZE,
    tier2_size: int = TIER2_SIZE,
) -> dict[str, int]:
    """Flatten :func:`build_player_tiers` into the agent's ``player_tiers`` dict.

    The agent matches news text against a flat ``{player_name: tier}`` map
    (:class:`~polymbappe.agent.nodes.AgentConfig`), so a player who plays for one national
    team is keyed by name. When the same name appears across teams (rare for full names),
    the **most important** tier wins so the agent never under-weights a key player.
    """

    tiers = build_player_tiers(attributes, tier1_size=tier1_size, tier2_size=tier2_size)
    out: dict[str, int] = {}
    for row in tiers.iter_rows(named=True):
        player, tier = row["player"], int(row["tier"])
        existing = out.get(player)
        if existing is None or tier < existing:
            out[player] = tier
    return out
