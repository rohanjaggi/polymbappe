"""Squad cohesion & chemistry features (spec 2.2 Group B).

Two backtestable signals derived from the same Transfermarkt squad scrape used for squad
value:

* **Club cluster index** — for each club contributing ``n`` called-up players,
  ``n*(n-1)/2`` (the number of within-club pairs), summed across clubs. Captures shared
  club chemistry (e.g. a national team built around one or two club cores).
* **Median squad age** — median age of the expected XI, a simple experience-vs-athleticism
  proxy (international peak ~27-30).

Squad valuations are per-tournament snapshots, so leakage is controlled by selecting the
snapshot for the tournament being predicted (the caller passes the right slice).
"""

from __future__ import annotations

import polars as pl


def club_cluster_index(club_counts: dict[str, int]) -> int:
    """Sum of within-club pair counts ``n*(n-1)/2`` across clubs."""

    return sum(n * (n - 1) // 2 for n in club_counts.values() if n > 0)


def build_cohesion_features(squads: pl.DataFrame) -> pl.DataFrame:
    """Per-team cohesion features from a squad-list frame.

    Args:
        squads: Frame with columns ``[team, tournament, player, club, age]``. One row per
            called-up player. ``club`` may be null (counts as its own singleton, adding no
            pairs); ``age`` may be null (excluded from the median).

    Returns:
        Frame keyed by ``(team, tournament)`` with ``[team, tournament,
        club_cluster_index, median_age, player_count]``.
    """

    required = {"team", "tournament", "player", "club", "age"}
    missing = required - set(squads.columns)
    if missing:
        raise ValueError(f"squads frame missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (team, tournament), group in squads.group_by(["team", "tournament"]):
        clubs = (
            group.filter(pl.col("club").is_not_null())
            .group_by("club")
            .agg(pl.len().alias("n"))
        )
        counts = {row["club"]: int(row["n"]) for row in clubs.iter_rows(named=True)}
        ages = group["age"].drop_nulls()
        rows.append(
            {
                "team": team,
                "tournament": tournament,
                "club_cluster_index": club_cluster_index(counts),
                "median_age": float(ages.median()) if ages.len() > 0 else None,
                "player_count": int(group.height),
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "team": pl.Utf8,
            "tournament": pl.Utf8,
            "club_cluster_index": pl.Int64,
            "median_age": pl.Float64,
            "player_count": pl.Int64,
        },
    )
