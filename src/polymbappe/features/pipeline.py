"""Feature pipeline orchestration.

Joins the per-team and per-match builders into the final match-level training matrix,
attaching home/away feature columns and the H/D/A label. Every builder is point-in-time,
so the assembled matrix is leakage-safe by construction; ``as_of`` additionally caps the
history used (for live "as of" snapshots).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.features.context import (
    HOSTS_2026,
    build_form_features,
    build_h2h_features,
    build_rest_features,
    build_structural_features,
)
from polymbappe.features.elo import EloConfig, build_elo_features

logger = structlog.get_logger(__name__)

#: Identity columns carried through from the matches frame.
_ID_COLUMNS = (
    "match_id",
    "date",
    "home_team",
    "away_team",
    "competition",
    "is_knockout",
    "neutral_site",
    "group",
)

Label = Literal["H", "D", "A"]


def result_label(home_goals: int, away_goals: int) -> Label:
    """Map a scoreline to its H/D/A outcome label."""

    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _join_team_table(matrix: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """Join a ``(match_id, team, date, *features)`` table onto both match sides."""

    feature_cols = [c for c in feats.columns if c not in ("match_id", "team", "date")]
    trimmed = feats.drop("date")

    home = trimmed.rename({c: f"home_{c}" for c in feature_cols})
    matrix = matrix.join(
        home, left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left"
    )
    away = trimmed.rename({c: f"away_{c}" for c in feature_cols})
    matrix = matrix.join(
        away, left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left"
    )
    return matrix


class FeaturePipeline:
    """Assembles the core (Tier 1-3) match-level feature matrix."""

    def __init__(
        self,
        hosts: frozenset[str] = HOSTS_2026,
        elo_config: EloConfig | None = None,
        form_windows: tuple[int, ...] = (5, 10),
    ) -> None:
        self.hosts = hosts
        self.elo_config = elo_config
        self.form_windows = form_windows

    def build_core_matrix(
        self,
        matches: pl.DataFrame,
        as_of_date: date | None = None,
        team_xg: pl.DataFrame | None = None,
        market_odds: pl.DataFrame | None = None,
        squad_valuations: pl.DataFrame | None = None,
        tournaments: object | None = None,
    ) -> pl.DataFrame:
        """Build the core feature matrix with the H/D/A label.

        Args:
            matches: Frame with the ``matches`` schema.
            as_of_date: Cap the history used to matches strictly before this date.
            team_xg: Optional FBref team-match xG table (enables real rolling xG).
            market_odds: Optional ``market_odds`` table; joined by ``match_id`` when given.
            squad_valuations: Optional ``squad_valuations`` table; when given, the Tier-1
                ``squad_value_ratio`` (``log(value_home / value_away)``) and the home/away
                log squad values are attached per tournament snapshot.
            tournaments: Tournament set locating each fixture's squad snapshot (defaults to
                :data:`~polymbappe.eval.backtest.DEFAULT_TOURNAMENTS`); only used when
                ``squad_valuations`` is given.

        Returns:
            One row per match: identity columns, ``home_*``/``away_*`` features, derived
            diffs, optional market probabilities, and the ``label`` target.
        """

        missing = [c for c in _ID_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches frame missing required columns: {missing}")

        matrix = matches.select(
            *_ID_COLUMNS, pl.col("home_goals"), pl.col("away_goals")
        )

        elo = build_elo_features(matches, as_of_date, self.elo_config)
        form = build_form_features(matches, as_of_date, self.form_windows)
        rest = build_rest_features(matches, as_of_date)
        for team_table in (elo, form, rest):
            matrix = _join_team_table(matrix, team_table)

        if team_xg is not None:
            from polymbappe.features.xg import build_xg_features

            matrix = _join_team_table(
                matrix, build_xg_features(matches, team_xg, as_of_date)
            )

        if squad_valuations is not None:
            from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS
            from polymbappe.features.squad import build_squad_match_features

            tours = tournaments if tournaments is not None else DEFAULT_TOURNAMENTS
            squad = build_squad_match_features(matches, squad_valuations, tours)
            if not squad.is_empty():
                matrix = _join_team_table(matrix, squad)
                matrix = matrix.with_columns(
                    (
                        pl.col("home_log_total_value") - pl.col("away_log_total_value")
                    ).alias("squad_value_ratio")
                )

        h2h = build_h2h_features(matches, as_of_date)
        structural = build_structural_features(matches, self.hosts)
        matrix = matrix.join(h2h, on="match_id", how="left")
        matrix = matrix.join(structural, on="match_id", how="left")

        matrix = matrix.with_columns(
            (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("elo_diff")
        )

        if market_odds is not None:
            odds = market_odds.select(
                "match_id", "home_win_prob", "draw_prob", "away_win_prob"
            )
            matrix = matrix.join(odds, on="match_id", how="left")

        matrix = matrix.with_columns(
            pl.when(pl.col("home_goals") > pl.col("away_goals"))
            .then(pl.lit("H"))
            .when(pl.col("home_goals") < pl.col("away_goals"))
            .then(pl.lit("A"))
            .otherwise(pl.lit("D"))
            .alias("label")
        )
        return matrix


def build_contextual_matrix(
    matches: pl.DataFrame,
    as_of_date: date | None = None,
    team_xg: pl.DataFrame | None = None,
    team_ppda: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Assemble the match-level contextual feature matrix (spec 2.2, Tier A).

    Joins the team-date contextual builders (xG overperformance, PPDA) onto both match
    sides and derives the ``ppda_diff`` match-pair feature. Richer inputs (squad cohesion,
    manager pedigree, fatigue, draw pressure) are joined here when their source tables are
    materialized; absent inputs are simply skipped so the matrix always builds.

    Returns one row per match with ``home_*``/``away_*`` contextual columns, ``ppda_diff``,
    and the H/D/A ``label``.
    """

    from polymbappe.context.ppda import build_ppda_features
    from polymbappe.context.sentiment import build_xg_overperformance

    matrix = matches.select(*_ID_COLUMNS, pl.col("home_goals"), pl.col("away_goals"))
    overperf = build_xg_overperformance(matches, team_xg, as_of_date)
    ppda = build_ppda_features(matches, team_ppda, as_of_date).select(
        ["match_id", "team", "date", "ppda"]
    )
    for team_table in (overperf, ppda):
        matrix = _join_team_table(matrix, team_table)

    matrix = matrix.with_columns(
        (pl.col("home_ppda") - pl.col("away_ppda")).alias("ppda_diff")
    )
    matrix = matrix.with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("label")
    )
    return matrix


def build_feature_matrix(as_of: date | None = None, contextual: bool = False) -> None:
    """CLI entrypoint: build the core (or contextual) feature matrix from stored matches.

    Reads the ``matches`` table, assembles the requested matrix (joining market odds /
    team xG when available), and writes it to ``data/processed/{core,contextual}_features.parquet``.
    """

    from polymbappe.data.store import read_table, table_exists, write_parquet
    from polymbappe.data.tables import Table

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    team_xg = (
        read_table(Table.TEAM_XG, settings) if table_exists(Table.TEAM_XG, settings) else None
    )

    if contextual:
        matrix = build_contextual_matrix(matches, as_of_date=as_of, team_xg=team_xg)
        out_path = settings.processed_data_dir / "contextual_features.parquet"
    else:
        market_odds = (
            read_table(Table.MARKET_ODDS, settings)
            if table_exists(Table.MARKET_ODDS, settings)
            else None
        )
        squad_valuations = (
            read_table(Table.SQUAD_VALUATIONS, settings)
            if table_exists(Table.SQUAD_VALUATIONS, settings)
            else None
        )
        matrix = FeaturePipeline().build_core_matrix(
            matches,
            as_of_date=as_of,
            team_xg=team_xg,
            market_odds=market_odds,
            squad_valuations=squad_valuations,
        )
        out_path = settings.processed_data_dir / "core_features.parquet"

    write_parquet(matrix, out_path)
    logger.info("features.built", rows=matrix.height, cols=len(matrix.columns), path=str(out_path))
