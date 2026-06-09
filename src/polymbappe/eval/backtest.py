"""Leave-one-tournament-out backtest for the minimum viable model.

Protocol (spec section 7.1): for each test tournament, Dixon-Coles trains on all matches
before it; base probabilities (DC + Elo + market) are precomputed once per tournament;
the meta-learner is then fit on every *other* tournament's base probabilities (true
out-of-fold stacking) and evaluated on the held-out tournament.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
import structlog

from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs
from polymbappe.eval.metrics import multiclass_log_loss, ranked_probability_score
from polymbappe.models.meta import OUTCOMES, MetaConfig, MetaLearner

logger = structlog.get_logger(__name__)

_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}

_DC_COLS = ["dc_home", "dc_draw", "dc_away"]
_ELO_COLS = ["elo_home", "elo_draw", "elo_away"]
_MKT_COLS = ["mkt_home", "mkt_draw", "mkt_away"]


@dataclass(frozen=True, slots=True)
class Tournament:
    """A test tournament defined by competition name and date window."""

    name: str
    competition: str
    start: date
    end: date


#: Leave-one-tournament-out evaluation set (spec section 7.1). Competition strings match
#: the Kaggle international results `tournament` field.
DEFAULT_TOURNAMENTS: tuple[Tournament, ...] = (
    Tournament("WC2010", "FIFA World Cup", date(2010, 6, 11), date(2010, 7, 11)),
    Tournament("WC2014", "FIFA World Cup", date(2014, 6, 12), date(2014, 7, 13)),
    Tournament("WC2018", "FIFA World Cup", date(2018, 6, 14), date(2018, 7, 15)),
    Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18)),
    Tournament("EU2016", "UEFA Euro", date(2016, 6, 10), date(2016, 7, 10)),
    Tournament("EU2020", "UEFA Euro", date(2021, 6, 11), date(2021, 7, 11)),
    Tournament("EU2024", "UEFA Euro", date(2024, 6, 14), date(2024, 7, 14)),
    Tournament("CA2016", "Copa América", date(2016, 6, 3), date(2016, 6, 26)),
    Tournament("CA2019", "Copa América", date(2019, 6, 14), date(2019, 7, 7)),
    Tournament("CA2021", "Copa América", date(2021, 6, 13), date(2021, 7, 10)),
    Tournament("CA2024", "Copa América", date(2024, 6, 20), date(2024, 7, 14)),
)


@dataclass(slots=True)
class BacktestResult:
    """Per-tournament and aggregate metrics from a backtest run."""

    per_tournament: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    @property
    def mean_rps(self) -> float:
        values = [m["rps"] for m in self.per_tournament.values()]
        return float(np.mean(values)) if values else float("nan")

    def to_frame(self) -> pl.DataFrame:
        rows = [{"tournament": name, **metrics} for name, metrics in self.per_tournament.items()]
        return pl.DataFrame(rows)


def select_fixtures(matches: pl.DataFrame, tournament: Tournament) -> pl.DataFrame:
    """Select a tournament's matches by competition name and date window."""

    return matches.filter(
        (pl.col("competition") == tournament.competition)
        & (pl.col("date") >= tournament.start)
        & (pl.col("date") <= tournament.end)
    )


def _has_market(df: pl.DataFrame) -> bool:
    """Whether a base-prob frame carries complete (non-null) market columns."""

    if not set(_MKT_COLS).issubset(df.columns):
        return False
    return int(df.select(_MKT_COLS).null_count().sum_horizontal()[0]) == 0


def _metrics(labels: list[str], y_prob: np.ndarray) -> dict[str, float]:
    idx = np.array([_LABEL_TO_IDX[label] for label in labels])
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(idx)), idx] = 1.0
    return {
        "rps": ranked_probability_score(idx, y_prob),
        "log_loss": multiclass_log_loss(idx, y_prob),
        "brier": float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))),
        "n": float(len(idx)),
    }


def run_leave_one_tournament_out(
    matches: pl.DataFrame,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    *,
    base_config: BaseProbConfig | None = None,
    meta_config: MetaConfig | None = None,
    market_odds: pl.DataFrame | None = None,
) -> BacktestResult:
    """Run the leave-one-tournament-out MVM backtest.

    Returns per-tournament RPS / log loss / Brier and the feature set used. Requires at
    least two tournaments with fixtures present in ``matches``.
    """

    base_config = base_config or BaseProbConfig()

    per_tournament_probs: dict[str, pl.DataFrame] = {}
    for tournament in tournaments:
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            logger.warning("backtest.no_history", tournament=tournament.name)
            continue
        per_tournament_probs[tournament.name] = compute_tournament_base_probs(
            history,
            fixtures,
            tournament=tournament.name,
            config=base_config,
            market_odds=market_odds,
        )

    if len(per_tournament_probs) < 2:
        raise ValueError(
            "Leave-one-tournament-out needs >=2 tournaments with fixtures and history; "
            f"found {len(per_tournament_probs)}."
        )

    # Use the market features only if every tournament has non-null market odds.
    use_market = all(_has_market(df) for df in per_tournament_probs.values())
    feature_cols = _DC_COLS + _ELO_COLS + (_MKT_COLS if use_market else [])

    result = BacktestResult(feature_columns=feature_cols)
    names = list(per_tournament_probs)
    for held_out in names:
        train = pl.concat(
            [per_tournament_probs[n] for n in names if n != held_out], how="vertical_relaxed"
        )
        meta = MetaLearner(feature_cols, meta_config).fit(train)
        test = per_tournament_probs[held_out]
        y_prob = meta.predict_proba(test)
        result.per_tournament[held_out] = _metrics(test["label"].to_list(), y_prob)

    logger.info(
        "backtest.done",
        mean_rps=round(result.mean_rps, 4),
        features=feature_cols,
        tournaments=names,
    )
    return result


def run_walk_forward_backtest() -> None:
    """CLI entrypoint: run the LOTO backtest over stored matches and print the report."""

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    market_odds = (
        read_table(Table.MARKET_ODDS, settings)
        if table_exists(Table.MARKET_ODDS, settings)
        else None
    )
    result = run_leave_one_tournament_out(matches, market_odds=market_odds)
    print(f"Features: {', '.join(result.feature_columns)}")
    print(result.to_frame().sort("tournament"))
    print(f"Mean RPS: {result.mean_rps:.4f}")
