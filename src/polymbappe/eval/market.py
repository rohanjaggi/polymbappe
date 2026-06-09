"""Market comparison utilities — edge detection against market-implied probabilities."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from polymbappe.models.meta import OUTCOMES


class Market(BaseModel):
    """Market probability snapshot for edge analysis."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    outcome: str
    market_price: float
    model_prob: float
    edge_bps: float
    kelly_fraction: float


def kelly_fraction(model_prob: float, market_price: float) -> float:
    """Full-Kelly stake fraction for a binary bet at decimal-implied ``market_price``.

    ``market_price`` is the market-implied probability (0-1); the fair decimal odds are
    ``1 / market_price``. Returns 0 when there is no positive edge.
    """

    if market_price <= 0.0 or market_price >= 1.0:
        return 0.0
    b = (1.0 / market_price) - 1.0  # net decimal odds
    edge = (b * model_prob - (1.0 - model_prob)) / b
    return max(edge, 0.0)


def compute_edges(
    model_probs: pl.DataFrame,
    market_probs: pl.DataFrame,
    *,
    threshold: float = 0.05,
    model_cols: tuple[str, str, str] = ("model_home", "model_draw", "model_away"),
    market_cols: tuple[str, str, str] = ("home_win_prob", "draw_prob", "away_win_prob"),
    id_col: str = "match_id",
) -> pl.DataFrame:
    """Flag outcomes where the model diverges from the market by more than ``threshold``.

    Returns one row per (match, outcome) edge with the model probability, market price,
    signed edge in basis points, and the full-Kelly stake fraction, sorted by absolute
    edge magnitude.
    """

    joined = model_probs.join(market_probs, on=id_col, how="inner")
    rows: list[dict[str, object]] = []
    for record in joined.iter_rows(named=True):
        for outcome, m_col, k_col in zip(OUTCOMES, model_cols, market_cols, strict=True):
            model_p = record.get(m_col)
            market_p = record.get(k_col)
            if model_p is None or market_p is None:
                continue
            edge = float(model_p) - float(market_p)
            if abs(edge) <= threshold:
                continue
            rows.append(
                {
                    "match_id": record[id_col],
                    "outcome": outcome,
                    "model_prob": float(model_p),
                    "market_prob": float(market_p),
                    "edge": edge,
                    "edge_bps": edge * 10_000.0,
                    "kelly_fraction": kelly_fraction(float(model_p), float(market_p)),
                }
            )

    schema = {
        "match_id": pl.Utf8,
        "outcome": pl.Utf8,
        "model_prob": pl.Float64,
        "market_prob": pl.Float64,
        "edge": pl.Float64,
        "edge_bps": pl.Float64,
        "kelly_fraction": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort(pl.col("edge").abs(), descending=True)


def compute_credible_edges(
    model_probs: pl.DataFrame,
    market_probs: pl.DataFrame,
    *,
    threshold: float = 0.05,
    model_cols: tuple[str, str, str] = ("model_home", "model_draw", "model_away"),
    low_cols: tuple[str, str, str] = ("ci_home_low", "ci_draw_low", "ci_away_low"),
    high_cols: tuple[str, str, str] = ("ci_home_high", "ci_draw_high", "ci_away_high"),
    market_cols: tuple[str, str, str] = ("home_win_prob", "draw_prob", "away_win_prob"),
    id_col: str = "match_id",
) -> pl.DataFrame:
    """Flag market-blind edges that also clear the Bayesian credible-interval test.

    A genuine edge (spec 3.6) requires both: the edge-pipeline model and the market
    diverge by more than ``threshold`` *and* the model's credible interval for that
    outcome does not contain the market's implied probability. ``model_probs`` must carry
    the edge-pipeline point probabilities plus per-outcome credible-interval bounds.

    Returns one row per qualifying (match, outcome) edge sorted by absolute edge.
    """

    joined = model_probs.join(market_probs, on=id_col, how="inner")
    rows: list[dict[str, object]] = []
    for record in joined.iter_rows(named=True):
        for outcome, m_col, lo_col, hi_col, k_col in zip(
            OUTCOMES, model_cols, low_cols, high_cols, market_cols, strict=True
        ):
            model_p = record.get(m_col)
            market_p = record.get(k_col)
            lo = record.get(lo_col)
            hi = record.get(hi_col)
            if model_p is None or market_p is None or lo is None or hi is None:
                continue
            edge = float(model_p) - float(market_p)
            ci_excludes_market = not (float(lo) <= float(market_p) <= float(hi))
            if abs(edge) <= threshold or not ci_excludes_market:
                continue
            rows.append(
                {
                    "match_id": record[id_col],
                    "outcome": outcome,
                    "model_prob": float(model_p),
                    "market_prob": float(market_p),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "edge": edge,
                    "edge_bps": edge * 10_000.0,
                    "kelly_fraction": kelly_fraction(float(model_p), float(market_p)),
                }
            )

    schema = {
        "match_id": pl.Utf8,
        "outcome": pl.Utf8,
        "model_prob": pl.Float64,
        "market_prob": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "edge": pl.Float64,
        "edge_bps": pl.Float64,
        "kelly_fraction": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort(pl.col("edge").abs(), descending=True)


def compare_model_to_market() -> None:
    """CLI entrypoint: load stored predictions + market odds and print the edge table."""

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    predictions_path = settings.outputs_data_dir / "match_predictions.parquet"
    if not predictions_path.exists() or not table_exists(Table.MARKET_ODDS, settings):
        raise FileNotFoundError(
            "Edge detection needs match predictions (run `polymbappe simulate`/`report`) "
            "and an ingested market_odds table."
        )

    model_probs = pl.read_parquet(predictions_path)
    market_probs = read_table(Table.MARKET_ODDS, settings)
    edges = compute_edges(model_probs, market_probs)
    print(edges)
