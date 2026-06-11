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


def compute_outright_edges(
    model_probs: pl.DataFrame,
    market_probs: pl.DataFrame,
    *,
    threshold: float = 0.03,
    model_col: str = "model_prob",
    market_col: str = "market_prob",
    id_col: str = "team",
) -> pl.DataFrame:
    """Flag team-level (outright/futures) edges where model and market diverge.

    For single-outcome futures (champion, reach-stage, group winner) where each row is a
    team's Yes probability. Returns one row per team whose ``|model - market|`` exceeds
    ``threshold``, with the signed edge and full-Kelly stake, sorted by absolute edge.
    """

    joined = model_probs.join(market_probs, on=id_col, how="inner")
    rows: list[dict[str, object]] = []
    for record in joined.iter_rows(named=True):
        model_p, market_p = record.get(model_col), record.get(market_col)
        if model_p is None or market_p is None:
            continue
        edge = float(model_p) - float(market_p)
        if abs(edge) <= threshold:
            continue
        rows.append(
            {
                "team": record[id_col],
                "model_prob": float(model_p),
                "market_prob": float(market_p),
                "edge": edge,
                "edge_bps": edge * 10_000.0,
                "kelly_fraction": kelly_fraction(float(model_p), float(market_p)),
            }
        )
    schema = {
        "team": pl.Utf8, "model_prob": pl.Float64, "market_prob": pl.Float64,
        "edge": pl.Float64, "edge_bps": pl.Float64, "kelly_fraction": pl.Float64,
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
    """CLI entrypoint: print the model-vs-market edge table.

    Prefers the precomputed ``edges.parquet`` written by ``simulate``. If that is absent,
    recomputes from ``match_predictions.parquet`` + the ``market_odds`` table when both
    exist, naming precisely which prerequisite is missing otherwise.
    """

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    edges_path = settings.outputs_data_dir / "edges.parquet"
    predictions_path = settings.outputs_data_dir / "match_predictions.parquet"

    # Preferred: the edges artifact already produced by `simulate`.
    if edges_path.exists():
        edges = pl.read_parquet(edges_path)
        if edges.is_empty():
            print(
                "No market edges found. Either no market_odds were ingested, or none of "
                "the fixtures' match_ids joined the odds (check `polymbappe ingest` odds "
                "sources and configs/team_aliases.yaml). Re-run `polymbappe simulate` after "
                "ingesting odds."
            )
        else:
            print(edges)
        return

    # Fallback: recompute, but say exactly what is missing.
    missing: list[str] = []
    if not predictions_path.exists():
        missing.append("match_predictions.parquet (run `polymbappe simulate`)")
    if not table_exists(Table.MARKET_ODDS, settings):
        missing.append("market_odds table (run `polymbappe ingest` with odds sources)")
    if missing:
        raise FileNotFoundError("Edge detection is missing: " + "; ".join(missing))

    model_probs = pl.read_parquet(predictions_path)
    market_probs = read_table(Table.MARKET_ODDS, settings)
    print(compute_edges(model_probs, market_probs))


def compare_outright_to_market(slug: str = "world-cup-winner") -> None:
    """CLI entrypoint: futures edges (champion / reach-stage) vs a Polymarket market.

    Polymarket lists no per-match H/D/A markets until fixtures are scheduled; the
    tradeable pre-tournament markets are futures (``world-cup-winner``,
    ``world-cup-nation-to-reach-*``, ``world-cup-team-to-advance-to-knockout-stages``).
    This loads the matching simulation probabilities, fetches the market, and prints
    team-level edges, writing ``data/outputs/futures_edges.parquet``.
    """

    from polymbappe.config import Settings
    from polymbappe.polymarket.adapter import (
        WORLD_CUP_FUTURES,
        fetch_polymarket_event,
        parse_team_yes_prices,
    )

    spec = WORLD_CUP_FUTURES.get(slug)
    if spec is None:
        raise ValueError(
            f"Unknown futures slug {slug!r}. Known: {', '.join(sorted(WORLD_CUP_FUTURES))}."
        )

    settings = Settings()
    output_path = settings.outputs_data_dir / f"{spec['output']}.parquet"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Need {spec['output']}.parquet (run `polymbappe simulate`) for futures edges."
        )

    model = pl.read_parquet(output_path).select(
        "team", pl.col(spec["column"]).alias("model_prob")
    )
    event = fetch_polymarket_event(slug)
    market = parse_team_yes_prices(event, normalize=spec["normalize"])
    edges = compute_outright_edges(model, market)

    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(settings.outputs_data_dir / "futures_edges.parquet")
    if edges.is_empty():
        print(f"No outright edges for {slug} (model and market agree within threshold).")
    else:
        print(edges)
