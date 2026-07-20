"""Tournament probability trajectory via honest replay, plus champion-market P&L.

Git history holds only a handful of ``stage_probabilities`` snapshots, so the
"how did each team's championship probability evolve?" story is *reconstructed*:
for every match day of the tournament the simulation is re-run conditioned on the
information set of that day — matches up to and including it — with the
Dixon-Coles model refit on exactly that history (no hindsight). The resulting
``champion_trajectory.parquet`` powers the retrospective chart and, joined with
Polymarket's championship price history, a Kelly-staked P&L backtest of trading
the champion market against the model (``market_pnl.parquet``).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import structlog

from polymbappe.simulate.tournament import WC2026_START, StrengthModel, simulate_tournament

logger = structlog.get_logger(__name__)

TRAJECTORY_FILE = "champion_trajectory.parquet"
MARKET_PNL_FILE = "market_pnl.parquet"

#: Stage columns carried in the trajectory frame (narrowest last).
TRAJECTORY_STAGES: tuple[str, ...] = ("SF", "FINAL", "champion")

TRAJECTORY_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "team": pl.Utf8,
    **{s: pl.Float64 for s in TRAJECTORY_STAGES},
}


def replay_dates(matches: pl.DataFrame) -> list[date]:
    """Information-cutoff dates for the replay: eve of the opener, then every match day.

    Each returned date D is an end-of-day cutoff (condition on ``date <= D``); the
    pre-tournament point (the day before :data:`WC2026_START`) yields the pure
    pre-tournament forecast.
    """

    wc = matches.filter(
        (pl.col("competition") == "FIFA World Cup") & (pl.col("date") >= WC2026_START)
    )
    days = sorted(set(wc["date"].to_list()))
    return [WC2026_START - timedelta(days=1), *days]


def compute_champion_trajectory(
    matches: pl.DataFrame,
    schedule: pl.DataFrame,
    structure: object,
    *,
    n_sims: int = 10_000,
    refit: bool = True,
    seed: int = 0,
    winner_overrides: dict[int, str] | None = None,
    fallback_model: StrengthModel | None = None,
) -> pl.DataFrame:
    """Per-team stage probabilities at every replay cutoff (long frame).

    For each cutoff date the tournament is simulated exactly like the production
    path — real bracket, played results locked, winner overrides applied — but on
    the *historical* information set. ``refit=True`` (the honest default) refits
    Dixon-Coles on each cutoff's history; ``refit=False`` reuses
    ``fallback_model`` for every date (fast, but early dates inherit knowledge
    the model only earned later — fine for a visual, not for a claim).

    Returns columns :data:`TRAJECTORY_SCHEMA`: ``date`` is the cutoff, and each
    stage column is that day's probability of the team reaching the stage.
    """

    from polymbappe.eval.base_probs import BaseProbConfig
    from polymbappe.features.context import HOSTS_2026
    from polymbappe.models.train import _all_history_dixon_coles
    from polymbappe.simulate.real_bracket import attach_played_results, build_real_bracket
    from polymbappe.simulate.tournament import build_played_group_results

    if not refit and fallback_model is None:
        raise ValueError("refit=False requires a fallback_model")

    dates = replay_dates(matches)
    frames: list[pl.DataFrame] = []
    for i, cutoff in enumerate(dates):
        history = matches.filter(pl.col("date") <= cutoff)
        if refit:
            dc = _all_history_dixon_coles(history, BaseProbConfig())
            model = StrengthModel.from_dixon_coles(dc, hosts=HOSTS_2026)
        else:
            model = fallback_model  # type: ignore[assignment]

        played = build_played_group_results(history, structure)  # type: ignore[arg-type]
        bracket = build_real_bracket(schedule)
        if bracket is not None:
            attach_played_results(bracket, history, winner_overrides)

        result = simulate_tournament(
            structure,  # type: ignore[arg-type]
            model,
            n_sims=n_sims,
            rng=np.random.default_rng(seed + i),
            played_results=played or None,
            real_bracket=bracket,
        )
        snap = (
            result.stage_probabilities()
            .select("team", *TRAJECTORY_STAGES)
            .with_columns(pl.lit(cutoff).alias("date"))
        )
        frames.append(snap.select(list(TRAJECTORY_SCHEMA)))
        logger.info(
            "trajectory.point",
            cutoff=str(cutoff),
            point=f"{i + 1}/{len(dates)}",
            refit=refit,
            top=snap.sort("champion", descending=True).row(0, named=True)["team"],
        )

    return pl.concat(frames) if frames else pl.DataFrame(schema=TRAJECTORY_SCHEMA)


# ---------------------------------------------------------------------------
# Champion-market P&L
# ---------------------------------------------------------------------------

PNL_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "team": pl.Utf8,
    "model_prob": pl.Float64,
    "market_price": pl.Float64,
    "edge": pl.Float64,
    "stake": pl.Float64,
    "payout": pl.Float64,
    "profit": pl.Float64,
}


def compute_champion_market_pnl(
    trajectory: pl.DataFrame,
    market_history: pl.DataFrame,
    champion: str,
    *,
    edge_threshold: float = 0.03,
    kelly_scale: float = 0.25,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Kelly-staked P&L of trading the champion market against the model.

    For every replay date x team where the model's championship probability
    exceeds that day's market price by ``edge_threshold``, a fractional-Kelly
    stake (``kelly_scale`` x full Kelly, of a unit bankroll per bet) buys Yes at
    the market price; every position settles at resolution (1 if ``team`` is the
    ``champion``, else 0). Only long-Yes positions are taken — shorting a
    binary at price q is equivalent to buying the complement, which the
    per-team loop already covers economically and keeps the accounting simple.

    ``market_history`` needs ``[date, team, price]`` (daily close). Returns the
    per-bet frame (:data:`PNL_SCHEMA`) and a summary dict with ``n_bets``,
    ``total_staked``, ``total_profit``, ``roi``.
    """

    from polymbappe.eval.market import kelly_fraction

    if trajectory.is_empty() or market_history.is_empty():
        return pl.DataFrame(schema=PNL_SCHEMA), {
            "n_bets": 0.0, "total_staked": 0.0, "total_profit": 0.0, "roi": 0.0,
        }

    joined = trajectory.select("date", "team", pl.col("champion").alias("model_prob")).join(
        market_history.select("date", "team", pl.col("price").alias("market_price")),
        on=["date", "team"],
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for r in joined.sort(["date", "team"]).iter_rows(named=True):
        p, q = float(r["model_prob"]), float(r["market_price"])
        edge = p - q
        if edge <= edge_threshold or not 0.0 < q < 1.0:
            continue
        stake = kelly_scale * kelly_fraction(p, q)
        if stake <= 0.0:
            continue
        won = r["team"] == champion
        # Buying Yes at q: stake buys stake/q shares, each paying 1 if resolved Yes.
        payout = (stake / q) if won else 0.0
        rows.append(
            {
                "date": r["date"], "team": r["team"], "model_prob": p,
                "market_price": q, "edge": edge, "stake": stake,
                "payout": payout, "profit": payout - stake,
            }
        )

    pnl = pl.DataFrame(rows, schema=PNL_SCHEMA)
    staked = float(pnl["stake"].sum()) if pnl.height else 0.0
    profit = float(pnl["profit"].sum()) if pnl.height else 0.0
    summary = {
        "n_bets": float(pnl.height),
        "total_staked": staked,
        "total_profit": profit,
        "roi": (profit / staked) if staked > 0 else 0.0,
    }
    return pnl, summary


def fetch_champion_market_history(slug: str = "world-cup-winner") -> pl.DataFrame:
    """Daily champion-market price history per team from Polymarket (``[date, team, price]``).

    Resolves the event's per-team Yes tokens, pulls each token's CLOB price
    history at daily fidelity, and keeps the last price per calendar day. Any
    network/shape failure degrades to a typed empty frame — the retrospective
    then renders an "unavailable" note instead of failing.
    """

    from polymbappe.polymarket.adapter import (
        fetch_polymarket_event,
        fetch_polymarket_price_history,
        parse_team_yes_tokens,
    )

    empty = pl.DataFrame(schema={"date": pl.Date, "team": pl.Utf8, "price": pl.Float64})
    try:
        event = fetch_polymarket_event(slug)
        tokens = parse_team_yes_tokens(event)
    except Exception as exc:  # noqa: BLE001 - market data must never break the replay
        logger.warning("trajectory.market_event_failed", slug=slug, error=str(exc))
        return empty
    if tokens.is_empty():
        logger.warning("trajectory.market_no_tokens", slug=slug)
        return empty

    frames: list[pl.DataFrame] = []
    for r in tokens.iter_rows(named=True):
        history = fetch_polymarket_price_history(str(r["token_id"]))
        if history.is_empty():
            continue
        frames.append(
            history.with_columns(
                pl.col("timestamp").dt.date().alias("date"),
                pl.lit(str(r["team"])).alias("team"),
            )
            .sort("timestamp")
            .group_by("date", "team", maintain_order=True)
            .agg(pl.col("price").last())
            .select("date", "team", "price")
        )
    if not frames:
        logger.warning("trajectory.market_no_history", slug=slug)
        return empty
    return pl.concat(frames)


def run_trajectory(
    n_sims: int = 10_000,
    refit: bool = True,
    seed: int | None = None,
    market: bool = False,
) -> None:
    """CLI entrypoint: compute the replay trajectory (and optionally market P&L).

    Writes ``data/outputs/champion_trajectory.parquet`` and, with ``market``,
    ``data/outputs/market_pnl.parquet``. The market leg needs a decided
    tournament (a team with champion probability 1.0 in the final replay point)
    to settle bets; before that it logs and skips settlement.
    """

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table
    from polymbappe.features.context import HOSTS_2026
    from polymbappe.models.train import load_artifact
    from polymbappe.simulate.real_bracket import load_ko_winner_overrides
    from polymbappe.simulate.structure import load_structure_2026

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    schedule = (
        read_table(Table.SCHEDULE, settings)
        if table_exists(Table.SCHEDULE, settings)
        else pl.DataFrame()
    )
    structure = load_structure_2026(settings)
    overrides = load_ko_winner_overrides(settings)

    fallback: StrengthModel | None = None
    if not refit:
        dc = load_artifact("dixon_coles", settings)
        fallback = StrengthModel.from_dixon_coles(dc, hosts=HOSTS_2026)

    trajectory = compute_champion_trajectory(
        matches, schedule, structure,
        n_sims=n_sims, refit=refit,
        seed=settings.random_seed if seed is None else seed,
        winner_overrides=overrides, fallback_model=fallback,
    )
    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_data_dir / TRAJECTORY_FILE
    trajectory.write_parquet(out)
    logger.info("trajectory.written", path=str(out), points=trajectory["date"].n_unique())

    if not market:
        return
    last = trajectory.filter(pl.col("date") == trajectory["date"].max())
    decided = last.filter(pl.col("champion") >= 0.999)
    if decided.is_empty():
        logger.warning(
            "trajectory.market_skipped",
            reason="tournament undecided; settle bets after the final is ingested",
        )
        return
    champion = str(decided.row(0, named=True)["team"])
    history = fetch_champion_market_history()
    pnl, summary = compute_champion_market_pnl(trajectory, history, champion)
    pnl.write_parquet(settings.outputs_data_dir / MARKET_PNL_FILE)
    logger.info("trajectory.market_pnl", champion=champion, **summary)
