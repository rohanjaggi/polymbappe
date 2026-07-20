"""Full-tournament retrospective document (``docs/results.md``).

Assembles the completed-tournament story from the same helpers the dashboard's
Tournament Retrospective page uses: headline scorecard, per-round accuracy, the
honest replay trajectory, the upset gallery, the bookmaker head-to-head, and the
champion-market P&L. The markdown builder is a pure frames-in/string-out function
(mirroring :mod:`polymbappe.eval.report`) so it is unit-testable without IO.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from polymbappe.config import Settings

logger = structlog.get_logger(__name__)

_ROUND_LABELS = {
    "Group": "Group stage", "R32": "Round of 32", "R16": "Round of 16",
    "QF": "Quarter-finals", "SF": "Semi-finals", "TP": "Third place", "F": "Final",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _headline_section(scorecard: dict[str, float]) -> list[str]:
    n = int(scorecard["n"])
    return [
        "## Headline numbers",
        "",
        f"Scored over **{n} matches** (group stage through the knockout rounds):",
        "",
        "| Metric | Value | Skill vs uniform | What it measures |",
        "|--------|------:|-----------------:|------------------|",
        f"| Top-pick accuracy | {_pct(scorecard['accuracy'])} | vs 33.3% random | "
        "Share of matches where the model's pick matched the result |",
        f"| RPS | {scorecard['rps']:.4f} | {_pct(scorecard['rps_skill'])} | "
        "Ranked probability score over ordered H/D/A — lower is better |",
        f"| Brier score | {scorecard['brier_score']:.3f} | {_pct(scorecard['brier_skill'])} | "
        "Mean squared probability error — lower is better |",
        f"| Log loss | {scorecard['log_loss']:.3f} | {_pct(scorecard['log_loss_skill'])} | "
        "Surprise at realized outcomes — punishes confident misses |",
        "",
    ]


def _per_round_section(per_round: pl.DataFrame) -> list[str]:
    lines = [
        "## Accuracy by round",
        "",
        "| Round | Matches | Top-pick accuracy | Avg P(actual) |",
        "|-------|--------:|------------------:|--------------:|",
    ]
    for r in per_round.iter_rows(named=True):
        label = _ROUND_LABELS.get(str(r["round"]), str(r["round"]))
        lines.append(
            f"| {label} | {r['n']} | {_pct(float(r['accuracy']))} "
            f"| {_pct(float(r['avg_p_actual']))} |"
        )
    lines.append("")
    return lines


def _trajectory_section(trajectory: pl.DataFrame, top_n: int = 6) -> list[str]:
    """Champion-probability snapshots: pre-tournament, pre-knockout, pre-final, final."""

    dates = sorted(trajectory["date"].unique().to_list())
    if len(dates) < 2:
        return []
    picks = {
        "Pre-tournament": dates[0],
        "Mid-tournament": dates[len(dates) // 2],
        "Pre-final": dates[-2],
        "Final": dates[-1],
    }
    last = trajectory.filter(pl.col("date") == dates[-1])
    top_teams = (
        trajectory.group_by("team")
        .agg(pl.col("champion").max().alias("peak"))
        .sort("peak", descending=True)
        .head(top_n)["team"]
        .to_list()
    )
    lines = [
        "## The title race, replayed honestly",
        "",
        "Each column re-simulates the tournament using only information available on "
        "that date (Dixon-Coles refit on pre-date history, played results locked, real "
        "bracket walked — no hindsight). Full daily resolution lives in "
        "`data/outputs/champion_trajectory.parquet` and on the dashboard's Tournament "
        "Retrospective page.",
        "",
        "| Team | " + " | ".join(picks) + " |",
        "|------|" + "|".join("---:" for _ in picks) + "|",
    ]
    for team in top_teams:
        cells = []
        for d in picks.values():
            row = trajectory.filter((pl.col("team") == team) & (pl.col("date") == d))
            cells.append(_pct(float(row["champion"][0])) if row.height else "—")
        lines.append(f"| {team} | " + " | ".join(cells) + " |")
    lines.append("")
    champion = last.filter(pl.col("champion") >= 0.999)
    if not champion.is_empty():
        lines += [f"**Champion: {champion.row(0, named=True)['team']}.**", ""]
    return lines


def _upsets_section(upsets: pl.DataFrame) -> list[str]:
    if upsets.is_empty():
        return ["## Upsets", "", "_No result fell below 25% model probability._", ""]
    lines = [
        "## Upsets the model didn't see coming",
        "",
        "Results given under 25% probability:",
        "",
        "| Fixture | Score | Model pick | Actual | P(actual) |",
        "|---------|-------|-----------|--------|----------:|",
    ]
    for r in upsets.iter_rows(named=True):
        lines.append(
            f"| {r['Fixture']} | {r['Score']} | {r['Model Pick']} "
            f"({r['Pick Confidence']}) | {r['Actual Result']} | {r['P(Actual)']} |"
        )
    lines.append("")
    return lines


def _bookmaker_section(comparison: dict[str, object]) -> list[str]:
    if not comparison.get("available"):
        return [
            "## Model vs bookmaker favorites",
            "",
            f"_Not available: {comparison.get('reason', 'no workbook')}_",
            "",
        ]
    p = comparison.get("mcnemar_p")
    return [
        "## Model vs bookmaker favorites",
        "",
        f"On the {int(float(comparison['n_overlap']))} matches shared with the "  # type: ignore[arg-type]
        "bookmaker favorite tracker:",
        "",
        f"- Model top-pick accuracy: **{_pct(float(comparison['model_accuracy']))}**",  # type: ignore[arg-type]
        f"- Bookmaker favorite accuracy: **{_pct(float(comparison['book_accuracy']))}**",  # type: ignore[arg-type]
        f"- McNemar's test on disagreements: p = {float(p):.3f}" if p is not None else "",
        "",
        "The workbook tracks only the shortest-odds favorite (no full 1X2 prices), so "
        "this is an accuracy comparison, not a probability-scoring one.",
        "",
    ]


def _market_section(pnl: pl.DataFrame) -> list[str]:
    header = ["## Trading the champion market", ""]
    if pnl.is_empty():
        return [
            *header,
            "_No Polymarket price history was available for the resolved champion "
            "market, so no P&L backtest was run._",
            "",
        ]
    staked = float(pnl["stake"].sum())
    profit = float(pnl["profit"].sum())
    roi = profit / staked if staked > 0 else 0.0
    return [
        *header,
        "Quarter-Kelly stakes on positive model-vs-Polymarket edges (>3pp) in the "
        "world-cup-winner market at each replay date, settled at resolution "
        "(long-Yes only):",
        "",
        f"- Bets: **{pnl.height}**",
        f"- Total staked: **{staked:.3f} units**",
        f"- Profit: **{profit:+.3f} units** ({roi:+.1%} ROI)",
        "",
    ]


def build_retrospective_markdown(
    scorecard: dict[str, float],
    per_round: pl.DataFrame,
    upsets: pl.DataFrame,
    trajectory: pl.DataFrame,
    bookmaker: dict[str, object],
    pnl: pl.DataFrame,
    *,
    tournament: int = 2026,
) -> str:
    """Assemble the retrospective markdown from pre-computed frames (pure)."""

    sections: list[str] = [
        f"# FIFA World Cup {tournament} — Tournament Retrospective",
        "",
        "How the model actually did, scored against every completed match. Regenerate "
        "with `polymbappe retrospective` (trajectory via `polymbappe trajectory`).",
        "",
    ]
    sections += _headline_section(scorecard)
    if not per_round.is_empty():
        sections += _per_round_section(per_round)
    if not trajectory.is_empty():
        sections += _trajectory_section(trajectory)
    sections += _upsets_section(upsets)
    sections += _bookmaker_section(bookmaker)
    sections += _market_section(pnl)
    return "\n".join(sections).rstrip() + "\n"


def generate_retrospective(settings: Settings | None = None) -> str:
    """Compute the retrospective from current artifacts and write ``docs/results.md``."""

    from polymbappe.dashboard import data

    settings = settings or Settings()
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))
    if match_df.is_empty():
        raise FileNotFoundError("No match_predictions.parquet — run `polymbappe simulate`.")
    _, finished = data.split_fixtures(match_df, results)
    if finished.is_empty():
        raise ValueError("No finished matches to retrospect on — run `polymbappe ingest --live`.")

    markdown = build_retrospective_markdown(
        data.prediction_scorecard(finished),
        data.per_round_accuracy(match_df, results, data.load_schedule(settings)),
        data.actual_upsets(finished, threshold=0.25),
        data.load_champion_trajectory(settings),
        data.bookmaker_comparison(finished, settings),
        data.load_market_pnl(settings),
    )
    out_path = Path("docs") / "results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    logger.info("retrospective.written", path=str(out_path), bytes=len(markdown))
    return str(out_path)
