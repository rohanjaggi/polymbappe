"""Prediction report generation.

Assembles tournament probability outputs and edge reports into the artifacts
under ``data/outputs/``.
"""

from __future__ import annotations

import polars as pl
import structlog

from polymbappe.config import Settings

logger = structlog.get_logger(__name__)

#: Reach-stage columns in ``stage_probabilities`` (group-stage exit -> title), in order.
_STAGE_COLUMNS = ("R32", "R16", "QF", "SF", "FINAL", "champion")
_STAGE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-final",
    "SF": "Semi-final",
    "FINAL": "Final",
    "champion": "Champion",
}


def _read(settings: Settings, name: str) -> pl.DataFrame | None:
    """Read an ``data/outputs/<name>.parquet`` table, or ``None`` when absent."""

    path = settings.outputs_data_dir / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def _fmt_pct(value: float) -> str:
    """Render a 0-1 probability as a one-decimal percentage."""

    return f"{value * 100:.1f}%"


def _champions_section(stage: pl.DataFrame, top: int = 12) -> list[str]:
    """Championship-odds leaderboard sorted by ``champion`` probability."""

    ranked = stage.sort("champion", descending=True).head(top)
    lines = [
        "## Championship odds",
        "",
        "| Rank | Team | Champion | Reach final | Reach SF |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked.iter_rows(named=True), start=1):
        lines.append(
            f"| {i} | {row['team']} | {_fmt_pct(row['champion'])} "
            f"| {_fmt_pct(row['FINAL'])} | {_fmt_pct(row['SF'])} |"
        )
    lines.append("")
    return lines


def _stage_section(stage: pl.DataFrame, top: int = 16) -> list[str]:
    """Per-stage reach probabilities for the strongest teams."""

    ranked = stage.sort("champion", descending=True).head(top)
    header = " | ".join(_STAGE_LABELS[c] for c in _STAGE_COLUMNS)
    sep = " | ".join("---:" for _ in _STAGE_COLUMNS)
    lines = [
        "## Stage-reach probabilities",
        "",
        f"| Team | {header} |",
        f"| --- | {sep} |",
    ]
    for row in ranked.iter_rows(named=True):
        cells = " | ".join(_fmt_pct(row[c]) for c in _STAGE_COLUMNS)
        lines.append(f"| {row['team']} | {cells} |")
    lines.append("")
    return lines


def _group_section(group: pl.DataFrame, predictions: pl.DataFrame | None) -> list[str]:
    """Group-stage outlook: P(win group) / P(advance) per team, grouped by group label.

    ``finish_1``/``finish_2`` are the simulated probabilities of finishing 1st/2nd (both
    advance to the knockouts). The group label for each team is recovered from
    ``match_predictions`` when available; teams without one fall to an "Unassigned" bucket.
    """

    advance = group.with_columns(
        (pl.col("finish_1") + pl.col("finish_2")).alias("advance")
    )

    team_group: dict[str, str] = {}
    if predictions is not None and "group" in predictions.columns:
        for row in predictions.iter_rows(named=True):
            team_group.setdefault(row["home_team"], row["group"])
            team_group.setdefault(row["away_team"], row["group"])

    advance = advance.with_columns(
        pl.col("team")
        .map_elements(lambda t: team_group.get(t, "—"), return_dtype=pl.Utf8)
        .alias("group")
    )

    lines = ["## Group-stage outlook", ""]
    for label in sorted(advance["group"].unique().to_list()):
        block = advance.filter(pl.col("group") == label).sort("finish_1", descending=True)
        title = "Unassigned" if label == "—" else f"Group {label}"
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Team | Win group | Advance |")
        lines.append("| --- | ---: | ---: |")
        for row in block.iter_rows(named=True):
            lines.append(
                f"| {row['team']} | {_fmt_pct(row['finish_1'])} | {_fmt_pct(row['advance'])} |"
            )
        lines.append("")
    return lines


def _matches_section(predictions: pl.DataFrame, top: int = 10) -> list[str]:
    """Highlight the most uncertain (lowest favourite-probability) group fixtures."""

    annotated = predictions.with_columns(
        pl.max_horizontal("model_home", "model_draw", "model_away").alias("_fav")
    ).sort("_fav").head(top)

    lines = [
        "## Closest fixtures (most uncertain)",
        "",
        "| Group | Home | Away | Home win | Draw | Away win | xG (H–A) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in annotated.iter_rows(named=True):
        lines.append(
            f"| {row['group']} | {row['home_team']} | {row['away_team']} "
            f"| {_fmt_pct(row['model_home'])} | {_fmt_pct(row['model_draw'])} "
            f"| {_fmt_pct(row['model_away'])} "
            f"| {row['exp_home_goals']:.2f}–{row['exp_away_goals']:.2f} |"
        )
    lines.append("")
    return lines


def _edges_section(edges: pl.DataFrame | None) -> list[str]:
    """Per-match market edges, or a note when no live match odds are ingested."""

    lines = ["## Market edges", ""]
    if edges is None or edges.height == 0:
        lines.append(
            "_No per-match market odds ingested. Pre-tournament, only outright/futures "
            "markets are tradeable — run `polymbappe edges --outright` for those._"
        )
        lines.append("")
        return lines

    ranked = edges.sort("edge_bps", descending=True)
    lines.append("| Match | Outcome | Model | Market | Edge (bps) | Kelly |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in ranked.iter_rows(named=True):
        lines.append(
            f"| {row['match_id']} | {row['outcome']} | {_fmt_pct(row['model_prob'])} "
            f"| {_fmt_pct(row['market_prob'])} | {row['edge_bps']:.0f} "
            f"| {row['kelly_fraction']:.3f} |"
        )
    lines.append("")
    return lines


def generate_report(tournament: int = 2026, settings: Settings | None = None) -> str:
    """Generate the tournament prediction report from the simulation outputs.

    Reads the ``stage_probabilities`` / ``group_probabilities`` / ``match_predictions`` /
    ``edges`` parquet artifacts written by ``polymbappe simulate`` and assembles a Markdown
    report at ``data/outputs/report.md``. Raises if the simulation has not been run.

    Args:
        tournament: Tournament year to report on.
        settings: Optional settings override.

    Returns:
        The path (as a string) of the written report.
    """

    logger.info("report.start", tournament=tournament)
    settings = settings or Settings()

    stage = _read(settings, "stage_probabilities")
    if stage is None or stage.height == 0:
        raise FileNotFoundError(
            "No stage_probabilities.parquet — run `polymbappe simulate` before reporting."
        )
    group = _read(settings, "group_probabilities")
    predictions = _read(settings, "match_predictions")
    edges = _read(settings, "edges")

    sections: list[str] = [
        f"# FIFA World Cup {tournament} — Prediction Report",
        "",
        f"Monte Carlo simulation over {stage.height} teams. Probabilities are model "
        "estimates, not guarantees.",
        "",
    ]
    sections += _champions_section(stage)
    sections += _stage_section(stage)
    if group is not None and group.height > 0:
        sections += _group_section(group, predictions)
    if predictions is not None and predictions.height > 0:
        sections += _matches_section(predictions)
    sections += _edges_section(edges)

    report = "\n".join(sections).rstrip() + "\n"
    out_path = settings.outputs_data_dir / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("report.done", path=str(out_path), bytes=len(report))
    return str(out_path)
