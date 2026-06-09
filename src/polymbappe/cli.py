"""Typer command line interface."""

from __future__ import annotations

from datetime import date

import typer

from polymbappe.data.ingest import ingest_all_sources
from polymbappe.eval.backtest import run_walk_forward_backtest
from polymbappe.eval.market import compare_model_to_market
from polymbappe.eval.report import generate_report
from polymbappe.features.pipeline import build_feature_matrix
from polymbappe.models.train import train_models
from polymbappe.simulate.tournament import run_tournament_simulation

app = typer.Typer(help="polymbappe forecasting CLI")


@app.command("ingest")
def ingest_command(live: bool = False) -> None:
    """Ingest source datasets."""

    _ = live
    ingest_all_sources()


@app.command("features")
def features_command(
    as_of: str | None = typer.Option(None, help="Only use data before this YYYY-MM-DD date."),
    contextual: bool = typer.Option(False, help="Build contextual feature table."),
) -> None:
    """Build the feature matrix."""

    as_of_date = date.fromisoformat(as_of) if as_of else None
    build_feature_matrix(as_of=as_of_date, contextual=contextual)


@app.command("train")
def train_command(model: str | None = typer.Option(None, help="Fit a single model only.")) -> None:
    """Train forecasting models."""

    train_models(model=model)


@app.command("simulate")
def simulate_command(
    tournament: int = 2026,
    n_sims: int = 50_000,
    with_context: bool = False,
    live: bool = False,
) -> None:
    """Run Monte Carlo tournament simulation."""

    _ = (tournament, n_sims, with_context, live)
    run_tournament_simulation()


@app.command("backtest")
def backtest_command(format_version: int = 2026) -> None:
    """Run walk-forward backtest."""

    _ = format_version
    run_walk_forward_backtest()


@app.command("edges")
def edges_command(tournament: int = 2026) -> None:
    """Print model-vs-market edge table."""

    _ = tournament
    compare_model_to_market()


@app.command("report")
def report_command(tournament: int = 2026) -> None:
    """Generate the tournament prediction report."""

    generate_report(tournament=tournament)


@app.command("autotune")
def autotune_command(
    budget: str = "2h",
    metric: str = "rps",
    resume: bool = False,
    leaderboard: bool = False,
    apply_best: bool = False,
) -> None:
    """Run the automated hyperparameter tuning loop (Section 8)."""

    _ = (budget, metric, resume, leaderboard, apply_best)
    raise NotImplementedError("Autotuner is a later phase; not yet implemented.")


@app.command("agent")
def agent_command(
    run_now: bool = False,
    status: bool = False,
    history: bool = False,
    schedule: str | None = None,
) -> None:
    """Control the LangGraph live monitoring agent (Section 5)."""

    _ = (run_now, status, history, schedule)
    raise NotImplementedError("Live monitoring agent is a later phase; not yet implemented.")


@app.command("dashboard")
def dashboard_command() -> None:
    """Launch the Streamlit dashboard (Section 6)."""

    raise NotImplementedError("Dashboard is a later phase; not yet implemented.")


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
