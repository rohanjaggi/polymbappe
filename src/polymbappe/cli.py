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

    report = ingest_all_sources(live=live)
    typer.echo(report)


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

    _ = tournament
    run_tournament_simulation(n_sims=n_sims, with_context=with_context, live=live)


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

    from polymbappe.tune.runner import run_autotune

    run_autotune(
        budget=budget,
        metric=metric,
        resume=resume,
        leaderboard=leaderboard,
        apply_best=apply_best,
    )


@app.command("agent")
def agent_command(
    run_now: bool = False,
    status: bool = False,
    history: bool = False,
    schedule: str | None = None,
) -> None:
    """Control the LangGraph live monitoring agent (Section 5)."""

    from polymbappe.agent.scheduler import parse_interval
    from polymbappe.agent.scheduler import run_now as agent_run_now
    from polymbappe.agent.state import AgentState

    if run_now:
        summary = agent_run_now(teams=[])
        typer.echo(summary)
        return
    if status:
        with AgentState() as state:
            typer.echo(state.player_statuses_df())
        return
    if history:
        with AgentState() as state:
            typer.echo(state.changelog_df())
        return
    if schedule:
        secs = parse_interval(schedule)
        typer.echo(f"Scheduling agent every {secs}s (requires the 'context' extra).")
        return
    typer.echo("Specify one of --run-now / --status / --history / --schedule.")


@app.command("dashboard")
def dashboard_command() -> None:
    """Launch the Streamlit dashboard (Section 6)."""

    from polymbappe.dashboard.app import main

    main()


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
