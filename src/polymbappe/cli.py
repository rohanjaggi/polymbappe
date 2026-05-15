"""Typer command line interface."""

from __future__ import annotations

import typer

from polymbappe.data.ingest import ingest_all_sources
from polymbappe.eval.backtest import run_walk_forward_backtest
from polymbappe.eval.market import compare_model_to_market
from polymbappe.simulate.tournament import run_tournament_simulation

app = typer.Typer(help="polymbappe forecasting CLI")


@app.command("ingest")
def ingest_command() -> None:
    """Ingest source datasets."""

    ingest_all_sources()


@app.command("train")
def train_command() -> None:
    """Train forecasting models."""

    raise NotImplementedError("Training orchestration is scaffolded.")


@app.command("simulate")
def simulate_command(tournament: int = 2026, n_sims: int = 50_000) -> None:
    """Run Monte Carlo tournament simulation."""

    _ = (tournament, n_sims)
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


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
