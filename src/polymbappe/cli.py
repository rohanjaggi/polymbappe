"""Typer command line interface."""

from __future__ import annotations

from datetime import date

import polars as pl
import typer

from polymbappe.data.ingest import ingest_all_sources
from polymbappe.eval.backtest import run_bayesian_ab, run_walk_forward_backtest
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


@app.command("squad-coverage")
def squad_coverage_command() -> None:
    """Report Kaggle squad-valuation name-match coverage per (team, tournament), worst-first."""

    from polymbappe.data.ingest import squad_coverage

    coverage = squad_coverage()
    if coverage.is_empty():
        typer.echo(
            "No squad-valuation coverage to report "
            "(needs the squads table and data/raw/squad_valuations_kaggle.txt)."
        )
        return
    with pl.Config(tbl_rows=-1):
        typer.echo(coverage)


@app.command("features")
def features_command(
    as_of: str | None = typer.Option(None, help="Only use data before this YYYY-MM-DD date."),
    contextual: bool = typer.Option(False, help="Build contextual feature table."),
) -> None:
    """Build the feature matrix."""

    as_of_date = date.fromisoformat(as_of) if as_of else None
    build_feature_matrix(as_of=as_of_date, contextual=contextual)


@app.command("train")
def train_command(
    model: str | None = typer.Option(None, help="Fit a single model only."),
    bayesian: bool = typer.Option(
        False, "--bayesian", help="Also fit/stack the (expensive) Bayesian hierarchical DC model."
    ),
) -> None:
    """Train forecasting models."""

    train_models(model=model, bayesian=bayesian)


@app.command("simulate")
def simulate_command(
    tournament: int = 2026,
    n_sims: int = 50_000,
    with_context: bool = typer.Option(
        False,
        "--with-context",
        help="Apply adaptive contextual weights (from contextual-monitor --apply).",
    ),
    historical_context: bool = typer.Option(
        False,
        "--historical-context",
        help=(
            "Diagnostic: use the historically-trained LightGBM adjuster"
            " instead of adaptive weights."
        ),
    ),
    live: bool = False,
    refresh_odds: bool = typer.Option(
        False, "--refresh-odds", help="Re-pull market odds before computing edges (live updates)."
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Override POLYMBAPPE_RANDOM_SEED for this run (reproducible outputs)."
    ),
) -> None:
    """Run Monte Carlo tournament simulation."""

    _ = tournament
    run_tournament_simulation(
        n_sims=n_sims,
        with_context=with_context,
        historical_context=historical_context,
        live=live,
        refresh_odds=refresh_odds,
        seed=seed,
    )


@app.command("backtest")
def backtest_command(format_version: int = 2026) -> None:
    """Run walk-forward backtest."""

    _ = format_version
    run_walk_forward_backtest()


@app.command("bayesian-ab")
def bayesian_ab_command() -> None:
    """Run the Bayesian kill-criterion A/B (LOTO backtest with vs without Bayesian)."""

    run_bayesian_ab()


@app.command("edges")
def edges_command(
    tournament: int = 2026,
    outright: bool = typer.Option(
        False, help="Show outright/futures edges (e.g. champion) vs a Polymarket market."
    ),
    market: str = typer.Option(
        "world-cup-winner", help="Polymarket futures slug for --outright."
    ),
) -> None:
    """Print model-vs-market edge table (per-match by default, or --outright futures)."""

    _ = tournament
    if outright:
        from polymbappe.eval.market import compare_outright_to_market

        compare_outright_to_market(market)
    else:
        compare_model_to_market()


@app.command("report")
def report_command(tournament: int = 2026) -> None:
    """Generate the tournament prediction report."""

    path = generate_report(tournament=tournament)
    typer.echo(f"Report written to {path}")


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


@app.command("contextual-monitor")
def contextual_monitor_command(
    apply: bool = typer.Option(False, "--apply", help="Write weights to file after testing."),
    min_matches: int = typer.Option(
        32, "--min-matches", help="Skip if fewer completed WC2026 matches."
    ),
) -> None:
    """Test contextual feature signals on live WC2026 results.

    Runs a signal test for each contextual feature group (xg_overperformance, draw_pressure,
    cohesion, manager, fatigue) to check whether it correlates with live WC2026 outcomes
    beyond the base model. Gate: p < 0.05 AND RPS improvement > 0.003.

    With --apply, saves passing weights to data/outputs/contextual_wc2026_weights.json so
    the next ``simulate --with-context`` call picks them up automatically.
    """

    from polymbappe.config import Settings
    from polymbappe.context.adaptive import (
        MIN_MATCHES,
        AdaptiveWeightState,
        append_attribution,
        compute_wc2026_base_predictions,
        labels_from_matches,
        load_live_wc2026_matches,
        run_all_signal_tests,
        save_adaptive_weights,
    )
    from polymbappe.context.runtime import build_tournament_context_features
    from polymbappe.data.store import read_table
    from polymbappe.data.tables import Table

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)

    live = (
        load_live_wc2026_matches(matches, settings)
        .filter(pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null())
        .unique(subset=["match_id"], keep="last", maintain_order=True)
        .sort("date", "match_id")
    )
    n = live.height
    gate = max(min_matches, MIN_MATCHES)

    if n < gate:
        typer.echo(
            f"Only {n} completed WC2026 matches (need {gate}). Run `ingest --live` to update."
        )
        return

    typer.echo(f"Testing contextual signals on {n} completed WC2026 matches...")

    # Base predictions for the live fixtures
    base_preds = compute_wc2026_base_predictions(live, matches, settings)

    # Contextual features for those fixtures
    from dataclasses import dataclass
    from datetime import date as _date

    @dataclass
    class _WC2026Tournament:
        name: str = "WC2026"
        # Must match the matches table's competition value ("FIFA World Cup"); the
        # date window below is what scopes this to the 2026 edition.
        competition: str = "FIFA World Cup"
        start: _date = _date(2026, 6, 11)
        end: _date = _date(2026, 7, 19)

    ctx_df = build_tournament_context_features(matches, [_WC2026Tournament()], settings)
    # Align context rows to live match order (unique on match_id so the join can't fan out)
    ctx_aligned = (
        live.select("match_id")
        .join(
            ctx_df.unique(subset=["match_id"], keep="last", maintain_order=True),
            on="match_id",
            how="left",
        )
        .drop("match_id")
    )

    labels = labels_from_matches(live)

    results = run_all_signal_tests(labels, base_preds, ctx_aligned)
    append_attribution(results, settings)

    typer.echo(f"\n{'Group':<25} {'p-value':>10} {'RPS Δ':>10} {'Weight':>10} {'Active':>8}")
    typer.echo("-" * 65)
    for r in results:
        marker = "✓" if r.active else " "
        typer.echo(
            f"{marker} {r.feature_group:<23} {r.p_value:>10.4f} {r.rps_delta:>10.4f}"
            f" {r.weight:>10.4f} {str(r.active):>8}"
        )

    active_groups = [r.feature_group for r in results if r.active]
    typer.echo(f"\nActive groups: {active_groups or 'none'}")

    if apply:
        state = AdaptiveWeightState(
            weights={r.feature_group: r.weight for r in results},
            n_matches=n,
            last_updated=__import__("datetime").datetime.now().isoformat(),
        )
        path = save_adaptive_weights(state, settings)
        typer.echo(f"Weights saved → {path}")
    else:
        typer.echo("\nRe-run with --apply to activate weights in simulation.")


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

    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


@app.command("refresh")
def refresh_command(
    n_sims: int = typer.Option(50_000, help="Monte Carlo simulation count."),
) -> None:
    """Full live-update cycle: ingest results → re-train DC → re-simulate.

    Run once after each match day. Ingest fetches live results and odds;
    train re-fits Dixon-Coles; simulate rebuilds all parquet outputs including
    knockout predictions and market edges.
    """
    ingest_all_sources(live=True)
    train_models(model="dixon_coles")
    run_tournament_simulation(n_sims=n_sims, live=True, refresh_odds=True)


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
