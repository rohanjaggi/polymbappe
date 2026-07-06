"""Page 7 — Predictions vs Actuals (spec section 6.1).

Scores the model's pre-match H/D/A forecasts against recorded tournament results. Where
the Match Predictor page (page 3) lists fixtures and per-fixture probabilities, this page
is the *evaluation* view: headline accuracy / Brier / log-loss, a calibration reliability
diagram, accuracy broken down by realized outcome, and a per-match scorecard. It reuses
:func:`polymbappe.dashboard.data.split_fixtures` to join predictions to results, so it
stays in sync with how finished matches are determined elsewhere. ``streamlit`` is
imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts

#: Display label for each H/D/A outcome key.
_OUTCOME_LABEL = {"home": "Home win", "draw": "Draw", "away": "Away win"}


def render(settings: Settings) -> None:
    """Render the Predictions vs Actuals page (spec 6.1, page 7)."""

    import streamlit as st

    st.header("Predictions vs Actuals")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info(
            "No match predictions yet. Run `polymbappe simulate`/`report` to populate the "
            "dashboard."
        )
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)
    match_xg = data.load_match_xg(settings)

    st.caption(
        "How the model's pre-match H/D/A probabilities held up against recorded tournament "
        "results. Only finished fixtures are scored; predictions come from the calibration "
        "pipeline (spec 3.6) and results from `polymbappe ingest`."
    )

    if finished.is_empty():
        st.info(
            "No finished matches recorded yet. Ingest results (`polymbappe ingest`) as the "
            "tournament progresses."
        )
        return

    _render_scorecard(st, finished)
    st.divider()
    _render_breakdowns(st, finished)
    st.divider()
    _render_xg_analysis(st, finished, match_xg)
    st.divider()
    _render_match_table(st, finished)


def _render_scorecard(st: object, finished: pl.DataFrame) -> None:
    """Headline accuracy / Brier / log-loss metrics across all finished matches."""

    scorecard = data.prediction_scorecard(finished)
    cols = st.columns(4)
    cols[0].metric("Matches scored", int(scorecard["n"]))
    cols[1].metric("Top-pick accuracy", f"{scorecard['accuracy']:.1%}")
    cols[2].metric(
        "Brier score",
        f"{scorecard['brier_score']:.3f}",
        help="Mean squared error over H/D/A — lower is better (0 best, 2 worst).",
    )
    cols[3].metric(
        "Log loss",
        f"{scorecard['log_loss']:.3f}",
        help="Mean negative log-probability of the realized outcome — lower is better.",
    )


def _render_breakdowns(st: object, finished: pl.DataFrame) -> None:
    """Side-by-side accuracy-by-outcome bar and calibration reliability diagram."""

    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy by outcome")
        st.plotly_chart(
            charts.outcome_accuracy_bar(data.accuracy_by_outcome(finished)),
            use_container_width=True,
        )
    with right:
        st.subheader("Calibration")
        st.plotly_chart(
            charts.calibration_curve(data.calibration_bins(finished)),
            use_container_width=True,
        )


def _render_xg_analysis(
    st: object, finished: pl.DataFrame, match_xg: pl.DataFrame
) -> None:
    """xG error analysis: model vs goals, model vs actual xG, and finishing luck."""

    st.subheader("xG prediction error")

    needed = {"exp_home_goals", "exp_away_goals", "home_goals", "away_goals"}
    if not needed.issubset(finished.columns):
        st.info("No xG predictions in this simulation run.")
        return

    has_actual_xg = not match_xg.is_empty()
    summary = data.xg_error_summary(finished, match_xg if has_actual_xg else None)

    if has_actual_xg and "xg_n" in summary:
        st.caption(
            f"Actual xG from FBref available for {int(summary['xg_n'])} matches. "
            "Error is decomposed into model quality (vs actual xG) and finishing luck "
            "(actual xG vs goals)."
        )
        cols = st.columns(3)
        cols[0].metric(
            "Model vs actual xG (MAE)",
            f"{summary['model_vs_xg_mae']:.2f}",
            help="Mean |model predicted xG − FBref actual xG|. Pure model quality.",
        )
        cols[1].metric(
            "Finishing luck (MAE)",
            f"{summary['xg_vs_goals_mae']:.2f}",
            help="Mean |FBref actual xG − actual goals|. Variance from conversion luck.",
        )
        cols[2].metric(
            "Model vs goals (MAE)",
            f"{summary['total_mae']:.2f}",
            help="Combined: model quality + finishing luck.",
        )
    else:
        if not has_actual_xg:
            st.caption(
                "Run `polymbappe ingest --live` to pull FBref actual xG and decompose "
                "model error from finishing-luck variance."
            )
        cols = st.columns(3)
        cols[0].metric("Home xG MAE", f"{summary['home_mae']:.2f}",
                       help="Mean |predicted home xG − actual home goals|.")
        cols[1].metric("Away xG MAE", f"{summary['away_mae']:.2f}",
                       help="Mean |predicted away xG − actual away goals|.")
        cols[2].metric("Overall xG MAE", f"{summary['total_mae']:.2f}")

    st.plotly_chart(
        charts.xg_scatter(finished, match_xg if has_actual_xg else None),
        use_container_width=True,
    )

    st.subheader(f"Per-match xG breakdown ({finished.height})")
    st.dataframe(_xg_table(finished, match_xg), use_container_width=True, hide_index=True)


def _xg_table(finished: pl.DataFrame, match_xg: pl.DataFrame) -> object:
    """Per-match xG table; adds FBref actual xG columns when available."""

    has_xg = not match_xg.is_empty()
    if has_xg:
        xg_slim = match_xg.select(["home_team", "away_team", "home_xg", "away_xg"])
        joined = finished.join(xg_slim, on=["home_team", "away_team"], how="left")
        unmatched = joined.filter(pl.col("home_xg").is_null()).select(finished.columns)
        if not unmatched.is_empty():
            xg_rev = xg_slim.rename(
                {"home_team": "away_team", "away_team": "home_team",
                 "home_xg": "away_xg", "away_xg": "home_xg"}
            )
            rev_joined = unmatched.join(xg_rev, on=["home_team", "away_team"], how="left")
            matched = joined.filter(pl.col("home_xg").is_not_null())
            joined = pl.concat([matched, rev_joined], how="diagonal_relaxed")
    else:
        joined = finished

    rows = []
    for r in joined.iter_rows(named=True):
        ph = float(r["exp_home_goals"])
        pa = float(r["exp_away_goals"])
        ah = float(r["home_goals"])
        aa = float(r["away_goals"])
        row: dict[str, object] = {
            "Fixture": f"{r['home_team']} vs {r['away_team']}",
            "Model xG (H)": f"{ph:.2f}",
            "Model xG (A)": f"{pa:.2f}",
        }
        if has_xg and r.get("home_xg") is not None:
            fh = float(r["home_xg"])
            fa = float(r["away_xg"])
            row["FBref xG (H)"] = f"{fh:.2f}"
            row["FBref xG (A)"] = f"{fa:.2f}"
            row["Model err (H)"] = f"{abs(ph - fh):.2f}"
            row["Model err (A)"] = f"{abs(pa - fa):.2f}"
            row["Luck (H)"] = f"{abs(fh - ah):.2f}"
            row["Luck (A)"] = f"{abs(fa - aa):.2f}"
        row["Actual (H)"] = int(ah)
        row["Actual (A)"] = int(aa)
        rows.append(row)
    return pl.DataFrame(rows).to_pandas()


def _render_match_table(st: object, finished: pl.DataFrame) -> None:
    """Per-match scorecard: scoreline, model pick, and probabilities of pick vs. outcome."""

    st.subheader(f"Per-match scorecard ({finished.height})")
    st.dataframe(_scorecard_table(finished), use_container_width=True, hide_index=True)


def _scorecard_table(finished: pl.DataFrame) -> object:
    """Pandas display frame: each finished match with its model call vs. the result."""

    rows = []
    for r in finished.iter_rows(named=True):
        probs = {
            "home": float(r["model_home"]),
            "draw": float(r["model_draw"]),
            "away": float(r["model_away"]),
        }
        actual = str(r["actual_outcome"])
        rows.append(
            {
                "Date": str(r["date"]) if r.get("date") is not None else "",
                "Group": r["group"],
                "Fixture": f"{r['home_team']} vs {r['away_team']}",
                "Score": f"{int(r['home_goals'])} – {int(r['away_goals'])}",
                "Result": _OUTCOME_LABEL.get(actual, actual),
                "Model pick": _OUTCOME_LABEL.get(str(r["model_pick"]), str(r["model_pick"])),
                "P(pick)": f"{max(probs.values()):.1%}",
                "P(actual)": f"{probs[actual]:.1%}",
                "Correct": "✅" if r["model_correct"] else "❌",
            }
        )
    return pl.DataFrame(rows).to_pandas()
