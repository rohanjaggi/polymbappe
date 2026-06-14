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
