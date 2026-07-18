"""Page 3 — Predictions vs Actuals.

Scores the model's pre-match H/D/A forecasts against recorded tournament results
with a stage filter (All / Group Stage / Knockout).
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts

#: Display label for each H/D/A outcome key.
_OUTCOME_LABEL = {"home": "Home win", "draw": "Draw", "away": "Away win"}


def render(settings: Settings) -> None:
    """Render the Predictions vs Actuals page."""

    import streamlit as st

    st.header("Predictions vs Actuals")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("No match predictions yet. Run `polymbappe simulate`/`report` to populate.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    _, finished = data.split_fixtures(match_df, results)
    match_xg = data.load_match_xg(settings)

    if finished.is_empty():
        st.info("No finished matches recorded yet.")
        return

    stage_filter = st.radio(
        "Stage", ["All", "Group Stage", "Knockout"], horizontal=True
    )
    if stage_filter == "Group Stage":
        if "group" in finished.columns:
            finished = finished.filter(pl.col("group") != "KO")
    elif stage_filter == "Knockout":
        finished = (
            finished.filter(pl.col("group") == "KO")
            if "group" in finished.columns
            else pl.DataFrame()
        )

    if finished.is_empty():
        st.info(f"No finished {stage_filter.lower()} matches yet.")
        return

    _render_scorecard(st, finished)
    st.divider()
    _render_breakdowns(st, finished)
    st.divider()
    _render_significance(st, finished)
    st.divider()
    _render_competitive_subset(st, finished)
    st.divider()
    _render_market(st, finished, settings)
    st.divider()
    _render_xg_analysis(st, finished, match_xg)
    st.divider()
    _render_match_table(st, finished)


def _render_scorecard(st: object, finished: pl.DataFrame) -> None:
    """Headline scoring rules across all finished matches, with skill vs. a uniform guess."""

    scorecard = data.prediction_scorecard(finished)
    cols = st.columns(5)
    cols[0].metric("Matches scored", int(scorecard["n"]))
    cols[1].metric("Top-pick accuracy", f"{scorecard['accuracy']:.1%}")
    cols[2].metric(
        "RPS",
        f"{scorecard['rps']:.3f}",
        help="Ranked Probability Score — the ordinal (H<D<A) proper scoring rule and the "
        "standard headline for 1X2 football. Lower is better; uniform ≈ 0.198.",
    )
    cols[3].metric(
        "Brier score",
        f"{scorecard['brier_score']:.3f}",
        help="Summed squared error over H/D/A — lower is better (0 best, 2 worst).",
    )
    cols[4].metric(
        "Log loss",
        f"{scorecard['log_loss']:.3f}",
        help="Mean negative log-probability of the realized outcome — lower is better.",
    )
    st.caption(
        "**Skill vs. a uniform (1/3,1/3,1/3) guess** — positive means the model carries "
        f"information: RPS **{scorecard['rps_skill']:+.1%}**, "
        f"log-loss **{scorecard['log_loss_skill']:+.1%}**, "
        f"Brier **{scorecard['brier_skill']:+.1%}**."
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
        cal = data.calibration_summary(finished)
        slope = "—" if cal["slope"] != cal["slope"] else f"{cal['slope']:.2f}"  # nan check
        intercept = "—" if cal["intercept"] != cal["intercept"] else f"{cal['intercept']:+.2f}"
        st.caption(
            f"**ECE {cal['ece']:.3f}** · MCE {cal['mce']:.3f} · calibration slope "
            f"**{slope}** (1 = perfect; <1 overconfident, >1 underconfident), "
            f"intercept {intercept}."
        )


def _render_significance(st: object, finished: pl.DataFrame) -> None:
    """Paired test that the model's per-match RPS genuinely beats a uniform forecast."""

    st.subheader("Is the edge real? (RPS vs. uniform)")
    sig = data.rps_significance(finished)
    cols = st.columns(3)
    cols[0].metric(
        "Mean per-match RPS gap",
        f"{sig['mean_diff']:+.4f}",
        help="Model minus uniform per-match RPS. Negative = the model is sharper.",
    )
    cols[1].metric(
        "95% bootstrap CI",
        f"[{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}]",
        help="Paired bootstrap over matches. Entirely below 0 ⇒ significant at 95%.",
    )
    cols[2].metric("Wilcoxon p", f"{sig['wilcoxon_p']:.3f}")
    beats = sig["ci_high"] < 0
    st.caption(
        "✅ The model's probabilities significantly beat a uniform guess on these "
        "fixtures (bootstrap CI below 0)." if beats else
        "⚠️ Not yet significant vs. a uniform guess at this sample size — expected with "
        "few matches; scoring rules need more fixtures to separate."
    )


def _render_competitive_subset(st: object, finished: pl.DataFrame) -> None:
    """Re-report the headline scoring rules on close games (favourite prob 40–60%)."""

    st.subheader("Competitive subset (favourite 40–60%)")
    st.caption(
        "The number that actually reveals skill: restrict to close games where the "
        "favourite's probability is 40–60%. If the edge survives here, it isn't just "
        "calling blowouts."
    )
    subset = data.competitive_subset(finished)
    if subset.is_empty():
        st.info("No finished matches fall in the 40–60% favourite band yet.")
        return
    card = data.prediction_scorecard(subset)
    cols = st.columns(4)
    cols[0].metric("Close games", int(card["n"]))
    cols[1].metric("Accuracy", f"{card['accuracy']:.1%}")
    cols[2].metric("RPS", f"{card['rps']:.3f}")
    cols[3].metric("RPS skill vs uniform", f"{card['rps_skill']:+.1%}")


def _render_market(st: object, finished: pl.DataFrame, settings: Settings) -> None:
    """Head-to-head vs. the bookmaker favorite (accuracy + McNemar); market-prob stub."""

    st.subheader("Vs. the bookmaker (shortest-odds favorite)")
    cmp = data.bookmaker_comparison(finished, settings)
    if not cmp.get("available"):
        st.info(f"Bookmaker comparison unavailable: {cmp.get('reason', 'no data')}")
    else:
        cols = st.columns(4)
        cols[0].metric("Matches compared", int(cmp["n_overlap"]))
        cols[1].metric("Model accuracy", f"{cmp['model_accuracy']:.1%}")
        cols[2].metric("Bookmaker accuracy", f"{cmp['book_accuracy']:.1%}")
        cols[3].metric(
            "McNemar p",
            f"{cmp['mcnemar_p']:.3f}",
            help="Paired test on the fixtures where model and bookmaker disagree. "
            f"Model-right/book-wrong={int(cmp['mcnemar_b'])}, "
            f"book-right/model-wrong={int(cmp['mcnemar_c'])}.",
        )
        if cmp.get("n_unmatched"):
            st.caption(
                f"{int(cmp['n_unmatched'])} model fixture(s) had no workbook match and "
                "were excluded from the head-to-head."
            )
    st.warning(
        "**Probability-level market metrics (market RPS skill, ROI, CLV) are not shown.** "
        + str(cmp.get("market_prob_reason", ""))
    )


def _render_xg_analysis(
    st: object, finished: pl.DataFrame, match_xg: pl.DataFrame
) -> None:
    """xG error analysis: model vs goals, model vs actual xG, and finishing luck."""

    st.subheader("Expected Goals (xG) Analysis")

    needed = {"exp_home_goals", "exp_away_goals", "home_goals", "away_goals"}
    if not needed.issubset(finished.columns):
        st.info("No xG predictions in this simulation run.")
        return

    has_actual_xg = not match_xg.is_empty()
    summary = data.xg_error_summary(finished, match_xg if has_actual_xg else None)

    # High-level overview metrics
    total_pred = float((finished["exp_home_goals"] + finished["exp_away_goals"]).sum())
    total_actual = float((finished["home_goals"] + finished["away_goals"]).sum())
    avg_pred = float((finished["exp_home_goals"] + finished["exp_away_goals"]).mean())
    avg_actual = float((finished["home_goals"] + finished["away_goals"]).mean())

    xg_winner_correct = 0
    for r in finished.iter_rows(named=True):
        ph, pa = float(r["exp_home_goals"]), float(r["exp_away_goals"])
        ah, aa = int(r["home_goals"]), int(r["away_goals"])
        pred_w = "home" if ph > pa else ("away" if pa > ph else "draw")
        act_w = "home" if ah > aa else ("away" if aa > ah else "draw")
        if pred_w == act_w:
            xg_winner_correct += 1

    row1 = st.columns(4)
    row1[0].metric(
        "Total predicted goals", f"{total_pred:.0f}",
        delta=f"{total_pred - total_actual:+.0f} vs actual {total_actual:.0f}",
        delta_color="off",
    )
    row1[1].metric("Avg goals/match (predicted)", f"{avg_pred:.2f}")
    row1[2].metric("Avg goals/match (actual)", f"{avg_actual:.2f}")
    row1[3].metric(
        "xG winner accuracy",
        f"{xg_winner_correct}/{finished.height} ({xg_winner_correct / finished.height:.0%})",
        help="How often the team with higher predicted xG actually won.",
    )

    st.divider()

    # Error decomposition
    if has_actual_xg and "xg_n" in summary:
        st.caption(
            f"FBref actual xG available for {int(summary['xg_n'])} matches. "
            "The model's goal prediction error breaks down into two parts: "
            "how well it predicted the chances created (model error), and "
            "how much finishing variance affected the outcome (luck)."
        )
        row2 = st.columns(3)
        row2[0].metric(
            "Model error (MAE)",
            f"{summary['model_vs_xg_mae']:.2f}",
            help=(
                "Avg difference between our predicted xG and FBref's actual xG."
                " Measures pure model quality."
            ),
        )
        row2[1].metric(
            "Finishing luck (MAE)",
            f"{summary['xg_vs_goals_mae']:.2f}",
            help=(
                "Avg difference between actual xG and goals scored."
                " Variance outside model control."
            ),
        )
        row2[2].metric(
            "Total error (MAE)",
            f"{summary['total_mae']:.2f}",
            help=(
                "Avg difference between predicted xG and actual goals."
                " Combines model error + luck."
            ),
        )
    else:
        if not has_actual_xg:
            st.caption(
                "Run `polymbappe ingest --live` to pull FBref actual xG and decompose "
                "model error from finishing-luck variance."
            )
        row2 = st.columns(3)
        row2[0].metric("Home xG MAE", f"{summary['home_mae']:.2f}",
                       help="Mean |predicted home xG − actual home goals|.")
        row2[1].metric("Away xG MAE", f"{summary['away_mae']:.2f}",
                       help="Mean |predicted away xG − actual away goals|.")
        row2[2].metric("Total xG MAE", f"{summary['total_mae']:.2f}")

    st.plotly_chart(
        charts.xg_scatter(finished, match_xg if has_actual_xg else None),
        use_container_width=True,
    )

    st.subheader(f"Per-match xG breakdown ({finished.height})")
    st.dataframe(_xg_table(finished, match_xg), use_container_width=True, hide_index=True)


def _xg_table(finished: pl.DataFrame, match_xg: pl.DataFrame) -> object:
    """Per-match xG table with combined H-A columns for readability."""

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
        ah = int(r["home_goals"])
        aa = int(r["away_goals"])
        total_err = abs(ph - ah) + abs(pa - aa)
        row: dict[str, object] = {
            "Fixture": f"{r['home_team']} vs {r['away_team']}",
            "Predicted xG": f"{ph:.2f} - {pa:.2f}",
            "Score": f"{ah} - {aa}",
        }
        if has_xg and r.get("home_xg") is not None:
            fh = float(r["home_xg"])
            fa = float(r["away_xg"])
            row["Actual xG"] = f"{fh:.2f} - {fa:.2f}"
            row["Model err"] = f"{abs(ph - fh) + abs(pa - fa):.2f}"
            row["Luck"] = f"{abs(fh - ah) + abs(fa - aa):.2f}"
        row["Total err"] = f"{total_err:.2f}"
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
