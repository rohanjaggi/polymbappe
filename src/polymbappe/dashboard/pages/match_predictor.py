"""Page 3 — Match Predictor (spec section 6.1).

Scoped to the matches that are actually happening: the page lists every scheduled
tournament fixture with its model H/D/A probabilities (upcoming), and every fixture
that already has a recorded result (finished), with the scoreline and whether the
model's favoured outcome matched reality. Users no longer pick arbitrary team pairs —
only real fixtures from ``match_predictions.parquet`` are shown. ``streamlit`` is
imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts

#: Display name for each H/D/A outcome key, resolved per fixture against the team names.
_OUTCOME_TEAM = {"home": "home_team", "away": "away_team"}


def _outcome_label(record: dict[str, object], outcome: str) -> str:
    """Human label for an outcome key (``home``/``draw``/``away``) of one fixture."""

    if outcome == "draw":
        return "Draw"
    return str(record[_OUTCOME_TEAM[outcome]])


def _fixture_label(record: dict[str, object]) -> str:
    """``"Home vs Away"`` label for a fixture record."""

    return f"{record['home_team']} vs {record['away_team']}"


def render(settings: Settings) -> None:
    """Render the Match Predictor page (spec 6.1, page 3)."""

    import streamlit as st

    st.header("Match Predictor")

    tab_group, tab_r32 = st.tabs(["Group Stage", "Round of 32"])

    with tab_group:
        _render_group_stage(st, settings)

    with tab_r32:
        _render_r32(st, settings)


def _render_group_stage(st: object, settings: Settings) -> None:
    """Group-stage tab: upcoming and finished group fixtures."""

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info(
            "No match predictions yet. Run `polymbappe simulate`/`report` to populate the "
            "dashboard."
        )
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    upcoming, finished = data.split_fixtures(match_df, results)

    st.caption(
        "Only scheduled tournament fixtures are shown. H/D/A probabilities come from the "
        "calibration pipeline (spec 3.6); finished matches are joined against recorded "
        "results."
    )

    _render_upcoming(st, upcoming)
    st.divider()
    _render_finished(st, finished)


def _render_upcoming(st: object, upcoming: pl.DataFrame) -> None:
    """Upcoming fixtures: full probability table plus a per-fixture breakdown."""

    st.subheader(f"Upcoming fixtures ({upcoming.height})")
    if upcoming.is_empty():
        st.info("No upcoming fixtures — every scheduled match already has a recorded result.")
        return

    groups = ["All"] + sorted(upcoming["group"].unique().to_list())
    chosen_group = st.selectbox("Filter by group", groups, key="upcoming_group")
    view = upcoming if chosen_group == "All" else upcoming.filter(pl.col("group") == chosen_group)

    st.dataframe(_probability_table(view), use_container_width=True, hide_index=True)

    labels = [_fixture_label(r) for r in view.iter_rows(named=True)]
    choice = st.selectbox("Inspect a fixture", labels, key="upcoming_fixture")
    record = view.row(labels.index(choice), named=True)
    _render_fixture_detail(st, record)


def _render_finished(st: object, finished: pl.DataFrame) -> None:
    """Finished matches: recorded scoreline vs. the model's pre-match probabilities."""

    st.subheader(f"Finished matches ({finished.height})")
    if finished.is_empty():
        st.info(
            "No finished matches recorded yet. Ingest results (`polymbappe ingest`) as the "
            "tournament progresses."
        )
        return

    hits = int(finished["model_correct"].sum())
    st.metric("Model top-pick accuracy", f"{hits}/{finished.height}")

    st.dataframe(_results_table(finished), use_container_width=True, hide_index=True)

    labels = [_fixture_label(r) for r in finished.iter_rows(named=True)]
    choice = st.selectbox("Inspect a finished match", labels, key="finished_fixture")
    record = finished.row(labels.index(choice), named=True)
    _render_fixture_detail(st, record, finished=True)


def _render_fixture_detail(
    st: object, record: dict[str, object], *, finished: bool = False
) -> None:
    """H/D/A bar + metrics for one fixture; annotates the realized result when finished."""

    home = str(record["home_team"])
    away = str(record["away_team"])
    home_prob = float(record["model_home"])
    draw_prob = float(record["model_draw"])
    away_prob = float(record["model_away"])

    st.plotly_chart(
        charts.hda_bar(home_prob, draw_prob, away_prob, home=home, away=away),
        use_container_width=True,
    )

    cols = st.columns(3)
    cols[0].metric(f"{home} win", f"{home_prob:.1%}")
    cols[1].metric("Draw", f"{draw_prob:.1%}")
    cols[2].metric(f"{away} win", f"{away_prob:.1%}")

    if "exp_home_goals" in record and "exp_away_goals" in record:
        st.caption(
            f"Expected scoreline: {float(record['exp_home_goals']):.2f} – "
            f"{float(record['exp_away_goals']):.2f}"
        )

    if finished:
        actual = _outcome_label(record, str(record["actual_outcome"]))
        verdict = (
            "✅ model's top pick was correct" if record["model_correct"] else "❌ model missed"
        )
        st.markdown(
            f"**Final score: {home} {int(record['home_goals'])} – "
            f"{int(record['away_goals'])} {away}**  →  {actual}  ·  {verdict}"
        )


def _probability_table(view: pl.DataFrame) -> object:
    """Pandas display frame of upcoming fixtures and their H/D/A probabilities."""

    rows = []
    for r in view.iter_rows(named=True):
        rows.append(
            {
                "Group": r["group"],
                "Fixture": _fixture_label(r),
                f"P({r['home_team']})": f"{float(r['model_home']):.1%}",
                "P(Draw)": f"{float(r['model_draw']):.1%}",
                f"P({r['away_team']})": f"{float(r['model_away']):.1%}",
                "Favourite": _outcome_label(r, str(r["model_pick"])),
            }
        )
    return pl.DataFrame(rows).to_pandas()


def _results_table(finished: pl.DataFrame) -> object:
    """Pandas display frame of finished matches: score, outcome, and the model's call."""

    rows = []
    for r in finished.iter_rows(named=True):
        rows.append(
            {
                "Date": str(r["date"]) if r.get("date") is not None else "",
                "Group": r["group"],
                "Fixture": _fixture_label(r),
                "Score": f"{int(r['home_goals'])} – {int(r['away_goals'])}",
                "Result": _outcome_label(r, str(r["actual_outcome"])),
                "Model pick": _outcome_label(r, str(r["model_pick"])),
                "Correct": "✅" if r["model_correct"] else "❌",
            }
        )
    return pl.DataFrame(rows).to_pandas()


def _render_r32(st: object, settings: Settings) -> None:
    """Round of 32 tab: most probable R32 matchups from simulation frequency."""

    ko_df = data.load_knockout_predictions(settings)
    if ko_df.is_empty():
        st.info(
            "No Round of 32 predictions yet. Run `polymbappe simulate` to generate them."
        )
        return

    st.caption(
        "Matchup probability is the fraction of simulations where these two teams met in the "
        "Round of 32. The R32 bracket is seeded randomly (beyond the top-4 ranked group "
        "winners), so this reflects genuine pre-tournament uncertainty."
    )

    top_n = st.slider("Show top matchups", min_value=8, max_value=min(50, ko_df.height), value=16)
    view = ko_df.head(top_n)

    table_rows = []
    for r in view.iter_rows(named=True):
        home = str(r["home_team"])
        away = str(r["away_team"])
        table_rows.append(
            {
                "Rank": int(r["rank"]),
                "Fixture": f"{home} vs {away}",
                "Match prob": f"{float(r['matchup_prob']):.1%}",
                f"P({home})": f"{float(r['model_home']):.1%}",
                "P(Draw)": f"{float(r['model_draw']):.1%}",
                f"P({away})": f"{float(r['model_away']):.1%}",
                "xG": f"{float(r['exp_home_goals']):.2f} – {float(r['exp_away_goals']):.2f}",
            }
        )
    st.dataframe(pl.DataFrame(table_rows).to_pandas(), use_container_width=True, hide_index=True)

    st.subheader("Inspect a probable matchup")
    labels = [f"#{int(r['rank'])}: {r['home_team']} vs {r['away_team']}" for r in view.iter_rows(named=True)]
    choice = st.selectbox("Choose matchup", labels, key="r32_fixture")
    record = view.row(labels.index(choice), named=True)

    home = str(record["home_team"])
    away = str(record["away_team"])
    st.plotly_chart(
        charts.hda_bar(
            float(record["model_home"]),
            float(record["model_draw"]),
            float(record["model_away"]),
            home=home,
            away=away,
        ),
        use_container_width=True,
    )
    cols = st.columns(4)
    cols[0].metric(f"{home} win", f"{float(record['model_home']):.1%}")
    cols[1].metric("Draw", f"{float(record['model_draw']):.1%}")
    cols[2].metric(f"{away} win", f"{float(record['model_away']):.1%}")
    cols[3].metric("Match probability", f"{float(record['matchup_prob']):.1%}")
