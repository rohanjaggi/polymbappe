"""Match Predictor.

Seven tabs covering every tournament stage: Group Stage, R32, R16, QF, SF,
Third Place, Final. Every knockout tab shows played fixtures (model predictions
vs actual results) plus upcoming fixtures resolved from the bracket, with
advancement probabilities.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts

_OUTCOME_TEAM = {"home": "home_team", "away": "away_team"}


def _outcome_label(record: dict[str, object], outcome: str) -> str:
    if outcome == "draw":
        return "Draw"
    return str(record[_OUTCOME_TEAM[outcome]])


def _fixture_label(record: dict[str, object]) -> str:
    return f"{record['home_team']} vs {record['away_team']}"


def render(settings: Settings) -> None:
    """Render the Match Predictor page."""

    import streamlit as st

    st.header("Match Predictor")

    tabs = st.tabs(["Group Stage", "Round of 32", "Round of 16",
                     "Quarter-Finals", "Semi-Finals", "Third Place", "Final"])

    with tabs[0]:
        _render_group_stage(st, settings)
    with tabs[1]:
        _render_r32(st, settings)
    with tabs[2]:
        _render_r16(st, settings)
    with tabs[3]:
        _render_bracket_stage(st, settings, "Quarter-final", "QF", "QF")
    with tabs[4]:
        _render_bracket_stage(st, settings, "Semi-final", "SF", "SF")
    with tabs[5]:
        _render_third_place(st, settings)
    with tabs[6]:
        _render_final(st, settings)


# ---------------------------------------------------------------------------
# Group Stage (kept from original)
# ---------------------------------------------------------------------------

def _render_group_stage(st: object, settings: Settings) -> None:
    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("Match forecasts appear here once the first predictions are published.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    all_df = data.all_fixtures_with_results(match_df, results)
    gs = all_df.filter(pl.col("group") != "KO") if "group" in all_df.columns else all_df

    st.caption(
        "All group-stage fixtures with H/D/A probabilities. "
        "Actual scores and correctness fill in as results are ingested."
    )

    played = gs.filter(pl.col("model_correct").is_not_null())
    if not played.is_empty():
        hits = int(played["model_correct"].sum())
        st.metric("Model top-pick accuracy (group stage)", f"{hits}/{played.height}")

    groups = ["All"] + sorted(gs["group"].unique().to_list())
    chosen_group = st.selectbox("Filter by group", groups, key="gs_group")
    view = gs if chosen_group == "All" else gs.filter(pl.col("group") == chosen_group)

    st.dataframe(_unified_table(view), width="stretch", hide_index=True)

    labels = [_fixture_label(r) for r in view.iter_rows(named=True)]
    if labels:
        choice = st.selectbox("Inspect a fixture", labels, key="gs_fixture")
        record = view.row(labels.index(choice), named=True)
        _render_fixture_detail(st, record, finished=record.get("model_correct") is not None)


# ---------------------------------------------------------------------------
# Round of 32
# ---------------------------------------------------------------------------

def _render_r32(st: object, settings: Settings) -> None:
    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("No match predictions yet.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    schedule = data.load_schedule(settings)
    ko = data.classify_ko_fixtures(match_df, results, schedule_df=schedule)
    if ko.is_empty():
        st.info("No knockout predictions yet.")
        return

    r32 = ko.filter(pl.col("stage") == "R32")
    if r32.is_empty():
        st.info("No R32 fixtures found.")
        return

    st.caption("All Round of 32 fixtures with model predictions and actual results.")

    played = r32.filter(pl.col("model_correct").is_not_null())
    if not played.is_empty():
        hits = int(played["model_correct"].sum())
        st.metric("Model top-pick accuracy (R32)", f"{hits}/{played.height}")

    st.dataframe(_ko_table(r32), width="stretch", hide_index=True)

    labels = [_fixture_label(r) for r in r32.iter_rows(named=True)]
    if labels:
        choice = st.selectbox("Inspect a R32 fixture", labels, key="r32_fixture")
        record = r32.row(labels.index(choice), named=True)
        _render_fixture_detail(st, record, finished=record.get("model_correct") is not None)


# ---------------------------------------------------------------------------
# Round of 16
# ---------------------------------------------------------------------------

def _render_r16(st: object, settings: Settings) -> None:
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))
    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)

    # Show played R16 matches from predictions
    ko = (
        data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df)
        if not match_df.is_empty()
        else pl.DataFrame()
    )
    _render_played_stage(st, ko, "R16", "Round of 16 fixtures")

    # Show upcoming R16 matches from bracket resolution
    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(
            schedule_df,
            ko,
            data.load_group_probabilities(settings),
            match_df,
            stage_probs=stage_df,
        )
        r16_bracket = bracket.filter(pl.col("stage") == "Round of 16")
        upcoming = r16_bracket.filter(pl.col("status") != "played")

        if not upcoming.is_empty():
            st.subheader("Upcoming R16 Fixtures")
            st.caption(
                "Teams resolved from the R32 bracket."
                " Some slots may show 'TBD' for unresolved R32 draws."
            )
            _render_bracket_table(st, upcoming, stage_df, full_bracket=bracket)

    # Stage probabilities
    if not stage_df.is_empty():
        st.subheader("R16 Advancement Probabilities")
        r16_probs = stage_df.filter(pl.col("R16") > 0).sort("R16", descending=True).head(16)
        if not r16_probs.is_empty():
            st.dataframe(
                r16_probs.select(["team", "R16", "QF", "SF", "FINAL", "champion"]).to_pandas(),
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# QF / SF bracket stages
# ---------------------------------------------------------------------------

def _render_bracket_stage(
    st: object, settings: Settings, schedule_stage: str, prob_col: str, ko_stage: str
) -> None:
    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))

    ko = (
        data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df)
        if not match_df.is_empty()
        else pl.DataFrame()
    )

    _render_played_stage(st, ko, ko_stage, f"{schedule_stage}s")

    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(
            schedule_df,
            ko,
            data.load_group_probabilities(settings),
            match_df,
            stage_probs=stage_df,
        )
        upcoming = bracket.filter(
            (pl.col("stage") == schedule_stage) & (pl.col("status") != "played")
        )
        if not upcoming.is_empty():
            st.subheader(f"Upcoming {schedule_stage} Fixtures")
            st.caption(
                "Teams are resolved from prior round results;"
                " unresolved slots list their possible occupants."
            )
            _render_bracket_table(st, upcoming, stage_df, full_bracket=bracket)

    if not stage_df.is_empty():
        st.subheader(f"Most Likely to Reach {schedule_stage}s")
        probs = stage_df.filter(pl.col(prob_col) > 0).sort(prob_col, descending=True).head(10)
        if not probs.is_empty():
            cols = list(dict.fromkeys(["team", prob_col, "SF", "FINAL", "champion"]))
            st.dataframe(
                probs.select(cols).to_pandas(),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(f"No team still has a chance of reaching the {schedule_stage}.")


def _render_played_stage(st: object, ko: pl.DataFrame, ko_stage: str, label: str) -> None:
    """Played fixtures for one knockout round: accuracy metric, table, inspector."""

    if ko.is_empty() or "stage" not in ko.columns:
        return
    played = ko.filter(pl.col("stage") == ko_stage)
    if played.is_empty():
        return

    st.caption(f"Played {label} with model predictions and actual results.")
    with_result = played.filter(pl.col("model_correct").is_not_null())
    if not with_result.is_empty():
        hits = int(with_result["model_correct"].sum())
        st.metric(f"Model top-pick accuracy ({ko_stage})", f"{hits}/{with_result.height}")

    st.dataframe(_ko_table(played), width="stretch", hide_index=True)

    labels = [_fixture_label(r) for r in played.iter_rows(named=True)]
    choice = st.selectbox("Inspect a fixture", labels, key=f"{ko_stage.lower()}_fixture")
    record = played.row(labels.index(choice), named=True)
    _render_fixture_detail(st, record, finished=record.get("model_correct") is not None)


# ---------------------------------------------------------------------------
# Third place & Final
# ---------------------------------------------------------------------------

def _render_third_place(st: object, settings: Settings) -> None:
    """The third-place play-off: played result, or its projected bracket slot."""

    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))

    ko = (
        data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df)
        if not match_df.is_empty()
        else pl.DataFrame()
    )

    played = not ko.is_empty() and not ko.filter(pl.col("stage") == "TP").is_empty()
    _render_played_stage(st, ko, "TP", "Third-place match")

    shown_upcoming = False
    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(
            schedule_df,
            ko,
            data.load_group_probabilities(settings),
            match_df,
            stage_probs=stage_df,
        )
        upcoming = bracket.filter(
            (pl.col("stage") == "Match for third place") & (pl.col("status") != "played")
        )
        if not upcoming.is_empty():
            st.caption("Third-place match bracket slot.")
            _render_bracket_table(st, upcoming, stage_df, full_bracket=bracket)
            shown_upcoming = True

    if not played and not shown_upcoming:
        st.info("The third-place match appears once the semi-finals are decided.")


def _render_final(st: object, settings: Settings) -> None:
    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))

    ko = (
        data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df)
        if not match_df.is_empty()
        else pl.DataFrame()
    )

    _render_played_stage(st, ko, "F", "Final")

    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(
            schedule_df,
            ko,
            data.load_group_probabilities(settings),
            match_df,
            stage_probs=stage_df,
        )
        upcoming = bracket.filter(
            (pl.col("stage") == "Final") & (pl.col("status") != "played")
        )
        if not upcoming.is_empty():
            st.caption("Final bracket slot.")
            _render_bracket_table(st, upcoming, stage_df, full_bracket=bracket)

    if not stage_df.is_empty():
        st.subheader("Championship Probabilities")
        champs = stage_df.filter(pl.col("champion") > 0).sort("champion", descending=True)
        if not champs.is_empty():
            st.plotly_chart(charts.trophy_bar(stage_df, n=10), width="stretch")
            st.dataframe(
                champs.select(["team", "FINAL", "champion"]).to_pandas(),
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Stage-probability column giving P(team advances past a match at this stage).
_ADVANCE_COL = {
    "Round of 32": "R16",
    "Round of 16": "QF",
    "Quarter-final": "SF",
    "Semi-final": "FINAL",
    "Final": "champion",
}


def _render_bracket_table(
    st: object,
    bracket_df: pl.DataFrame,
    stage_df: pl.DataFrame,
    full_bracket: pl.DataFrame | None = None,
) -> None:
    """Render a bracket-stage table with resolved teams and advancement probabilities.

    "Advance %" is the probability of surviving *this* match, i.e. of reaching the
    next stage (winning it, for the final) — not of having reached the current one.
    """

    prob_rows: dict[str, dict[str, object]] = {}
    if not stage_df.is_empty():
        for r in stage_df.iter_rows(named=True):
            prob_rows[str(r["team"])] = r

    def _advance_prob(team: str, stage: str) -> str:
        col = _ADVANCE_COL.get(stage)
        row = prob_rows.get(team)
        if col is None or row is None or col not in row:
            return "—"
        return f"{float(row[col]):.0%}"

    match_teams: dict[int, tuple[str, str]] = {}
    if full_bracket is not None:
        for r in full_bracket.iter_rows(named=True):
            mn = r.get("match_number")
            if mn is not None:
                h = r.get("home_resolved") or r.get("home_code", "TBD")
                a = r.get("away_resolved") or r.get("away_code", "TBD")
                match_teams[int(mn)] = (str(h) if h else "TBD", str(a) if a else "TBD")

    import re

    def _resolve_code(code: str, depth: int = 0) -> str:
        if depth > 3:
            return code
        m = re.match(r"^[WL](\d+)$", code)
        if not m or full_bracket is None:
            return code
        mn = int(m.group(1))
        if mn not in match_teams:
            return code
        t1, t2 = match_teams[mn]
        t1 = _resolve_code(t1, depth + 1) if re.match(r"^[WL]\d+$", t1) else t1
        t2 = _resolve_code(t2, depth + 1) if re.match(r"^[WL]\d+$", t2) else t2
        resolved = f"{t1} / {t2}"
        if resolved.count("/") > 3:
            return "TBD"
        return resolved

    def _short_city(city: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", city)

    rows = []
    for r in bracket_df.iter_rows(named=True):
        h = r.get("home_resolved") or _resolve_code(r.get("home_code") or "TBD")
        a = r.get("away_resolved") or _resolve_code(r.get("away_code") or "TBD")
        stage = str(r.get("stage", ""))
        h_is_team = "/" not in str(h) and str(h) != "TBD"
        a_is_team = "/" not in str(a) and str(a) != "TBD"
        rows.append({
            "Date": str(r["date"]),
            "City": _short_city(str(r["city"])),
            "Home": str(h) if h else "TBD",
            "Away": str(a) if a else "TBD",
            "Home Advance %": _advance_prob(str(h), stage) if h_is_team else "—",
            "Away Advance %": _advance_prob(str(a), stage) if a_is_team else "—",
            "Status": str(r.get("status", "tbd")).title(),
        })

    st.dataframe(pl.DataFrame(rows).to_pandas(), width="stretch", hide_index=True)


def _render_fixture_detail(
    st: object, record: dict[str, object], *, finished: bool = False
) -> None:
    home = str(record["home_team"])
    away = str(record["away_team"])
    home_prob = float(record["model_home"])
    draw_prob = float(record["model_draw"])
    away_prob = float(record["model_away"])

    st.plotly_chart(
        charts.hda_bar(home_prob, draw_prob, away_prob, home=home, away=away),
        width="stretch",
    )

    cols = st.columns(3)
    cols[0].metric(f"{home} win", f"{home_prob:.1%}")
    cols[1].metric("Draw", f"{draw_prob:.1%}")
    cols[2].metric(f"{away} win", f"{away_prob:.1%}")

    if "exp_home_goals" in record and "exp_away_goals" in record:
        ehg = record["exp_home_goals"]
        eag = record["exp_away_goals"]
        if ehg is not None and eag is not None:
            st.caption(f"Expected scoreline: {float(ehg):.2f} – {float(eag):.2f}")

    if finished and record.get("actual_outcome") is not None:
        actual = _outcome_label(record, str(record["actual_outcome"]))
        verdict = (
            "correct" if record["model_correct"] else "missed"
        )
        st.markdown(
            f"**Final score: {home} {int(record['home_goals'])} – "
            f"{int(record['away_goals'])} {away}**  ·  {actual}  ·  Model {verdict}"
        )


def _unified_table(view: pl.DataFrame) -> object:
    rows = []
    for r in view.iter_rows(named=True):
        has_result = r.get("home_goals") is not None
        actual_key = str(r["actual_outcome"]) if has_result else None
        rows.append({
            "Grp": r["group"],
            "Fixture": _fixture_label(r),
            "P(Home)": f"{float(r['model_home']):.0%}",
            "P(Draw)": f"{float(r['model_draw']):.0%}",
            "P(Away)": f"{float(r['model_away']):.0%}",
            "Predicted xG": (
                f"{float(r['exp_home_goals']):.2f} – {float(r['exp_away_goals']):.2f}"
                if r.get("exp_home_goals") is not None else ""
            ),
            "Score": f"{int(r['home_goals'])} – {int(r['away_goals'])}" if has_result else "—",
            "Result": _outcome_label(r, actual_key) if has_result else "—",
            "Correct": ("✅" if r["model_correct"] else "❌") if has_result else "⏳",
        })
    return pl.DataFrame(rows).to_pandas()


def _ko_table(view: pl.DataFrame) -> object:
    rows = []
    for r in view.iter_rows(named=True):
        has_result = r.get("home_goals") is not None and r.get("actual_outcome") is not None
        actual_key = str(r["actual_outcome"]) if has_result else None
        rows.append({
            "Fixture": _fixture_label(r),
            "P(Home)": f"{float(r['model_home']):.0%}",
            "P(Draw)": f"{float(r['model_draw']):.0%}",
            "P(Away)": f"{float(r['model_away']):.0%}",
            "Predicted xG": (
                f"{float(r['exp_home_goals']):.2f} – {float(r['exp_away_goals']):.2f}"
                if r.get("exp_home_goals") is not None else ""
            ),
            "Score": (
                f"{int(r['home_goals'])} – {int(r['away_goals'])}"
                if r.get("home_goals") is not None else "—"
            ),
            "Result": (
                ("Draw (ET/pens)" if actual_key == "draw" else _outcome_label(r, actual_key))
                if has_result else "—"
            ),
            "Correct": ("✅" if r["model_correct"] else "❌") if has_result else "⏳",
        })
    return pl.DataFrame(rows).to_pandas()
