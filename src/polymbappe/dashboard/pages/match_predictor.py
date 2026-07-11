"""Page 2 — Match Predictor.

Six tabs covering every tournament stage: Group Stage, R32, R16, QF, SF, Final.
Group and R32 show predictions with actual results. R16 shows played + upcoming.
QF/SF/Final show bracket structure with stage probabilities.
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
                     "Quarter-Finals", "Semi-Finals", "Final"])

    with tabs[0]:
        _render_group_stage(st, settings)
    with tabs[1]:
        _render_r32(st, settings)
    with tabs[2]:
        _render_r16(st, settings)
    with tabs[3]:
        _render_bracket_stage(st, settings, "Quarter-final", "QF")
    with tabs[4]:
        _render_bracket_stage(st, settings, "Semi-final", "SF")
    with tabs[5]:
        _render_final(st, settings)


# ---------------------------------------------------------------------------
# Group Stage (kept from original)
# ---------------------------------------------------------------------------

def _render_group_stage(st: object, settings: Settings) -> None:
    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.info("No match predictions yet. Run `polymbappe simulate`/`report` to populate.")
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

    st.dataframe(_unified_table(view), use_container_width=True, hide_index=True)

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

    st.dataframe(_ko_table(r32), use_container_width=True, hide_index=True)

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
    ko = data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df) if not match_df.is_empty() else pl.DataFrame()
    r16_played = ko.filter(pl.col("stage") == "R16") if not ko.is_empty() and "stage" in ko.columns else pl.DataFrame()

    if not r16_played.is_empty():
        st.caption("Played Round of 16 fixtures with predictions and results.")
        played_with_result = r16_played.filter(pl.col("model_correct").is_not_null())
        if not played_with_result.is_empty():
            hits = int(played_with_result["model_correct"].sum())
            st.metric("Model top-pick accuracy (R16)", f"{hits}/{played_with_result.height}")

        st.dataframe(_ko_table(r16_played), use_container_width=True, hide_index=True)

        labels = [_fixture_label(r) for r in r16_played.iter_rows(named=True)]
        if labels:
            choice = st.selectbox("Inspect a R16 fixture", labels, key="r16_fixture")
            record = r16_played.row(labels.index(choice), named=True)
            _render_fixture_detail(st, record, finished=record.get("model_correct") is not None)

    # Show upcoming R16 matches from bracket resolution
    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(schedule_df, ko, data.load_group_probabilities(settings), match_df, stage_probs=data.load_stage_probabilities(settings))
        r16_bracket = bracket.filter(pl.col("stage") == "Round of 16")
        upcoming = r16_bracket.filter(pl.col("status") != "played")

        if not upcoming.is_empty():
            st.subheader("Upcoming R16 Fixtures")
            st.caption("Teams resolved from the R32 bracket. Some slots may show 'TBD' for unresolved R32 draws.")
            _render_bracket_table(st, upcoming, stage_df, "R16", full_bracket=bracket)

    # Stage probabilities
    if not stage_df.is_empty():
        st.subheader("R16 Advancement Probabilities")
        r16_probs = stage_df.filter(pl.col("R16") > 0).sort("R16", descending=True).head(16)
        if not r16_probs.is_empty():
            st.dataframe(
                r16_probs.select(["team", "R16", "QF", "SF", "FINAL", "champion"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# QF / SF bracket stages
# ---------------------------------------------------------------------------

def _render_bracket_stage(
    st: object, settings: Settings, schedule_stage: str, prob_col: str
) -> None:
    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))

    ko = data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df) if not match_df.is_empty() else pl.DataFrame()

    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(schedule_df, ko, data.load_group_probabilities(settings), match_df, stage_probs=data.load_stage_probabilities(settings))
        stage_matches = bracket.filter(pl.col("stage") == schedule_stage)

        if not stage_matches.is_empty():
            st.caption(f"Bracket structure for the {schedule_stage}s. Teams are resolved from prior round results.")
            _render_bracket_table(st, stage_matches, stage_df, prob_col, full_bracket=bracket)

    if not stage_df.is_empty():
        st.subheader(f"Most Likely to Reach {schedule_stage}s")
        probs = stage_df.filter(pl.col(prob_col) > 0).sort(prob_col, descending=True).head(10)
        if not probs.is_empty():
            cols = list(dict.fromkeys(["team", prob_col, "SF", "FINAL", "champion"]))
            st.dataframe(
                probs.select(cols).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No teams with >{prob_col} probability > 0.")


# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------

def _render_final(st: object, settings: Settings) -> None:
    stage_df = data.load_stage_probabilities(settings)
    schedule_df = data.load_schedule(settings)
    match_df = data.load_match_predictions(settings)
    results = data.tournament_results(data.load_recorded_results(settings))

    ko = data.classify_ko_fixtures(match_df, results, schedule_df=schedule_df) if not match_df.is_empty() else pl.DataFrame()

    if not schedule_df.is_empty():
        bracket = data.resolve_bracket(schedule_df, ko, data.load_group_probabilities(settings), match_df, stage_probs=data.load_stage_probabilities(settings))
        final_matches = bracket.filter(
            (pl.col("stage") == "Final") | (pl.col("stage") == "Match for third place")
        )
        if not final_matches.is_empty():
            st.caption("Final and third-place match bracket slots.")
            _render_bracket_table(st, final_matches, stage_df, "FINAL", full_bracket=bracket)

    if not stage_df.is_empty():
        st.subheader("Championship Probabilities")
        champs = stage_df.filter(pl.col("champion") > 0).sort("champion", descending=True)
        if not champs.is_empty():
            st.plotly_chart(charts.trophy_bar(stage_df, n=10), use_container_width=True)
            st.dataframe(
                champs.select(["team", "FINAL", "champion"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_bracket_table(
    st: object,
    bracket_df: pl.DataFrame,
    stage_df: pl.DataFrame,
    prob_col: str,
    full_bracket: pl.DataFrame | None = None,
) -> None:
    """Render a bracket-stage table with resolved teams and advancement probabilities."""

    prob_map: dict[str, float] = {}
    if not stage_df.is_empty() and prob_col in stage_df.columns:
        for r in stage_df.iter_rows(named=True):
            prob_map[str(r["team"])] = float(r[prob_col])

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
        h = r.get("home_resolved") or _resolve_code(r.get("home_code", "TBD"))
        a = r.get("away_resolved") or _resolve_code(r.get("away_code", "TBD"))
        h_is_team = "/" not in str(h) and str(h) != "TBD"
        a_is_team = "/" not in str(a) and str(a) != "TBD"
        rows.append({
            "Date": str(r["date"]),
            "City": _short_city(str(r["city"])),
            "Home": str(h) if h else "TBD",
            "Away": str(a) if a else "TBD",
            f"Home Advance %": f"{prob_map.get(str(h), 0):.0%}" if h_is_team else "—",
            f"Away Advance %": f"{prob_map.get(str(a), 0):.0%}" if a_is_team else "—",
            "Status": str(r.get("status", "tbd")).title(),
        })

    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)


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
        use_container_width=True,
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
            "Correct": ("Yes" if r["model_correct"] else "No") if has_result else "Pending",
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
            "Result": _outcome_label(r, actual_key) if has_result else "—",
            "Correct": ("Yes" if r["model_correct"] else "No") if has_result else "Pending",
        })
    return pl.DataFrame(rows).to_pandas()
