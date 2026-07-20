"""Team Deep Dive page.

Team selector (defaulting to the champion / title favourite), stage-reaching
probabilities, group-stage fixture retrospective, and knockout journey.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Team Deep Dive page."""

    import streamlit as st

    st.header("Team Deep Dive")

    stage_df = data.load_stage_probabilities(settings)

    teams = data.available_teams(stage_df)
    if not teams:
        st.info(
            "Team-by-team forecasts appear here once the first simulation results "
            "are published."
        )
        return

    # Default to the biggest story in the data: the champion (or title favourite).
    favourite = data.champion_team(stage_df)
    if favourite is None:
        contenders = data.top_contenders(stage_df, n=1)
        favourite = str(contenders["team"][0]) if not contenders.is_empty() else None
    default_index = teams.index(favourite) if favourite in teams else 0
    team = st.selectbox("Select a team", teams, index=default_index)

    st.subheader("Stage-reaching probabilities")
    stage_probs = data.team_stage_row(stage_df, team)
    st.plotly_chart(charts.stage_waterfall(stage_probs, team=team), width="stretch")

    _render_group_predictions(st, settings, team)
    st.divider()
    _render_knockout_journey(st, settings, team)


def _render_group_predictions(st: object, settings: Settings, team: str) -> None:
    """Group-stage fixture predictions for the selected team."""

    st.subheader("Group-stage results")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.caption("No match predictions yet.")
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    upcoming, finished = data.split_fixtures(match_df, results)
    if "group" in upcoming.columns:
        upcoming = upcoming.filter(pl.col("group") != "KO")
    if "group" in finished.columns:
        finished = finished.filter(pl.col("group") != "KO")
    team_upcoming = _team_fixtures(upcoming, team)
    team_finished = _team_fixtures(finished, team)

    if team_upcoming.is_empty() and team_finished.is_empty():
        st.caption(f"No group-stage fixtures found for {team}.")
        return

    rows: list[dict[str, object]] = []
    actual_points = 0
    for record in team_finished.iter_rows(named=True):
        verdict, pts, score = _result_for_team(record, team)
        actual_points += pts
        rows.append(_prediction_row(record, team, status="Played", result=verdict, score=score))
    for record in team_upcoming.iter_rows(named=True):
        rows.append(_prediction_row(record, team, status="Upcoming"))

    predicted_df = data.predicted_group_points(match_df)
    team_pred = predicted_df.filter(pl.col("team") == team)
    predicted_points = (
        float(team_pred["predicted_points"].item()) if not team_pred.is_empty() else 0.0
    )

    standings_df = data.compute_group_standings(match_df, results)
    all_predicted = predicted_df.join(
        standings_df.select(["team", "points"]), on="team", how="inner"
    )
    overall_mae = float(
        (all_predicted["predicted_points"] - all_predicted["points"].cast(pl.Float64)).abs().mean()
    ) if not all_predicted.is_empty() else 0.0

    st.caption(
        f"H/D/A probabilities oriented to {team}. "
        "Predicted points = 3·P(win) + P(draw) summed over group-stage fixtures."
    )
    cols = st.columns(3)
    cols[0].metric("Predicted points", f"{predicted_points:.1f}")
    cols[1].metric("Actual points", str(actual_points))
    point_err = abs(predicted_points - actual_points)
    cols[2].metric(
        "Error",
        f"{point_err:.1f}",
        delta=f"Avg error across all teams: {overall_mae:.1f}",
        delta_color="off",
    )
    st.dataframe(pl.DataFrame(rows).to_pandas(), width="stretch", hide_index=True)


def _render_knockout_journey(st: object, settings: Settings, team: str) -> None:
    """Show the team's knockout path: R32 result, R16 status, future outlook."""

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    schedule = data.load_schedule(settings)
    ko = data.classify_ko_fixtures(match_df, results, schedule_df=schedule)
    if ko.is_empty() or "stage" not in ko.columns:
        return

    team_ko = ko.filter(
        (pl.col("home_team") == team) | (pl.col("away_team") == team)
    )
    if team_ko.is_empty():
        st.caption(f"{team} did not qualify for the knockout stage.")
        return

    st.subheader("Knockout Journey")

    for r in team_ko.sort("date").iter_rows(named=True):
        stage = str(r.get("stage", "KO"))
        h, a = str(r["home_team"]), str(r["away_team"])
        opp = a if h == team else h
        hg, ag = r.get("home_goals"), r.get("away_goals")

        if hg is not None and ag is not None:
            hg, ag = int(hg), int(ag)
            at_home = h == team
            team_goals = hg if at_home else ag
            opp_goals = ag if at_home else hg
            if team_goals > opp_goals:
                result_str = f"Won {team_goals}–{opp_goals}"
            elif team_goals < opp_goals:
                result_str = f"Lost {team_goals}–{opp_goals}"
            else:
                result_str = f"Drew {team_goals}–{opp_goals}"
            st.markdown(f"**{stage}**: {team} vs {opp} — {result_str}")
        else:
            st.markdown(f"**{stage}**: {team} vs {opp} — Upcoming")

    # Future outlook from stage probabilities
    stage_df = data.load_stage_probabilities(settings)
    if not stage_df.is_empty():
        team_row = stage_df.filter(pl.col("team") == team)
        if not team_row.is_empty():
            r = team_row.row(0, named=True)
            future_stages = []
            for col, label in [("QF", "Quarter-Final"), ("SF", "Semi-Final"),
                                ("FINAL", "Final"), ("champion", "Champion")]:
                prob = float(r.get(col, 0))
                if prob > 0:
                    future_stages.append(f"{label}: {prob:.0%}")
            if future_stages:
                st.caption("Advancement odds: " + " · ".join(future_stages))


def _team_fixtures(fixtures: pl.DataFrame, team: str) -> pl.DataFrame:
    if fixtures.is_empty():
        return fixtures
    return fixtures.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))


def _team_perspective(record: dict[str, object], team: str) -> tuple[float, float, float]:
    at_home = record["home_team"] == team
    p_team = float(record["model_home"] if at_home else record["model_away"])
    p_opp = float(record["model_away"] if at_home else record["model_home"])
    return p_team, float(record["model_draw"]), p_opp


def _favoured_team(record: dict[str, object]) -> str:
    pick = str(record["model_pick"])
    if pick == "draw":
        return "Draw"
    return str(record["home_team"] if pick == "home" else record["away_team"])


def _result_for_team(
    record: dict[str, object], team: str
) -> tuple[str, int, str]:
    at_home = record["home_team"] == team
    team_goals = int(record["home_goals"] if at_home else record["away_goals"])
    opp_goals = int(record["away_goals"] if at_home else record["home_goals"])
    if team_goals > opp_goals:
        return "Win", 3, f"{team_goals}–{opp_goals}"
    if team_goals < opp_goals:
        return "Loss", 0, f"{team_goals}–{opp_goals}"
    return "Draw", 1, f"{team_goals}–{opp_goals}"


def _prediction_row(
    record: dict[str, object],
    team: str,
    *,
    status: str,
    result: str = "—",
    score: str = "—",
) -> dict[str, object]:
    at_home = record["home_team"] == team
    opponent = str(record["away_team"] if at_home else record["home_team"])
    p_team, p_draw, p_opp = _team_perspective(record, team)
    return {
        "Group": record["group"],
        "Opponent": opponent,
        "Venue": "Home" if at_home else "Away",
        f"P({team})": f"{p_team:.1%}",
        "P(Draw)": f"{p_draw:.1%}",
        "P(Opp)": f"{p_opp:.1%}",
        "Model favours": _favoured_team(record),
        "Status": status,
        "Result": result,
        "Score": score,
    }
