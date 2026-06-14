"""Page 2 — Team Deep Dive (spec section 6.1).

Team selector, stage-reaching probability waterfall, and group-finish breakdown for
a single team. ``streamlit`` is imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts


def render(settings: Settings) -> None:
    """Render the Team Deep Dive page (spec 6.1, page 2)."""

    import streamlit as st

    st.header("Team Deep Dive")

    stage_df = data.load_stage_probabilities(settings)
    group_df = data.load_group_probabilities(settings)

    teams = data.available_teams(stage_df)
    if not teams:
        st.info("No simulation results yet. Run `polymbappe simulate` to populate the dashboard.")
        return

    team = st.selectbox("Select a team", teams)

    st.subheader("Stage-reaching probability waterfall")
    stage_probs = data.team_stage_row(stage_df, team)
    st.plotly_chart(charts.stage_waterfall(stage_probs, team=team), use_container_width=True)

    st.subheader("Stage-reaching probabilities")
    st.dataframe(
        stage_df.filter(stage_df["team"] == team).to_pandas(),
        use_container_width=True,
    )

    if not group_df.is_empty() and "team" in group_df.columns:
        st.subheader("Group-finish probabilities")
        team_group = group_df.filter(group_df["team"] == team)
        if team_group.is_empty():
            st.caption("No group-finish data for this team.")
        else:
            st.dataframe(team_group.to_pandas(), use_container_width=True)

    _render_group_predictions(st, settings, team)


def _render_group_predictions(st: object, settings: Settings, team: str) -> None:
    """Group-stage fixture predictions for ``team``, oriented to its perspective.

    Reuses the Match Predictor data path (:func:`data.load_match_predictions` +
    :func:`data.split_fixtures`) so the H/D/A probabilities and the played/upcoming
    split stay consistent across pages. Each of the team's three group fixtures is
    re-pivoted so probabilities read as ``P(team win) / P(draw) / P(opponent win)``,
    and a projected-points metric blends actual points for played matches with
    expected points (3·P(win) + P(draw)) for upcoming ones.
    """

    st.subheader("Group-stage fixture predictions")

    match_df = data.load_match_predictions(settings)
    if match_df.is_empty():
        st.caption(
            "No match predictions yet. Run `polymbappe simulate`/`report` to populate them."
        )
        return

    results = data.tournament_results(data.load_recorded_results(settings))
    upcoming, finished = data.split_fixtures(match_df, results)
    team_upcoming = _team_fixtures(upcoming, team)
    team_finished = _team_fixtures(finished, team)

    if team_upcoming.is_empty() and team_finished.is_empty():
        st.caption(f"No group-stage fixtures found for {team}.")
        return

    rows: list[dict[str, object]] = []
    projected_points = 0.0
    for record in team_finished.iter_rows(named=True):
        verdict, pts, score = _result_for_team(record, team)
        projected_points += pts
        rows.append(_prediction_row(record, team, status="Played", result=verdict, score=score))
    for record in team_upcoming.iter_rows(named=True):
        p_team, p_draw, _ = _team_perspective(record, team)
        projected_points += 3.0 * p_team + p_draw
        rows.append(_prediction_row(record, team, status="Upcoming"))

    st.caption(
        f"H/D/A probabilities for every scheduled group-stage fixture, oriented to {team}. "
        "Played matches show the recorded result; projected points blend actual points "
        "(played) with expected points 3·P(win)+P(draw) (upcoming)."
    )
    st.metric(f"{team} projected group points", f"{projected_points:.1f}")
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)


def _team_fixtures(fixtures: pl.DataFrame, team: str) -> pl.DataFrame:
    """Fixtures in which ``team`` plays (home or away)."""

    if fixtures.is_empty():
        return fixtures
    return fixtures.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))


def _team_perspective(record: dict[str, object], team: str) -> tuple[float, float, float]:
    """Return ``(P(team win), P(draw), P(opponent win))`` for one fixture."""

    at_home = record["home_team"] == team
    p_team = float(record["model_home"] if at_home else record["model_away"])
    p_opp = float(record["model_away"] if at_home else record["model_home"])
    return p_team, float(record["model_draw"]), p_opp


def _favoured_team(record: dict[str, object]) -> str:
    """Name of the team the model favours (or ``"Draw"``) for one fixture."""

    pick = str(record["model_pick"])
    if pick == "draw":
        return "Draw"
    return str(record["home_team"] if pick == "home" else record["away_team"])


def _result_for_team(
    record: dict[str, object], team: str
) -> tuple[str, int, str]:
    """``(verdict, points, score)`` for a finished fixture, from ``team``'s view."""

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
    """One team-perspective display row for the group-stage predictions table."""

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
