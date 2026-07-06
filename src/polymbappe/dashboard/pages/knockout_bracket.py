"""Knockout Bracket page (spec section 6.1).

Renders the *real* 2026 knockout draw in bracket format (R32 → Final) from
``knockout_bracket.parquet``: concrete R32/R16 fixtures anchored to the ingested schedule, and
QF → Final slots projected forward through the fixed bracket. For every game the page shows the
model's advance probabilities and the FT/ET/penalties decided-phase split, and folds in actual
scorelines as ``polymbappe ingest`` records knockout results.

Because future slots are not yet drawn, each carries its **most-probable** occupants in the tree;
the per-round drill-down expands any slot into *all* the matchups it can still produce — the
expected outcomes of every possible future game. ``streamlit`` is imported lazily.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings
from polymbappe.dashboard import data
from polymbappe.dashboard.components import charts

#: Human labels for each knockout round key.
_ROUND_LABELS: dict[str, str] = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "FINAL": "Final",
}

#: A rank-1 matchup at/above this probability is treated as a concrete (already-drawn) fixture.
_CONCRETE_PROB = 0.999


def render(settings: Settings) -> None:
    """Render the Knockout Bracket page (spec 6.1)."""

    import streamlit as st

    st.header("Knockout Bracket")

    bracket = data.load_knockout_bracket(settings)
    if bracket.is_empty():
        st.info(
            "No knockout bracket yet. Run `polymbappe simulate` once the knockout schedule is "
            "ingested to generate `knockout_bracket.parquet`."
        )
        return

    played = _played_lookup(data.knockout_results(data.load_recorded_results(settings)))

    st.caption(
        "Anchored to the real draw: R32/R16 fixtures come from the ingested schedule, and later "
        "rounds are projected through the fixed bracket. Advance probability and the "
        "FT/ET/penalties split come from the strength model; already-drawn slots show their "
        "concrete teams, future slots their **most-probable** occupants. Played ties show the "
        "actual scoreline and who advanced."
    )
    if played:
        st.success(f"{len(played)} knockout result(s) ingested and folded into the bracket.")

    _render_tree(st, bracket, played)
    st.divider()
    _render_drilldowns(st, bracket, played)


# -- played-result integration ------------------------------------------------


def _played_lookup(results: pl.DataFrame) -> dict[frozenset[str], dict[str, object]]:
    """Order-independent ``{frozenset(teams): result}`` map for played knockout ties."""

    lookup: dict[frozenset[str], dict[str, object]] = {}
    if results.is_empty():
        return lookup
    for r in results.iter_rows(named=True):
        home, away = str(r["home_team"]), str(r["away_team"])
        hg, ag = r.get("home_goals"), r.get("away_goals")
        if hg is None or ag is None:
            continue
        hg, ag = int(hg), int(ag)
        advanced = home if hg > ag else away if ag > hg else None
        lookup[frozenset((home, away))] = {
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "advanced": advanced,
        }
    return lookup


def _result_for(played: dict[frozenset[str], dict[str, object]], a: str, b: str) -> dict[str, object] | None:
    return played.get(frozenset((a, b)))


def _score_text(res: dict[str, object]) -> str:
    return (
        f"{res['home_team']} {int(res['home_goals'])} – "
        f"{int(res['away_goals'])} {res['away_team']}"
    )


def _is_concrete(row: dict[str, object]) -> bool:
    """A drawn fixture (both teams known) vs a projected future slot."""

    return float(row["matchup_prob"]) >= _CONCRETE_PROB


# -- bracket tree -------------------------------------------------------------


def _render_tree(st: object, bracket: pl.DataFrame, played: dict[frozenset[str], dict[str, object]]) -> None:
    """Bracket-style tree: one column per round, one card per real fixture/slot in bracket order."""

    st.subheader("Bracket")
    rounds = [r for r in data.KNOCKOUT_ROUND_ORDER if not data.bracket_slots(bracket, r).is_empty()]
    if not rounds:
        return
    columns = st.columns(len(rounds))
    for col, round_name in zip(columns, rounds, strict=False):
        with col:
            st.markdown(f"**{_ROUND_LABELS.get(round_name, round_name)}**")
            for r in data.bracket_slots(bracket, round_name).iter_rows(named=True):
                st.markdown(_card_markdown(r, played))


def _card_markdown(r: dict[str, object], played: dict[frozenset[str], dict[str, object]]) -> str:
    """One slot card: the (concrete or most-likely) teams with advance %, plus any real result."""

    a, b = str(r["team_a"]), str(r["team_b"])
    pa, pb = float(r["p_a_advance"]), float(r["p_b_advance"])
    res = _result_for(played, a, b) if _is_concrete(r) else None
    if res is not None:
        adv = res["advanced"]
        tail = f"✅ {adv}" if adv else "pens"
        return (
            f"`{a}` {int(res['home_goals'])}–{int(res['away_goals'])} `{b}`  \n"
            f"↳ {tail}\n\n---"
        )
    lead_a = "▸ " if pa >= pb else "  "
    lead_b = "▸ " if pb > pa else "  "
    lines = [f"`{lead_a}{a}`  {pa:.0%}", f"`{lead_b}{b}`  {pb:.0%}"]
    if not _is_concrete(r):  # projected slot: note how likely this exact matchup is
        lines.append(f"_~{float(r['matchup_prob']):.0%} likely pairing_")
    return "  \n".join(lines) + "\n\n---"


# -- per-round drill-down ------------------------------------------------------


def _render_drilldowns(
    st: object, bracket: pl.DataFrame, played: dict[frozenset[str], dict[str, object]]
) -> None:
    """One expander per round: the expected slots, then any slot's full set of possible matchups."""

    st.subheader("Round-by-round predictions")
    for round_name in data.KNOCKOUT_ROUND_ORDER:
        slots = data.bracket_slots(bracket, round_name)
        if slots.is_empty():
            continue
        label = _ROUND_LABELS.get(round_name, round_name)
        with st.expander(f"{label} — {slots.height} fixture(s)", expanded=round_name in ("R16", "QF")):
            st.dataframe(_slots_table(slots, played), use_container_width=True, hide_index=True)

            # Any fixture can be expanded into all the matchups it can still produce.
            options = list(slots.iter_rows(named=True))
            labels = [_slot_label(r, played) for r in options]
            choice = st.selectbox("Inspect a fixture", labels, key=f"ko_{round_name}")
            chosen = options[labels.index(choice)]
            _render_slot_detail(st, bracket, int(chosen["match_number"]), played)


def _slot_label(r: dict[str, object], played: dict[frozenset[str], dict[str, object]]) -> str:
    a, b = str(r["team_a"]), str(r["team_b"])
    if _is_concrete(r):
        return f"{a} vs {b}"
    return f"{a} vs {b} (projected)"


def _slots_table(slots: pl.DataFrame, played: dict[frozenset[str], dict[str, object]]) -> object:
    """Expected fixture per slot: teams, advance %, FT/ET/pens, and actual result if played."""

    rows = []
    for r in slots.iter_rows(named=True):
        a, b = str(r["team_a"]), str(r["team_b"])
        res = _result_for(played, a, b) if _is_concrete(r) else None
        rows.append(
            {
                "Fixture": (
                    f"{a} vs {b}" if _is_concrete(r) else f"{a} vs {b} ({float(r['matchup_prob']):.0%})"
                ),
                f"P({a} adv)": f"{float(r['p_a_advance']):.0%}",
                f"P({b} adv)": f"{float(r['p_b_advance']):.0%}",
                "FT": f"{float(r['p_decided_reg']):.0%}",
                "ET": f"{float(r['p_decided_et']):.0%}",
                "Pens": f"{float(r['p_decided_pens']):.0%}",
                "xG": f"{float(r['exp_a_goals']):.2f} – {float(r['exp_b_goals']):.2f}",
                "Actual": _score_text(res) if res is not None else "—",
            }
        )
    return pl.DataFrame(rows).to_pandas()


def _render_slot_detail(
    st: object,
    bracket: pl.DataFrame,
    match_number: int,
    played: dict[frozenset[str], dict[str, object]],
) -> None:
    """All possible matchups at one fixture + the model breakdown for a chosen one."""

    candidates = data.bracket_slot_candidates(bracket, match_number)
    if candidates.height > 1:
        st.caption(f"{candidates.height} possible matchups can still occur at this slot:")
        st.dataframe(_candidates_table(candidates), use_container_width=True, hide_index=True)

    labels = [
        f"{r['team_a']} vs {r['team_b']} ({float(r['matchup_prob']):.0%})"
        for r in candidates.iter_rows(named=True)
    ]
    choice = st.selectbox("Model breakdown for matchup", labels, key=f"cand_{match_number}")
    record = candidates.row(labels.index(choice), named=True)
    _render_matchup_detail(st, record, played)


def _candidates_table(candidates: pl.DataFrame) -> object:
    rows = []
    for r in candidates.iter_rows(named=True):
        a, b = str(r["team_a"]), str(r["team_b"])
        rows.append(
            {
                "Matchup": f"{a} vs {b}",
                "Occurs": f"{float(r['matchup_prob']):.0%}",
                f"P({a} adv)": f"{float(r['p_a_advance']):.0%}",
                f"P({b} adv)": f"{float(r['p_b_advance']):.0%}",
                "FT/ET/Pens": (
                    f"{float(r['p_decided_reg']):.0%} / {float(r['p_decided_et']):.0%} / "
                    f"{float(r['p_decided_pens']):.0%}"
                ),
            }
        )
    return pl.DataFrame(rows).to_pandas()


def _render_matchup_detail(
    st: object, record: dict[str, object], played: dict[frozenset[str], dict[str, object]]
) -> None:
    """H/D/A bar + advance metrics + FT/ET/pens split for a single knockout matchup."""

    a, b = str(record["team_a"]), str(record["team_b"])
    st.plotly_chart(
        charts.hda_bar(
            float(record["model_a"]),
            float(record["model_draw"]),
            float(record["model_b"]),
            home=a,
            away=b,
        ),
        use_container_width=True,
    )

    cols = st.columns(3)
    cols[0].metric(f"{a} advances", f"{float(record['p_a_advance']):.1%}")
    cols[1].metric(f"{b} advances", f"{float(record['p_b_advance']):.1%}")
    cols[2].metric("Matchup occurs", f"{float(record['matchup_prob']):.1%}")

    st.plotly_chart(
        charts.phase_decided_bar(
            float(record["p_decided_reg"]),
            float(record["p_decided_et"]),
            float(record["p_decided_pens"]),
        ),
        use_container_width=True,
    )
    st.caption(
        f"Expected scoreline (regulation): {float(record['exp_a_goals']):.2f} – "
        f"{float(record['exp_b_goals']):.2f}"
    )

    res = _result_for(played, a, b)
    if res is not None:
        adv = res["advanced"]
        verdict = f"✅ **{adv}** advanced" if adv else "decided on penalties"
        st.markdown(f"**Final: {_score_text(res)}**  →  {verdict}")
