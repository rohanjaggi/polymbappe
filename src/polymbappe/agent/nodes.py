"""The five agent nodes and the cycle orchestrator (spec section 5.2).

Scan -> Assess -> Cross-Reference -> Act -> Reflect. Each node is a plain function over
explicit inputs so the pipeline is fully testable offline; :func:`run_cycle` wires them in
sequence with the spec's conditional routing (skip non-material findings, skip already-known
statuses, log-only on insignificant shifts) and persists the decision trace.

Assessment classification uses Qwen via Ollama when available, falling back to a
deterministic keyword heuristic — critical for reliability given Qwen 9B's noise (spec 5.2
"False positive mitigation").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from polymbappe.agent.sources import NewsItem, scan_sources
from polymbappe.agent.state import AgentState

#: Severity ordering (ascending). Items must be >= "doubt" to pass Assess (spec 5.2).
SEVERITY_ORDER: tuple[str, ...] = ("non-issue", "minor", "doubt", "out")
_CONFIDENCE_PASS = {"confirmed", "likely"}


@dataclass(slots=True)
class AgentConfig:
    """Agent thresholds (spec 5.2 false-positive mitigation)."""

    player_tiers: dict[str, int] = field(default_factory=dict)
    """Map player name -> importance tier (1 = most important). Unknown players are tier 3."""
    min_tier: int = 2
    cooling_hours: float = 12.0
    significance_shift: float = 0.005  # 0.5pp dashboard-notification threshold
    default_tier: int = 3


@dataclass(slots=True)
class Assessment:
    """Classified finding from the Assess node."""

    player: str
    team: str | None
    tier: int
    confidence: str  # confirmed | likely | rumor
    category: str  # injury | suspension | retirement | tactical | selection
    severity: str  # out | doubt | minor | non-issue
    item: NewsItem


def severity_at_least(severity: str, threshold: str) -> bool:
    """Whether ``severity`` ranks at or above ``threshold`` in :data:`SEVERITY_ORDER`."""

    try:
        return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)
    except ValueError:
        return False


# -- Scan ----------------------------------------------------------------------


def scan_node(
    teams: list[str], injected: list[NewsItem] | None = None, include_reddit: bool = False
) -> list[NewsItem]:
    """Pull raw news items across configured sources (spec 5.2 Scan)."""

    return scan_sources(teams, include_reddit=include_reddit, injected=injected)


# -- Assess --------------------------------------------------------------------


def heuristic_classifier(item: NewsItem, config: AgentConfig) -> Assessment | None:
    """Deterministic keyword classifier (Ollama-free fallback)."""

    text = f"{item.title} {item.snippet}".lower()
    player = next((p for p in config.player_tiers if p.lower() in text), None)
    if player is None:
        return None

    if any(k in text for k in ("ruled out", "confirmed", "officially")):
        confidence = "confirmed"
    elif any(k in text for k in ("reportedly", "rumour", "rumor", "speculation")):
        confidence = "rumor"
    else:
        confidence = "likely"

    if any(k in text for k in ("suspended", "suspension", "banned", "red card")):
        category = "suspension"
    elif "retire" in text:
        category = "retirement"
    else:
        category = "injury"

    if any(k in text for k in ("out for the tournament", "ruled out", "season-ending")):
        severity = "out"
    elif any(k in text for k in ("doubt", "could miss", "fitness test", "injured")):
        severity = "doubt"
    elif any(k in text for k in ("knock", "minor")):
        severity = "minor"
    else:
        severity = "non-issue"

    return Assessment(
        player=player,
        team=item.team,
        tier=config.player_tiers.get(player, config.default_tier),
        confidence=confidence,
        category=category,
        severity=severity,
        item=item,
    )


def assess_node(
    items: list[NewsItem],
    config: AgentConfig,
    classifier: Callable[[NewsItem, AgentConfig], Assessment | None] | None = None,
) -> list[Assessment]:
    """Classify items and keep only material findings (spec 5.2 Assess).

    Passes forward only: tier <= ``min_tier`` AND confidence in {confirmed, likely} AND
    severity >= "doubt".
    """

    classify = classifier or heuristic_classifier
    passed: list[Assessment] = []
    for item in items:
        assessment = classify(item, config)
        if assessment is None:
            continue
        if (
            assessment.tier <= config.min_tier
            and assessment.confidence in _CONFIDENCE_PASS
            and severity_at_least(assessment.severity, "doubt")
        ):
            passed.append(assessment)
    return passed


# -- Cross-Reference -----------------------------------------------------------


def cross_reference_node(
    assessments: list[Assessment],
    state: AgentState,
    now: datetime,
    config: AgentConfig,
) -> list[Assessment]:
    """Filter to net-new material changes (spec 5.2 Cross-Reference).

    Drops items whose player already has the same status, items inside the per-player
    cooling period, and "likely" items lacking corroboration (a "likely" item is held
    unless the player already has a non-"fit" status from a different source).
    """

    net_new: list[Assessment] = []
    for a in assessments:
        existing = state.get_player_status(a.player)
        new_status = "out" if a.severity == "out" else "doubt"
        if existing is not None and existing["status"] == new_status:
            continue  # already known
        if state.recently_assessed(a.player, now, config.cooling_hours):
            continue  # cooling period
        if a.confidence == "likely":
            corroborated = (
                existing is not None
                and existing["status"] != "fit"
                and existing["source"] != a.item.source
            )
            if not corroborated:
                continue  # needs a second source within 12h
        net_new.append(a)
    return net_new


# -- Act -----------------------------------------------------------------------


def act_node(
    changes: list[Assessment],
    state: AgentState,
    now: datetime,
    simulate_fn: Callable[[], None] | None = None,
) -> int:
    """Apply changes: update statuses, log, optionally re-simulate (spec 5.2 Act)."""

    for a in changes:
        new_status = "out" if a.severity == "out" else "doubt"
        confidence = 0.9 if a.confidence == "confirmed" else 0.6
        state.upsert_player_status(
            a.player, a.team or "", new_status, now, a.item.source, confidence
        )
        state.append_changelog(
            timestamp=now,
            team=a.team or "",
            player=a.player,
            change=f"{a.category}: {new_status}",
            reasoning=f"{a.confidence} via {a.item.source}: {a.item.title}",
            prob_shift=0.0,
        )
    if changes and simulate_fn is not None:
        simulate_fn()
    return len(changes)


# -- Reflect -------------------------------------------------------------------


def reflect_node(
    prev_probs: dict[str, float],
    new_probs: dict[str, float],
    config: AgentConfig,
) -> list[dict[str, float]]:
    """Flag teams whose trophy probability moved by > the significance threshold (spec 5.2)."""

    significant: list[dict[str, float]] = []
    for team, new_p in new_probs.items():
        shift = new_p - prev_probs.get(team, new_p)
        if abs(shift) > config.significance_shift:
            significant.append({"team": team, "shift": shift, "new_prob": new_p})
    return significant


# -- Orchestrator --------------------------------------------------------------


def run_cycle(
    state: AgentState,
    teams: list[str],
    config: AgentConfig,
    *,
    now: datetime,
    injected_items: list[NewsItem] | None = None,
    classifier: Callable[[NewsItem, AgentConfig], Assessment | None] | None = None,
    simulate_fn: Callable[[], None] | None = None,
    prev_probs: dict[str, float] | None = None,
    new_probs: dict[str, float] | None = None,
) -> dict[str, object]:
    """Run one full Scan -> Assess -> Cross-Ref -> Act -> Reflect cycle.

    Persists a run record, the per-node decision trace, and the changelog, and returns a
    summary dict (items scanned/assessed/acted and any significant probability shifts).
    """

    started = now
    items = scan_node(teams, injected=injected_items)
    assessments = assess_node(items, config, classifier)
    net_new = cross_reference_node(assessments, state, now, config)
    acted = act_node(net_new, state, now, simulate_fn)
    significant = (
        reflect_node(prev_probs, new_probs, config)
        if prev_probs is not None and new_probs is not None
        else []
    )

    run_id = state.record_run(
        timestamp=started,
        duration_s=0.0,
        items_scanned=len(items),
        items_acted=acted,
    )
    state.record_decision(run_id, now, "scan", {"items": len(items)})
    state.record_decision(run_id, now, "assess", {"passed": len(assessments)})
    state.record_decision(run_id, now, "cross_reference", {"net_new": len(net_new)})
    state.record_decision(run_id, now, "act", {"acted": acted})
    state.record_decision(run_id, now, "reflect", {"significant": significant})

    return {
        "run_id": run_id,
        "scanned": len(items),
        "assessed": len(assessments),
        "net_new": len(net_new),
        "acted": acted,
        "significant_shifts": significant,
    }
