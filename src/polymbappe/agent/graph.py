"""LangGraph state-machine assembly (spec section 5.1).

Wraps the pure node functions (:mod:`.nodes`) in a LangGraph ``StateGraph`` with the
spec's conditional routing: Scan -> Assess -> Cross-Reference -> Act -> Reflect, with
short-circuit edges when a stage yields nothing material. LangGraph is an optional
(``context``) dependency imported lazily; :func:`run_agent_cycle` runs the same pipeline
without LangGraph so the agent is usable (and testable) without it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from polymbappe.agent.nodes import AgentConfig, run_cycle
from polymbappe.agent.state import AgentState


def langgraph_available() -> bool:
    """Whether the optional ``langgraph`` dependency is importable."""

    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def build_agent_graph() -> Any:
    """Construct the LangGraph ``StateGraph`` for the agent (requires langgraph).

    The compiled graph threads a mutable state dict through the five nodes with conditional
    routing. Raises ``RuntimeError`` if langgraph is not installed.
    """

    if not langgraph_available():  # pragma: no cover - exercised only with langgraph
        raise RuntimeError(
            "langgraph is not installed; install the 'context' extra or use run_agent_cycle()."
        )

    # pragma: no cover below — only runs with the optional dependency present.
    from langgraph.graph import END, START, StateGraph  # pragma: no cover

    from polymbappe.agent.nodes import (  # pragma: no cover
        assess_node,
        cross_reference_node,
    )

    graph = StateGraph(dict)  # pragma: no cover

    def _scan(state: dict) -> dict:  # pragma: no cover
        from polymbappe.agent.nodes import scan_node

        state["items"] = scan_node(state["teams"], injected=state.get("injected_items"))
        return state

    def _assess(state: dict) -> dict:  # pragma: no cover
        state["assessments"] = assess_node(state["items"], state["config"])
        return state

    def _cross_ref(state: dict) -> dict:  # pragma: no cover
        state["net_new"] = cross_reference_node(
            state["assessments"], state["state"], state["now"], state["config"]
        )
        return state

    def _act(state: dict) -> dict:  # pragma: no cover
        from polymbappe.agent.nodes import act_node

        state["acted"] = act_node(
            state["net_new"], state["state"], state["now"], state.get("simulate_fn")
        )
        return state

    def _reflect(state: dict) -> dict:  # pragma: no cover
        from polymbappe.agent.nodes import reflect_node

        if state.get("prev_probs") is not None and state.get("new_probs") is not None:
            state["significant"] = reflect_node(
                state["prev_probs"], state["new_probs"], state["config"]
            )
        return state

    for name, fn in (
        ("scan", _scan),
        ("assess", _assess),
        ("cross_reference", _cross_ref),
        ("act", _act),
        ("reflect", _reflect),
    ):
        graph.add_node(name, fn)  # pragma: no cover

    graph.add_edge(START, "scan")  # pragma: no cover
    graph.add_edge("scan", "assess")  # pragma: no cover
    graph.add_conditional_edges(  # pragma: no cover
        "assess", lambda s: "cross_reference" if s["assessments"] else END
    )
    graph.add_conditional_edges(  # pragma: no cover
        "cross_reference", lambda s: "act" if s["net_new"] else END
    )
    graph.add_edge("act", "reflect")  # pragma: no cover
    graph.add_edge("reflect", END)  # pragma: no cover
    return graph.compile()  # pragma: no cover


def run_agent_cycle(
    state: AgentState,
    teams: list[str],
    config: AgentConfig | None = None,
    *,
    now: datetime | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Run one agent cycle (LangGraph-free path); the canonical programmatic entrypoint."""

    return run_cycle(
        state,
        teams,
        config or AgentConfig(),
        now=now or datetime.now(),
        **kwargs,
    )
