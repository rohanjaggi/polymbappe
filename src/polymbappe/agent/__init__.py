"""LangGraph live-monitoring agent (spec section 5).

A five-node state machine — Scan -> Assess -> Cross-Reference -> Act -> Reflect — that
watches squad news during the pre-tournament and tournament windows, updates player
statuses, triggers feature recomputation / re-simulation, and logs a transparent decision
trace. The node decision logic is pure Python (testable offline); LangGraph orchestration
(:mod:`.graph`) and APScheduler (:mod:`.scheduler`) wrap it behind lazy imports, and Qwen
reasoning (via Ollama) degrades to deterministic heuristics when unavailable.
"""

from polymbappe.agent.nodes import (
    AgentConfig,
    Assessment,
    assess_node,
    cross_reference_node,
    reflect_node,
    run_cycle,
)
from polymbappe.agent.sources import NewsItem
from polymbappe.agent.state import AgentState

__all__ = [
    "Assessment",
    "AgentConfig",
    "assess_node",
    "cross_reference_node",
    "reflect_node",
    "run_cycle",
    "NewsItem",
    "AgentState",
]
