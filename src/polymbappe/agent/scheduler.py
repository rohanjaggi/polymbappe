"""APScheduler integration for the live agent (spec section 5.4).

Triggers the agent cycle every 6 hours pre-tournament and every 2 hours during the
tournament, with a manual ``--run-now``. APScheduler is an optional (``context``)
dependency imported lazily. :func:`parse_interval` and the CLI control functions are pure
and testable; :func:`start_scheduler` requires APScheduler.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from polymbappe.agent.graph import run_agent_cycle
from polymbappe.agent.nodes import AgentConfig
from polymbappe.agent.state import AgentState

PRE_TOURNAMENT_HOURS = 6
DURING_TOURNAMENT_HOURS = 2


def parse_interval(interval: str) -> int:
    """Parse a ``"6h"`` / ``"120m"`` interval string into seconds (default 6h)."""

    match = re.fullmatch(r"\s*(\d+)\s*([hm])\s*", interval.lower())
    if not match:
        return PRE_TOURNAMENT_HOURS * 3600
    value, unit = int(match.group(1)), match.group(2)
    return value * 3600 if unit == "h" else value * 60


def run_now(
    teams: list[str],
    config: AgentConfig | None = None,
    settings: Any = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Run a single agent cycle immediately (``polymbappe agent --run-now``)."""

    state = AgentState(settings)
    try:
        summary = run_agent_cycle(state, teams, config, now=datetime.now(), **kwargs)
        state.export_changelog_parquet()
        return summary
    finally:
        state.close()


def start_scheduler(
    cycle: Callable[[], Any],
    interval_seconds: int = PRE_TOURNAMENT_HOURS * 3600,
    block: bool = True,
) -> Any:
    """Schedule ``cycle`` on a fixed interval via APScheduler (requires apscheduler)."""

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "apscheduler is not installed; install the 'context' extra to schedule the agent."
        ) from exc

    scheduler = BlockingScheduler() if block else BackgroundScheduler()  # pragma: no cover
    scheduler.add_job(cycle, "interval", seconds=interval_seconds)  # pragma: no cover
    scheduler.start()  # pragma: no cover
    return scheduler  # pragma: no cover
