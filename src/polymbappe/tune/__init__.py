"""Automated hyperparameter tuning — the autoresearch loop (spec section 8).

Two phases:

* **Phase 1** (:mod:`.llm_search`) — an LLM proposes qualitatively different *structural*
  experiments (feature inclusion, architecture, meta-learner choice) that a numeric
  optimizer cannot search.
* **Phase 2** (:mod:`.optuna_tuner`) — Optuna TPE optimizes the numeric search space
  (:mod:`.search_space`) within the fixed structure.

Both gate candidates through the acceptance criteria (:mod:`.leaderboard`): a config is
accepted only if it improves mean RPS by >0.003 *and* wins on >=3/4 individual tournaments.
"""

from polymbappe.tune.leaderboard import AcceptanceGate, Leaderboard, accept_config
from polymbappe.tune.objective import BacktestObjective, config_to_metrics
from polymbappe.tune.search_space import SearchSpace, load_search_space

__all__ = [
    "AcceptanceGate",
    "Leaderboard",
    "accept_config",
    "BacktestObjective",
    "config_to_metrics",
    "SearchSpace",
    "load_search_space",
]
