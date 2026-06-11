"""Phase 1: LLM-guided structural search (spec section 8.3).

The LLM acts as the *researcher*, not the optimizer: it proposes qualitatively different
structural experiments (feature inclusion, training scope, architecture, meta-learner
choice) that a continuous optimizer cannot search. Each proposal is a config override plus
a hypothesis; the runner backtests it and applies the acceptance gate.

Qwen via Ollama is used when available (structured JSON output). When Ollama is not
installed/running, a curated fallback list of structural experiments is used so Phase 1 is
fully runnable offline (and in CI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StructuralExperiment:
    """One Phase-1 structural proposal."""

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    exclude_market: bool = False


def default_structural_experiments() -> list[StructuralExperiment]:
    """Curated structural experiments (offline fallback / Phase-1 seed set)."""

    return [
        StructuralExperiment(
            "zero_friendly_weight",
            {"dixon_coles.friendly_weight": 0.1},
            "Friendlies are weak signal; down-weighting them should sharpen strength estimates.",
        ),
        StructuralExperiment(
            "heavier_friendly_weight",
            {"dixon_coles.friendly_weight": 0.5},
            "More friendly data may stabilize sparse international samples.",
        ),
        StructuralExperiment(
            "slower_time_decay",
            {"dixon_coles.xi": 0.0008},
            "A slower decay keeps more history, helping teams with few recent matches.",
        ),
        StructuralExperiment(
            "wider_draw_band",
            {"features.draw_max": 0.30},
            "Poisson models under-predict draws; widening the Elo draw band may help RPS.",
        ),
        StructuralExperiment(
            "stronger_meta_regularization",
            {"ensemble.meta_C": 0.3},
            "With few stacking features, stronger L2 should reduce overfit.",
        ),
        StructuralExperiment(
            "market_blind",
            {},
            "Measure the market-blind (edge) pipeline's standalone calibration.",
            exclude_market=True,
        ),
    ]


def _ollama_available() -> bool:
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return True


def propose_structural_experiment(
    prior_results: list[dict[str, Any]],
    *,
    model: str = "qwen2.5:7b",
    fallback: list[StructuralExperiment] | None = None,
) -> StructuralExperiment:
    """Propose the next structural experiment from prior results.

    Tries Qwen via Ollama (structured JSON); on any failure returns the next unused
    experiment from the fallback list (cycling), so the loop always makes progress.
    """

    fallback = fallback or default_structural_experiments()
    tried = {r.get("name") for r in prior_results}
    remaining = [e for e in fallback if e.name not in tried]
    next_fallback = remaining[0] if remaining else fallback[len(prior_results) % len(fallback)]

    if not _ollama_available():
        return next_fallback

    try:  # pragma: no cover - exercised only with a live Ollama server
        import ollama

        prompt = (
            "You are tuning a football forecasting ensemble. Given prior experiment "
            "results (JSON), propose ONE structural change as JSON with keys "
            '"name", "config" (a flat dict of namespaced params), "hypothesis". '
            f"Prior results: {json.dumps(prior_results)}"
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        data = json.loads(resp["message"]["content"])
        return StructuralExperiment(
            name=str(data.get("name", next_fallback.name)),
            config=dict(data.get("config", {})),
            hypothesis=str(data.get("hypothesis", "")),
        )
    except Exception:  # noqa: BLE001 - any LLM/parse failure -> deterministic fallback
        return next_fallback
