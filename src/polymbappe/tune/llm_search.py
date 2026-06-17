"""Phase 1: LLM-guided structural search (spec section 8.3).

The LLM acts as the *researcher*, not the optimizer: it proposes qualitatively different
structural experiments (feature inclusion, training scope, architecture, meta-learner
choice) that a continuous optimizer cannot search. Each proposal is a config override plus
a hypothesis; the runner backtests it and applies the acceptance gate.

Qwen via Ollama is used when available (structured JSON output). When Ollama is not
installed/running, a curated fallback list of structural experiments is used so Phase 1 is
fully runnable offline (and in CI).

The LLM prompt is constrained to the real tunable parameters (loaded from the Phase-2
search space), and its proposed config is filtered to those keys, so the model cannot
invent namespaces the backtest objective silently ignores — every accepted proposal maps
onto a live knob instead of degenerating into a no-op rerun of the baseline.
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


def _tunable_schema() -> tuple[str, set[str]]:
    """Return (human-readable param menu for the prompt, set of valid namespaced keys).

    Sourced from the Phase-2 search space so the two phases share one definition of which
    knobs the backtest actually exercises.
    """

    from polymbappe.tune.search_space import load_search_space

    lines: list[str] = []
    keys: set[str] = set()
    for spec in load_search_space().params:
        keys.add(spec.name)
        if spec.kind == "categorical":
            desc = f"categorical, one of {spec.choices}"
        elif spec.kind == "int":
            desc = f"integer in [{int(spec.low)}, {int(spec.high)}]"
        else:
            desc = f"float in [{spec.low}, {spec.high}]" + (" (log scale)" if spec.log else "")
        lines.append(f"- {spec.name}: {desc}")
    return "\n".join(lines), keys


def propose_structural_experiment(
    prior_results: list[dict[str, Any]],
    *,
    model: str = "qwen3.5:9b",
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

        schema, valid_keys = _tunable_schema()
        prompt = (
            "You are tuning a football forecasting ensemble. Given prior experiment "
            "results (JSON), propose ONE structural change as JSON with keys "
            '"name", "config", "hypothesis".\n'
            '"config" MUST be a flat dict whose keys are chosen ONLY from this menu of '
            "tunable parameters (use the exact key names; values must respect the stated "
            "type/range). Pick a qualitatively different combination from the prior "
            "experiments; do not invent keys outside the menu.\n"
            f"Tunable parameters:\n{schema}\n"
            f"Prior results: {json.dumps(prior_results)}"
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        data = json.loads(resp["message"]["content"])
        config = {
            k: v for k, v in dict(data.get("config", {})).items() if k in valid_keys
        }
        if not config:
            # Every proposed key was hallucinated/inert -> the experiment would just rerun
            # the baseline. Fall back to a curated change so the loop makes real progress.
            return next_fallback
        return StructuralExperiment(
            name=str(data.get("name", next_fallback.name)),
            config=config,
            hypothesis=str(data.get("hypothesis", "")),
        )
    except Exception:  # noqa: BLE001 - any LLM/parse failure -> deterministic fallback
        return next_fallback
