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

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


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


def _parse_llm_json(content: str) -> dict[str, Any]:
    """Parse the model's reply into a dict, tolerating two qwen quirks.

    Even with ``format="json"`` set, qwen3.5 (via Ollama 0.30) routinely (a) wraps the
    object in a ```json ... ``` markdown fence and (b) emits Python literals such as
    ``None`` for a nullable value. Either makes a bare ``json.loads`` raise, which the
    caller's broad ``except`` would silently turn into a fallback -- so the LLM path would
    run (paying ~30s/call) yet never contribute a single proposal. Strip the fence and, on
    failure, recover via ``ast.literal_eval`` after normalizing JSON keywords to Python.
    """

    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        span = text[start : end + 1]
        span = re.sub(r"\btrue\b", "True", span)
        span = re.sub(r"\bfalse\b", "False", span)
        span = re.sub(r"\bnull\b", "None", span)
        parsed = ast.literal_eval(span)
        if not isinstance(parsed, dict):
            raise
        return parsed


def _ollama_available() -> bool:
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return True


def _tunable_schema() -> tuple[str, dict[str, Any]]:
    """Return (human-readable param menu for the prompt, specs keyed by namespaced name).

    Sourced from the Phase-2 search space so the two phases share one definition of which
    knobs the backtest actually exercises.
    """

    from polymbappe.tune.search_space import load_search_space

    lines: list[str] = []
    specs: dict[str, Any] = {}
    for spec in load_search_space().params:
        specs[spec.name] = spec
        if spec.kind == "categorical":
            desc = f"categorical, one of {spec.choices}"
        elif spec.kind == "int":
            desc = f"integer in [{int(spec.low)}, {int(spec.high)}]"
        else:
            desc = f"float in [{spec.low}, {spec.high}]" + (" (log scale)" if spec.log else "")
        lines.append(f"- {spec.name}: {desc}")
    return "\n".join(lines), specs


def _valid_value(spec: Any, value: Any) -> bool:
    """Whether ``value`` respects ``spec``'s declared type and bound/choice set.

    The LLM can return a valid key with an out-of-spec value (e.g. the string ``"None"``
    for a categorical, or a number past its range). Such values would otherwise reach the
    objective and either skew the backtest or crash a cast, so they are rejected here.
    """

    if spec.kind == "categorical":
        return value in (spec.choices or [])
    if isinstance(value, bool):  # bool subclasses int; never a valid numeric knob value
        return False
    if not isinstance(value, (int, float)):
        return False
    if spec.low is not None and value < spec.low:
        return False
    if spec.high is not None and value > spec.high:
        return False
    return True


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

        schema, specs = _tunable_schema()
        allowed_keys = list(specs)
        prompt = (
            "You are tuning a football forecasting ensemble. Given prior experiment "
            "results (JSON), propose ONE structural change as JSON with keys "
            '"name", "config", "hypothesis".\n'
            '"config" MUST be a flat dict. Every key MUST be copied CHARACTER-FOR-CHARACTER '
            "from the allowed-keys list below -- do not rename, abbreviate, reorder letters, "
            "add or remove underscores, or pluralize. In particular the boosting namespace is "
            'exactly "gbm." (g-b-m), never "gmb." or "g_bm.". Any key that is not a verbatim '
            "match is silently discarded, which collapses your experiment into a no-op rerun "
            "of the baseline, so accuracy of the key strings matters more than how many you "
            "include. Values must respect the stated type/range. Pick a qualitatively "
            "different combination from the prior experiments; do not invent keys.\n"
            f"Allowed keys (use these exact strings only): {json.dumps(allowed_keys)}\n"
            f"Per-key types/ranges:\n{schema}\n"
            f"Prior results: {json.dumps(prior_results)}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            # Disable "thinking" on reasoning models (e.g. Qwen3): with format=json the
            # reasoning budget is spent on a long hidden trace that intermittently leaves
            # the JSON content empty. think=False makes the model emit the JSON directly.
            resp = ollama.chat(model=model, messages=messages, format="json", think=False)
        except Exception:  # noqa: BLE001 - older Ollama / non-thinking model rejects think=
            resp = ollama.chat(model=model, messages=messages, format="json")
        data = _parse_llm_json(resp["message"]["content"])
        name = str(data.get("name", next_fallback.name))
        config: dict[str, Any] = {}
        dropped: dict[str, str] = {}
        for key, value in dict(data.get("config", {})).items():
            if key not in specs:
                dropped[key] = "unknown_key"
            elif not _valid_value(specs[key], value):
                dropped[key] = "out_of_spec_value"
            else:
                config[key] = value
        if dropped:
            # Surface keys the LLM proposed that the objective cannot act on, so an
            # experiment that silently collapses toward the baseline is visible in the log
            # rather than hiding behind a bare "inconclusive".
            logger.warning(
                "autotune.llm_proposal_dropped_keys",
                name=name,
                dropped=dropped,
                kept=sorted(config),
            )
        if not config:
            # Nothing survived -> this proposal would just rerun the baseline (or crash a
            # cast). Fall back to a curated change so the loop makes real progress.
            logger.warning(
                "autotune.llm_proposal_discarded", name=name, fallback=next_fallback.name
            )
            return next_fallback
        return StructuralExperiment(
            name=name,
            config=config,
            hypothesis=str(data.get("hypothesis", "")),
        )
    except Exception:  # noqa: BLE001 - any LLM/parse failure -> deterministic fallback
        return next_fallback
