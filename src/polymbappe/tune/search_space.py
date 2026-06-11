"""Search-space definition and Optuna sampling (spec section 8.2).

Loads ``configs/autotuner_search_space.yaml`` into a flat parameter registry and samples
a concrete config from an Optuna trial. Parameter keys are namespaced ``group.name`` to
keep components separate while staying a flat dict for the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from polymbappe.config import Settings


@dataclass(slots=True)
class ParamSpec:
    """One tunable parameter."""

    name: str  # namespaced "group.param"
    kind: str  # float | int | categorical
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None


@dataclass(slots=True)
class SearchSpace:
    """Flat collection of parameter specs."""

    params: list[ParamSpec]

    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample a concrete config dict from an Optuna trial."""

        out: dict[str, Any] = {}
        for spec in self.params:
            if spec.kind == "float":
                out[spec.name] = trial.suggest_float(
                    spec.name, float(spec.low), float(spec.high), log=spec.log
                )
            elif spec.kind == "int":
                out[spec.name] = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
            elif spec.kind == "categorical":
                out[spec.name] = trial.suggest_categorical(spec.name, spec.choices)
            else:  # pragma: no cover - guarded by loader
                raise ValueError(f"Unknown parameter kind: {spec.kind}")
        return out


def _parse_group(group: str, spec: dict[str, Any]) -> list[ParamSpec]:
    params: list[ParamSpec] = []
    for name, body in spec.items():
        kind = body["type"]
        params.append(
            ParamSpec(
                name=f"{group}.{name}",
                kind=kind,
                low=body.get("low"),
                high=body.get("high"),
                log=bool(body.get("log", False)),
                choices=body.get("choices"),
            )
        )
    return params


def load_search_space(path: Path | None = None, settings: Settings | None = None) -> SearchSpace:
    """Load the search space from YAML (defaults to ``configs/autotuner_search_space.yaml``)."""

    settings = settings or Settings()
    path = path or (settings.configs_dir / "autotuner_search_space.yaml")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    params: list[ParamSpec] = []
    for group, body in data.items():
        if isinstance(body, dict):
            params.extend(_parse_group(group, body))
    return SearchSpace(params=params)
