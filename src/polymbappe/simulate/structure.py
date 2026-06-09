"""2026 World Cup tournament structure loading (12 groups of 4).

The real draw is loaded from ``configs/tournament_2026.yaml`` when present (a mapping of
group letter -> list of four team names, optionally with ``elo`` and ``penalty_rate``
maps). Until the draw is published a deterministic placeholder of 48 teams is generated so
the simulation pipeline is runnable end-to-end. The placeholder is clearly labelled and
must not be mistaken for the real draw.
"""

from __future__ import annotations

from pathlib import Path

from polymbappe.config import Settings
from polymbappe.simulate.tournament import TournamentStructure

GROUP_LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJKL")  # 12 groups


def build_structure(
    groups: dict[str, list[str]],
    elo: dict[str, float] | None = None,
    penalty_rate: dict[str, float] | None = None,
) -> TournamentStructure:
    """Validate and build a :class:`TournamentStructure` from a group mapping."""

    if len(groups) != 12:
        raise ValueError(f"2026 needs 12 groups; got {len(groups)}.")
    for letter, members in groups.items():
        if len(members) != 4:
            raise ValueError(f"Group {letter} must have 4 teams; got {len(members)}.")
    return TournamentStructure(
        groups=groups, elo=elo or {}, penalty_rate=penalty_rate or {}
    )


def placeholder_structure_2026() -> TournamentStructure:
    """Deterministic 48-team placeholder (``Team01``..``Team48``) — NOT the real draw."""

    teams = [f"Team{ i + 1:02d}" for i in range(48)]
    groups = {GROUP_LETTERS[g]: teams[g * 4 : g * 4 + 4] for g in range(12)}
    # Descending placeholder Elo so the upset-floor / seeding paths exercise real spread.
    elo = {team: 1900.0 - 12.0 * i for i, team in enumerate(teams)}
    return build_structure(groups, elo=elo)


def load_structure_2026(settings: Settings | None = None) -> TournamentStructure:
    """Load the 2026 draw from config, falling back to the labelled placeholder."""

    settings = settings or Settings()
    config_path: Path = settings.configs_dir / "tournament_2026.yaml"
    if config_path.exists():
        import yaml

        with config_path.open() as fh:
            data = yaml.safe_load(fh)
        return build_structure(
            data["groups"], elo=data.get("elo"), penalty_rate=data.get("penalty_rate")
        )
    return placeholder_structure_2026()
