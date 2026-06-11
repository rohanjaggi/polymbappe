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


def team_strengths(model: object) -> dict[str, float]:
    """Overall strength per team from a fitted Dixon-Coles model (``attack - defense``).

    Higher is stronger: high attack and low (often negative) defense — matching the
    ``lam = exp(... + attack_home + defense_away)`` convention.
    """

    idx = model.team_to_index  # type: ignore[attr-defined]
    return {t: float(model.attack[i] - model.defense[i]) for t, i in idx.items()}  # type: ignore[attr-defined]


def _pseudo_elo(strengths: dict[str, float]) -> dict[str, float]:
    """Map model strengths onto an Elo-like scale (~1500 ± spread) for the upset floor."""

    import statistics

    values = list(strengths.values())
    if len(values) < 2:
        return {t: 1500.0 for t in strengths}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) or 1.0
    return {t: 1500.0 + (s - mean) / std * 180.0 for t, s in strengths.items()}


def pot_seed_groups(ranked_teams: list[str]) -> dict[str, list[str]]:
    """Pot-seed 48 ranked teams into 12 groups of 4 (one team per strength pot).

    Pots are the four contiguous twelfths of the ranking (best -> worst); group ``g`` takes
    the ``g``-th team from each pot, so every group spans the full strength range — the
    same balancing principle as the real FIFA pot draw (minus confederation constraints,
    which need data we don't reliably have).
    """

    if len(ranked_teams) != 48:
        raise ValueError(f"pot seeding needs exactly 48 teams; got {len(ranked_teams)}.")
    pots = [ranked_teams[p * 12 : p * 12 + 12] for p in range(4)]
    return {GROUP_LETTERS[g]: [pots[p][g] for p in range(4)] for g in range(12)}


def structure_from_strengths(
    model: object,
    elo: dict[str, float] | None = None,
    n_teams: int = 48,
) -> TournamentStructure:
    """Build a pot-seeded 2026 structure from a fitted model's strongest ``n_teams``.

    Teams are ranked by real Elo when an ``elo`` map is supplied, otherwise by model
    strength (:func:`team_strengths`). The top ``n_teams`` are pot-seeded into the 12
    groups, and an Elo map (real or strength-derived) is attached so the knockout upset
    floor has a meaningful scale. Requires the model to know at least ``n_teams`` teams.
    """

    strengths = team_strengths(model)
    if len(strengths) < n_teams:
        raise ValueError(
            f"model knows {len(strengths)} teams; need >= {n_teams} for a 2026 draw."
        )
    rank_key = elo if elo else strengths
    ranked = sorted(strengths, key=lambda t: rank_key.get(t, float("-inf")), reverse=True)
    selected = ranked[:n_teams]
    groups = pot_seed_groups(selected)
    elo_map = {t: float(elo[t]) for t in selected if t in elo} if elo else _pseudo_elo(
        {t: strengths[t] for t in selected}
    )
    return TournamentStructure(groups=groups, elo=elo_map)


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
