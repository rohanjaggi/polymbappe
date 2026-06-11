"""Team-name normalization to a single canonical spelling.

Different sources spell national teams differently — the Kaggle results feed uses one
spelling, Polymarket and bookmakers use others ("USA", "Czechia", "S. Korea"). Joining
odds to fixtures by ``date__home__away`` / ``2026__home__away`` only works if both sides
agree on the name, so every source's team names are pushed through
:func:`normalize_team_name` first.

The canonical spelling is the one the Kaggle (martj42) results feed uses, since that drives
the ``matches`` table and the simulation. :data:`DEFAULT_ALIASES` maps lowercased alternate
spellings onto those canonical names; ``configs/team_aliases.yaml`` (optional) extends or
overrides it without code changes. Names with no alias entry pass through unchanged (only
whitespace-trimmed), so canonical names are always safe.
"""

from __future__ import annotations

import polars as pl

from polymbappe.config import Settings

#: Lowercased alias -> canonical (Kaggle/martj42) spelling. Extend via config as needed.
DEFAULT_ALIASES: dict[str, str] = {
    # United States
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    # Korea
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "s. korea": "South Korea",
    "korea dpr": "North Korea",
    "dpr korea": "North Korea",
    # Iran
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    # Ivory Coast
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    # Czechia
    "czechia": "Czech Republic",
    # Bosnia
    "bosnia": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    # Macedonia
    "macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    # Congo
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    # Ireland
    "ireland": "Republic of Ireland",
    "republic of ireland": "Republic of Ireland",
    # China
    "china": "China PR",
    "china pr": "China PR",
    # Türkiye
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    # Netherlands
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    # Gulf / others
    "uae": "United Arab Emirates",
    "ksa": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "cape verde islands": "Cape Verde",
    "cabo verde": "Cape Verde",
    "south africa rsa": "South Africa",
}

_CACHE: dict[str, str] | None = None


def team_aliases(settings: Settings | None = None) -> dict[str, str]:
    """Return the alias map (defaults merged with ``configs/team_aliases.yaml`` if present).

    The config file is a flat ``alias: Canonical`` mapping; its keys are lowercased and it
    takes precedence over the defaults. The merged map is cached after first load.
    """

    global _CACHE
    if _CACHE is not None:
        return _CACHE
    merged = dict(DEFAULT_ALIASES)
    settings = settings or Settings()
    path = settings.configs_dir / "team_aliases.yaml"
    if path.exists():
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        for alias, canonical in data.items():
            merged[str(alias).strip().lower()] = str(canonical)
    _CACHE = merged
    return merged


def normalize_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Map one team name to its canonical spelling (whitespace-trimmed if unknown)."""

    aliases = aliases if aliases is not None else team_aliases()
    cleaned = " ".join(name.split())
    return aliases.get(cleaned.lower(), cleaned)


def normalize_team_expr(column: str, aliases: dict[str, str] | None = None) -> pl.Expr:
    """Polars expression mapping a team column to canonical spellings."""

    aliases = aliases if aliases is not None else team_aliases()
    stripped = pl.col(column).str.strip_chars()
    return stripped.str.to_lowercase().replace_strict(
        aliases, default=stripped, return_dtype=pl.Utf8
    )
