import pytest

from polymbappe.simulate.structure import load_structure_2026


def test_load_structure_2026_from_yaml() -> None:
    """Loading the real tournament_2026.yaml produces 48 teams in 12 groups."""
    from polymbappe.config import Settings

    settings = Settings()
    structure = load_structure_2026(settings)

    assert len(structure.groups) == 12
    assert all(len(members) == 4 for members in structure.groups.values())
    assert len(structure.teams) == 48
    assert len(set(structure.teams)) == 48


def test_structure_teams_exist_in_matches_db() -> None:
    """All teams in the YAML must exist in the matches database (canonical names)."""
    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    if not table_exists(Table.MATCHES, settings):
        pytest.skip("No matches table ingested")

    matches = read_table(Table.MATCHES, settings)
    known_teams = set(
        matches["home_team"].unique().to_list() + matches["away_team"].unique().to_list()
    )

    structure = load_structure_2026(settings)
    missing = [t for t in structure.teams if t not in known_teams]
    assert missing == [], f"Teams in YAML but not in matches database: {missing}"
