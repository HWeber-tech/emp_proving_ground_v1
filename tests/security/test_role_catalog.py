from __future__ import annotations

from src.security.roles import INITIAL_ROLE_NAMES, INITIAL_ROLES, ROLE_DEFINITIONS


def test_initial_roles_include_expected_entries() -> None:
    assert INITIAL_ROLE_NAMES == ("admin", "reader", "ingest_process")


def test_role_definitions_are_consistent() -> None:
    defined_names = tuple(role.name for role in INITIAL_ROLES)
    assert defined_names == INITIAL_ROLE_NAMES
    for name, role in ROLE_DEFINITIONS.items():
        assert name == role.name
        assert role.description

