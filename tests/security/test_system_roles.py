from __future__ import annotations

from src.security.roles import INITIAL_ROLES, SystemRole


def test_initial_roles_are_defined() -> None:
    expected = {
        SystemRole.ADMIN.value,
        SystemRole.READER.value,
        SystemRole.INGEST_PROCESS.value,
    }
    assert set(INITIAL_ROLES) == expected
