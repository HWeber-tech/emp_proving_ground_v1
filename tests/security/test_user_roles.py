from __future__ import annotations

import pytest

from src.security.user_roles import InMemoryUserRoleStore, SQLiteUserRoleStore


def test_in_memory_role_store_assignments() -> None:
    store = InMemoryUserRoleStore()
    store.assign_role("alice", "ops")
    store.assign_roles("alice", ["viewer", "ops", " ", "analyst"])

    assert store.get_roles("alice") == ("analyst", "ops", "viewer")
    assert store.has_role("alice", "ops")
    assert store.users_with_role("viewer") == ("alice",)

    store.assign_role("bob", "viewer")
    assert store.list_users() == ("alice", "bob")

    assert store.revoke_role("alice", "ops") is True
    assert store.get_roles("alice") == ("analyst", "viewer")

    store.clear_roles("alice")
    assert store.get_roles("alice") == ()

    with pytest.raises(ValueError):
        store.assign_role(" ", "ops")


def test_sqlite_role_store_persistence(tmp_path) -> None:
    db_path = tmp_path / "roles.db"
    store = SQLiteUserRoleStore(db_path)

    store.assign_roles("alice", ["ops", "viewer", "ops"])
    store.assign_role("bob", "viewer")
    store.assign_role("carol", "analyst")

    assert store.list_users() == ("alice", "bob", "carol")
    assert store.users_with_role("viewer") == ("alice", "bob")

    store.close()

    reopened = SQLiteUserRoleStore(db_path)
    assert reopened.get_roles("alice") == ("ops", "viewer")
    assert reopened.has_role("alice", "ops")

    assert reopened.revoke_role("bob", "viewer") is True
    assert reopened.get_roles("bob") == ()

    reopened.set_roles("carol", ["admin"])  # replace existing roles
    assert reopened.get_roles("carol") == ("admin",)

    reopened.set_roles("dave", [])
    assert reopened.get_roles("dave") == ()

    with pytest.raises(ValueError):
        reopened.assign_role("", "ops")

    reopened.close()
