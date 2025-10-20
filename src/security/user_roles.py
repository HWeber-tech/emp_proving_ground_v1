"""Basic utilities for storing user-to-role mappings."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import RLock


__all__ = [
    "InMemoryUserRoleStore",
    "SQLiteUserRoleStore",
]


def _normalise_user(user_id: str) -> str:
    text = str(user_id).strip() if user_id is not None else ""
    if not text:
        raise ValueError("user_id must be a non-empty string")
    return text


def _normalise_role(role: str | None) -> str | None:
    if role is None:
        return None
    text = str(role).strip()
    return text or None


def _export_roles(roles: Iterable[str]) -> tuple[str, ...]:
    unique = {str(role) for role in roles if str(role).strip()}
    return tuple(sorted(unique))


class InMemoryUserRoleStore:
    """Simple in-memory role assignment store."""

    def __init__(self, initial: Mapping[str, Iterable[str]] | None = None) -> None:
        self._roles: dict[str, set[str]] = {}
        self._lock = RLock()
        if initial:
            for user_id, roles in initial.items():
                self.set_roles(user_id, roles)

    def assign_role(self, user_id: str, role: str) -> None:
        role_value = _normalise_role(role)
        if role_value is None:
            return
        user = _normalise_user(user_id)
        with self._lock:
            bucket = self._roles.setdefault(user, set())
            bucket.add(role_value)

    def assign_roles(self, user_id: str, roles: Iterable[str]) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        with self._lock:
            bucket = self._roles.setdefault(user, set())
            for role in roles:
                role_value = _normalise_role(role)
                if role_value is None:
                    continue
                bucket.add(role_value)
            return _export_roles(bucket)

    def set_roles(self, user_id: str, roles: Iterable[str]) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        with self._lock:
            bucket = {role for role in (_normalise_role(role) for role in roles) if role is not None}
            if bucket:
                self._roles[user] = bucket
            else:
                self._roles.pop(user, None)
            return _export_roles(bucket)

    def revoke_role(self, user_id: str, role: str) -> bool:
        role_value = _normalise_role(role)
        if role_value is None:
            return False
        user = _normalise_user(user_id)
        with self._lock:
            bucket = self._roles.get(user)
            if not bucket or role_value not in bucket:
                return False
            bucket.remove(role_value)
            if not bucket:
                self._roles.pop(user, None)
            return True

    def clear_roles(self, user_id: str) -> None:
        user = _normalise_user(user_id)
        with self._lock:
            self._roles.pop(user, None)

    def get_roles(self, user_id: str) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        with self._lock:
            bucket = self._roles.get(user, set())
            return _export_roles(bucket)

    def has_role(self, user_id: str, role: str) -> bool:
        role_value = _normalise_role(role)
        if role_value is None:
            return False
        user = _normalise_user(user_id)
        with self._lock:
            bucket = self._roles.get(user)
            return role_value in bucket if bucket else False

    def list_users(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._roles.keys()))

    def users_with_role(self, role: str) -> tuple[str, ...]:
        role_value = _normalise_role(role)
        if role_value is None:
            return tuple()
        with self._lock:
            users = [user for user, roles in self._roles.items() if role_value in roles]
            return tuple(sorted(users))


class SQLiteUserRoleStore:
    """SQLite-backed role assignment store."""

    def __init__(self, database: str | Path) -> None:
        if not database:
            raise ValueError("database path must be provided")
        self._path = Path(database).expanduser()
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS user_roles (\n"
            "    user_id TEXT NOT NULL,\n"
            "    role TEXT NOT NULL,\n"
            "    PRIMARY KEY (user_id, role)\n"
            ")"
        )
        self._conn.commit()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def assign_role(self, user_id: str, role: str) -> None:
        role_value = _normalise_role(role)
        if role_value is None:
            return
        user = _normalise_user(user_id)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?, ?)",
                (user, role_value),
            )

    def assign_roles(self, user_id: str, roles: Iterable[str]) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        normalised = {role for role in (_normalise_role(role) for role in roles) if role}
        if not normalised:
            return self.get_roles(user)
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?, ?)",
                ((user, role) for role in normalised),
            )
        return self.get_roles(user)

    def set_roles(self, user_id: str, roles: Iterable[str]) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        normalised = {role for role in (_normalise_role(role) for role in roles) if role}
        with self._lock, self._conn:
            current = {
                row[0]
                for row in self._conn.execute(
                    "SELECT role FROM user_roles WHERE user_id = ?", (user,)
                )
            }
            to_remove = current - normalised
            if to_remove:
                self._conn.executemany(
                    "DELETE FROM user_roles WHERE user_id = ? AND role = ?",
                    ((user, role) for role in to_remove),
                )
            to_add = normalised - current
            if to_add:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?, ?)",
                    ((user, role) for role in to_add),
                )
        if not normalised:
            self.clear_roles(user)
            return tuple()
        return self.get_roles(user)

    def revoke_role(self, user_id: str, role: str) -> bool:
        role_value = _normalise_role(role)
        if role_value is None:
            return False
        user = _normalise_user(user_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM user_roles WHERE user_id = ? AND role = ?",
                (user, role_value),
            )
            removed = cursor.rowcount > 0
        return removed

    def clear_roles(self, user_id: str) -> None:
        user = _normalise_user(user_id)
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM user_roles WHERE user_id = ?", (user,))

    def get_roles(self, user_id: str) -> tuple[str, ...]:
        user = _normalise_user(user_id)
        with self._lock:
            cursor = self._conn.execute(
                "SELECT role FROM user_roles WHERE user_id = ? ORDER BY role ASC",
                (user,),
            )
            roles = [row[0] for row in cursor]
        return tuple(roles)

    def has_role(self, user_id: str, role: str) -> bool:
        role_value = _normalise_role(role)
        if role_value is None:
            return False
        user = _normalise_user(user_id)
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM user_roles WHERE user_id = ? AND role = ? LIMIT 1",
                (user, role_value),
            )
            return cursor.fetchone() is not None

    def list_users(self) -> tuple[str, ...]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT user_id FROM user_roles ORDER BY user_id ASC"
            )
            users = [row[0] for row in cursor]
        return tuple(users)

    def users_with_role(self, role: str) -> tuple[str, ...]:
        role_value = _normalise_role(role)
        if role_value is None:
            return tuple()
        with self._lock:
            cursor = self._conn.execute(
                "SELECT user_id FROM user_roles WHERE role = ? ORDER BY user_id ASC",
                (role_value,),
            )
            users = [row[0] for row in cursor]
        return tuple(users)
