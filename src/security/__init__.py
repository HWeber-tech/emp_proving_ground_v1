"""Authentication helpers including JWT-style token utilities."""

from __future__ import annotations

from .auth_tokens import (
    AuthTokenError,
    ExpiredTokenError,
    InvalidTokenError,
    create_access_token,
    create_auth_token,
    decode_access_token,
    decode_auth_token,
)
from .roles import INITIAL_ROLES, SystemRole
from .user_roles import InMemoryUserRoleStore, SQLiteUserRoleStore

__all__ = [
    "AuthTokenError",
    "InvalidTokenError",
    "ExpiredTokenError",
    "create_access_token",
    "create_auth_token",
    "decode_access_token",
    "decode_auth_token",
    "INITIAL_ROLES",
    "SystemRole",
    "InMemoryUserRoleStore",
    "SQLiteUserRoleStore",
]
