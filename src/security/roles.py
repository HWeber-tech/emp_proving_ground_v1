"""Canonical role enumerations for the EMP security model."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SystemRole", "INITIAL_ROLES"]


class SystemRole(StrEnum):
    """EMP runtime roles that gate access to protected capabilities."""

    ADMIN = "admin"
    READER = "reader"
    INGEST_PROCESS = "ingest_process"


INITIAL_ROLES: tuple[str, ...] = tuple(role.value for role in SystemRole)
