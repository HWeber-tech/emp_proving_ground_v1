"""Utilities for protecting DuckDB storage."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ENCRYPTED_ROOT_ENV = "EMP_DUCKDB_ENCRYPTED_ROOT"
_REQUIRE_ENV = "EMP_REQUIRE_ENCRYPTED_DUCKDB"
_FALSEY = {"0", "false", "no", "off", ""}


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in _FALSEY


def resolve_encrypted_duckdb_path(candidate: Path | str) -> Path:
    """Return a path located on the encrypted volume when configured.

    When the environment variable EMP_DUCKDB_ENCRYPTED_ROOT is set, all DuckDB
    files are stored beneath that directory. If EMP_REQUIRE_ENCRYPTED_DUCKDB is
    enabled, the helper enforces the configuration and raises when the encrypted
    root misbehaves.
    """

    original = Path(candidate).expanduser()
    encrypted_root_raw = os.environ.get(_ENCRYPTED_ROOT_ENV)
    require_encrypted = _parse_bool(os.environ.get(_REQUIRE_ENV))

    if not encrypted_root_raw:
        if require_encrypted:
            raise RuntimeError(
                "EMP_REQUIRE_ENCRYPTED_DUCKDB is set but EMP_DUCKDB_ENCRYPTED_ROOT is missing"
            )
        return original

    encrypted_root = Path(encrypted_root_raw).expanduser()
    try:
        encrypted_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        message = (
            "Failed to create DuckDB encrypted root %s; falling back to %s",
            encrypted_root,
            original,
        )
        logger.warning(*message, exc_info=exc)
        if require_encrypted:
            raise RuntimeError(
                f"Unable to prepare encrypted DuckDB root {encrypted_root}: {exc}"
            ) from exc
        return original

    if original.is_absolute():
        try:
            original.relative_to(encrypted_root)
        except ValueError:
            relative_target = original.name
        else:
            relative_target = original.relative_to(encrypted_root)
    else:
        relative_target = original

    secure_path = encrypted_root / relative_target
    if not secure_path.parent.exists():
        secure_path.parent.mkdir(parents=True, exist_ok=True)

    return secure_path


def verify_encrypted_duckdb_path(path: Path | str) -> None:
    """Ensure the provided path resides on the encrypted root when required."""

    encrypted_root_raw = os.environ.get(_ENCRYPTED_ROOT_ENV)
    if not encrypted_root_raw:
        return

    encrypted_root = Path(encrypted_root_raw).expanduser()
    target = Path(path).expanduser()
    try:
        target.relative_to(encrypted_root)
    except ValueError as exc:
        if _parse_bool(os.environ.get(_REQUIRE_ENV)):
            raise RuntimeError(
                f"DuckDB path {target} is not within encrypted root {encrypted_root}"
            ) from exc
        logger.warning(
            "DuckDB path %s is outside encrypted root %s; continuing without enforcement",
            target,
            encrypted_root,
        )

