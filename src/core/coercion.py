"""Utilities for safely coercing heterogeneous numeric inputs."""

from __future__ import annotations

import re
import math
from numbers import Real
from typing import Any, overload, cast

__all__ = ["coerce_float", "coerce_int"]


_NUMERIC_SEPARATOR_PATTERN = re.compile(r"(?<=\d)[_',](?=\d)")
_GROUP_SEPARATOR_PATTERN = re.compile(r"(?<=\d)[\s\u00A0\u2007\u202F\u2009](?=\d)")


def _strip_group_separators(value: str) -> str:
    """Remove common locale-specific grouping separators embedded in numbers."""

    return _GROUP_SEPARATOR_PATTERN.sub("", value)


@overload
def coerce_float(value: object | None, *, default: float) -> float:
    ...


@overload
def coerce_float(value: object | None, *, default: None = ...) -> float | None:
    ...


def coerce_float(value: object | None, *, default: float | None = None) -> float | None:
    """Best-effort conversion of ``value`` to ``float``.

    Strings are stripped before conversion, and non-numeric values fall back to
    ``default`` (``None`` by default).
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            decoded = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return default
        value = decoded
    if isinstance(value, Real):
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return result if math.isfinite(result) else default
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("(") and candidate.endswith(")"):
            inner = candidate[1:-1].strip()
            if inner:
                if inner[0] in "+-":
                    inner = inner[1:].lstrip()
                if inner:
                    candidate = f"-{inner}"
                else:
                    candidate = ""
        if not candidate:
            return default
        normalized = _NUMERIC_SEPARATOR_PATTERN.sub("", candidate)
        normalized = _strip_group_separators(normalized)
        try:
            result = float(normalized)
        except ValueError:
            return default
        return result if math.isfinite(result) else default
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@overload
def coerce_int(value: object | None, *, default: int) -> int:
    ...


@overload
def coerce_int(value: object | None, *, default: None = ...) -> int | None:
    ...


def coerce_int(value: object | None, *, default: int | None = None) -> int | None:
    """Best-effort conversion of ``value`` to ``int``.

    Strings are stripped before conversion, floating-point values are truncated,
    and non-numeric values fall back to ``default`` (``None`` by default).
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            decoded = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return default
        value = decoded
    if isinstance(value, Real):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("(") and candidate.endswith(")"):
            inner = candidate[1:-1].strip()
            if inner:
                if inner[0] in "+-":
                    inner = inner[1:].lstrip()
                if inner:
                    candidate = f"-{inner}"
                else:
                    candidate = ""
        if not candidate:
            return default
        normalized = _NUMERIC_SEPARATOR_PATTERN.sub("", candidate)
        normalized = _strip_group_separators(normalized)
        try:
            return int(normalized)
        except ValueError:
            prefix = normalized.lower().lstrip("+-")
            if prefix.startswith(("0x", "0o", "0b")):
                try:
                    return int(normalized, 0)
                except ValueError:
                    pass
            try:
                return int(float(normalized))
            except ValueError:
                return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default
