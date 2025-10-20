"""Utilities for safely coercing heterogeneous numeric inputs."""

from __future__ import annotations

import math
import re
import unicodedata
from numbers import Real
from typing import Any, overload, cast

__all__ = ["coerce_float", "coerce_int"]


_NUMERIC_SEPARATOR_PATTERN = re.compile(r"(?<=\d)[_',](?=\d)")
_GROUP_SEPARATOR_PATTERN = re.compile(r"(?<=\d)[\s\u00A0\u2007\u202F\u2009](?=\d)")
_SIGN_NORMALIZATION = str.maketrans({"\u2212": "-", "\uFF0D": "-", "\uFE63": "-"})


def _normalize_sign_characters(value: str) -> str:
    return value.translate(_SIGN_NORMALIZATION)


def _strip_group_separators(value: str) -> str:
    """Remove common locale-specific grouping separators embedded in numbers."""

    return _GROUP_SEPARATOR_PATTERN.sub("", value)


def _strip_currency_symbols(value: str) -> str:
    """Drop leading/trailing currency symbols while preserving sign."""

    if not value:
        return value

    sign = ""
    stripped = value
    if stripped and stripped[0] in "+-":
        sign, stripped = stripped[0], stripped[1:]
        stripped = stripped.lstrip()

    while stripped and unicodedata.category(stripped[0]) == "Sc":
        stripped = stripped[1:]
        stripped = stripped.lstrip()

    while stripped and unicodedata.category(stripped[-1]) == "Sc":
        stripped = stripped[:-1]
        stripped = stripped.rstrip()

    if not stripped:
        return sign

    return f"{sign}{stripped}" if sign else stripped


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
        candidate = _normalize_sign_characters(candidate)
        candidate = _strip_currency_symbols(candidate)
        candidate = _normalize_sign_characters(candidate)
        if candidate.startswith("(") and candidate.endswith(")"):
            inner = candidate[1:-1].strip()
            inner = _strip_currency_symbols(inner)
            inner = _normalize_sign_characters(inner)
            if inner:
                if inner[0] in "+-":
                    inner = inner[1:].lstrip()
                if inner:
                    candidate = f"-{inner}"
                else:
                    candidate = ""
        if not candidate:
            return default
        candidate = _strip_currency_symbols(candidate)
        candidate = _normalize_sign_characters(candidate)
        if not candidate or candidate in "+-":
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
        candidate = _normalize_sign_characters(candidate)
        candidate = _strip_currency_symbols(candidate)
        candidate = _normalize_sign_characters(candidate)
        if candidate.startswith("(") and candidate.endswith(")"):
            inner = candidate[1:-1].strip()
            inner = _strip_currency_symbols(inner)
            inner = _normalize_sign_characters(inner)
            if inner:
                if inner[0] in "+-":
                    inner = inner[1:].lstrip()
                if inner:
                    candidate = f"-{inner}"
                else:
                    candidate = ""
        if not candidate:
            return default
        candidate = _strip_currency_symbols(candidate)
        candidate = _normalize_sign_characters(candidate)
        if not candidate or candidate in "+-":
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
