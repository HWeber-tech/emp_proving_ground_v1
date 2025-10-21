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
_GROUP_CHAR_TRANSLATION = str.maketrans("", "", ",._'")


def _normalize_sign_characters(value: str) -> str:
    return value.translate(_SIGN_NORMALIZATION)


def _strip_group_separators(value: str) -> str:
    """Remove common locale-specific grouping separators embedded in numbers."""

    return _GROUP_SEPARATOR_PATTERN.sub("", value)


def _remove_group_characters(value: str) -> str:
    """Drop common grouping punctuation from ``value``."""

    return value.translate(_GROUP_CHAR_TRANSLATION)


def _normalize_decimal_markers(value: str) -> str:
    """Coerce locale-specific decimal markers to ``.`` while preserving sign."""

    if not value:
        return value

    stripped = _strip_group_separators(value)
    if "," not in stripped:
        return stripped

    sign = ""
    rest = stripped
    if rest[0] in "+-":
        sign, rest = rest[0], rest[1:]

    if not rest:
        return sign

    last_comma = rest.rfind(",")
    last_dot = rest.rfind(".")
    if last_dot > last_comma:
        head = _remove_group_characters(rest[:last_dot])
        tail = rest[last_dot:]
        return f"{sign}{head}{tail}"

    if rest.count(",") > 1:
        return f"{sign}{rest.replace(',', '')}"

    integer_part = rest[:last_comma]
    fractional_part = rest[last_comma + 1 :]

    integer_digits = _remove_group_characters(integer_part)
    fractional_digits = _remove_group_characters(fractional_part)
    digits_before = len([ch for ch in integer_part if ch.isdigit()])
    digits_after = len([ch for ch in fractional_part if ch.isdigit()])
    integer_is_zero = integer_digits.strip("0") == ""

    treat_as_decimal = False
    if "." in integer_part:
        treat_as_decimal = True
    elif digits_after and (
        digits_after != 3 or digits_before > 3 or integer_is_zero
    ):
        treat_as_decimal = True

    if not treat_as_decimal:
        return f"{sign}{rest.replace(',', '')}"

    if not integer_digits:
        integer_digits = "0"
    if not fractional_digits:
        fractional_digits = "0"

    return f"{sign}{integer_digits}.{fractional_digits}"


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


def _normalize_decimal_separator(value: str) -> str:
    """Convert locale-specific decimal delimiters to the canonical period."""

    if "," not in value:
        return value

    decimal_is_comma = False
    if "." in value:
        if value.rfind(",") > value.rfind("."):
            decimal_is_comma = True
    else:
        head, _, tail = value.rpartition(",")
        if head and tail and len(tail) in (1, 2):
            decimal_is_comma = True

    if not decimal_is_comma:
        return value

    head, _, tail = value.rpartition(",")
    sanitized_head = head.replace(".", "").replace(",", "")
    return f"{sanitized_head}.{tail}"


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
        trailing_sign: str | None = None
        if candidate and candidate[-1] in "+-":
            core = candidate[:-1].strip()
            if core and core[0] not in "+-":
                trailing_sign = candidate[-1]
                candidate = core
        percent = False
        if candidate.endswith("%"):
            percent = True
            candidate = candidate[:-1].strip()
            candidate = _strip_currency_symbols(candidate)
            candidate = _normalize_sign_characters(candidate)
        if candidate and candidate[-1] in "+-":
            core = candidate[:-1].strip()
            if core and core[0] not in "+-" and trailing_sign is None:
                trailing_sign = candidate[-1]
                candidate = core
        if not candidate or candidate in "+-":
            return default
        candidate = _normalize_decimal_markers(candidate)
        normalized = _NUMERIC_SEPARATOR_PATTERN.sub("", candidate)
        normalized = _strip_group_separators(normalized)
        try:
            result = float(normalized)
        except ValueError:
            return default
        if not math.isfinite(result):
            return default
        if percent:
            result /= 100.0
        if trailing_sign == "-":
            result = -result
        return result
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
        trailing_sign: str | None = None
        if candidate and candidate[-1] in "+-":
            core = candidate[:-1].strip()
            if core and core[0] not in "+-":
                trailing_sign = candidate[-1]
                candidate = core
        if not candidate or candidate in "+-":
            return default
        normalized = _NUMERIC_SEPARATOR_PATTERN.sub("", candidate)
        normalized = _strip_group_separators(normalized)
        if trailing_sign == "-":
            if normalized.startswith("+"):
                normalized = f"-{normalized[1:]}"
            elif not normalized.startswith("-"):
                normalized = f"-{normalized}"
        elif trailing_sign == "+" and not normalized.startswith(("+", "-")):
            normalized = f"+{normalized}"
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
