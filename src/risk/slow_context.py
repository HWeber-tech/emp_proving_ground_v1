"""Slow-context size multiplier heuristics.

The phase-D roadmap calls for combining macro, volatility, and earnings
signals into a coarse position-size multiplier.  This module provides a small
helper that resolves the requested multiplier while constraining outputs to the
roadmap-approved set ``{0.0, 0.3, 1.0}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "SlowContextDecision",
    "resolve_size_multiplier",
    "resolve_slow_context_multiplier",
]

_ALLOWED_MULTIPLIERS: tuple[float, ...] = (0.0, 0.3, 1.0)

_MACRO_KEYS: tuple[str, ...] = (
    "macro_block",
    "macro_halt",
    "macro_freeze",
    "macro_event_active",
)
_VOL_KEYS: tuple[str, ...] = (
    "volatility_throttle",
    "volatility_gate",
    "volatility_block",
    "high_volatility",
)
_EARNINGS_KEYS: tuple[str, ...] = (
    "earnings_throttle",
    "earnings_blackout",
    "earnings_quiet_period",
    "earnings_gate",
)
_OVERRIDE_KEYS: tuple[str, ...] = (
    "size_multiplier",
    "size_multiplier_override",
    "slow_context_multiplier",
    "override_multiplier",
)
_TRUE_TOKENS = {
    "1",
    "true",
    "yes",
    "y",
    "on",
    "halt",
    "block",
    "freeze",
    "high",
    "warn",
    "alert",
    "storm",
    "stress",
    "stressed",
    "dislocated",
}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True, slots=True)
class SlowContextDecision:
    """Resolved slow-context posture for sizing."""

    multiplier: float
    drivers: Mapping[str, bool]


def resolve_size_multiplier(
    context: Mapping[str, object] | None = None,
) -> SlowContextDecision:
    """Map slow-context signals to a size multiplier in {0, 0.3, 1}."""

    payload: Mapping[str, object] = context or {}

    macro = _any_flag(payload, _MACRO_KEYS)
    volatility = _any_flag(payload, _VOL_KEYS)
    earnings = _any_flag(payload, _EARNINGS_KEYS)

    drivers = {
        "macro": macro,
        "volatility": volatility,
        "earnings": earnings,
    }

    override = _resolve_override(payload)
    if override is not None:
        return SlowContextDecision(multiplier=override, drivers=drivers)

    if macro:
        multiplier = 0.0
    elif volatility or earnings:
        multiplier = 0.3
    else:
        multiplier = 1.0

    return SlowContextDecision(multiplier=multiplier, drivers=drivers)


def resolve_slow_context_multiplier(
    context: Mapping[str, object] | None = None,
) -> SlowContextDecision:
    """Backward-compatible alias for :func:`resolve_size_multiplier`."""

    return resolve_size_multiplier(context)


def _any_flag(context: Mapping[str, object], keys: tuple[str, ...]) -> bool:
    return any(_coerce_flag(context.get(key)) for key in keys if key in context)


def _coerce_flag(value: object | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) != 0.0
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return False
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        try:
            return float(token) != 0.0
        except ValueError:
            return False
    return bool(value)


def _resolve_override(context: Mapping[str, object]) -> float | None:
    for key in _OVERRIDE_KEYS:
        candidate = _coerce_multiplier(context.get(key))
        if candidate is not None:
            return candidate
    return None


def _coerce_multiplier(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    for allowed in _ALLOWED_MULTIPLIERS:
        if abs(candidate - allowed) <= 1e-9:
            return allowed
    return None
