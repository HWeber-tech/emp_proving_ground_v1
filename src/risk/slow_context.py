"""Slow-context size multiplier heuristics.

The phase-D roadmap calls for combining macro, volatility, and earnings
signals into a coarse position-size multiplier.  This module provides a small
helper that resolves the requested multiplier while constraining outputs to the
roadmap-approved set ``{0.0, 0.3, 1.0}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

__all__ = [
    "SlowContextDecision",
    "resolve_size_multiplier",
    "resolve_slow_context_multiplier",
]

_ALLOWED_MULTIPLIERS: tuple[float, ...] = (0.0, 0.3, 1.0)
_MACRO_PROXIMITY_SECONDS: float = 120.0
_HIGH_VIX_THRESHOLD: float = 35.0

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
    reason: str
    seconds_to_macro_event: float | None = None
    vix: float | None = None


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
        return SlowContextDecision(
            multiplier=override,
            drivers=drivers,
            reason="override",
        )

    if macro:
        multiplier = 0.0
        reason = "macro_signal"
    elif volatility:
        multiplier = 0.3
        reason = "volatility_signal"
    elif earnings:
        multiplier = 0.3
        reason = "earnings_signal"
    else:
        multiplier = 1.0
        reason = "normal"

    return SlowContextDecision(
        multiplier=multiplier,
        drivers=drivers,
        reason=reason,
    )


def resolve_slow_context_multiplier(
    *,
    as_of: datetime | None = None,
    macro_events: Sequence[object] | None = None,
    vix_value: object | None = None,
) -> SlowContextDecision:
    """Resolve slow-context multiplier using macro-event proximity and VIX."""

    reference = _coerce_datetime(as_of) or datetime.now(timezone.utc)
    events = _normalise_macro_events(macro_events)
    seconds_to_event = _seconds_to_next_event(reference, events)
    vix = _coerce_float(vix_value)

    macro_trigger = (
        seconds_to_event is not None and seconds_to_event <= _MACRO_PROXIMITY_SECONDS
    )
    volatility_trigger = vix is not None and vix >= _HIGH_VIX_THRESHOLD

    if macro_trigger:
        multiplier = 0.0
        reason = "macro_event_proximity"
    elif volatility_trigger:
        multiplier = 0.3
        reason = "high_volatility"
    else:
        multiplier = 1.0
        reason = "normal"

    drivers = {
        "macro": macro_trigger,
        "volatility": volatility_trigger,
        "earnings": False,
    }

    return SlowContextDecision(
        multiplier=multiplier,
        drivers=drivers,
        reason=reason,
        seconds_to_macro_event=seconds_to_event,
        vix=vix,
    )


def _coerce_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if hasattr(value, "to_pydatetime"):
        try:
            converted = value.to_pydatetime()
        except Exception:  # pragma: no cover - dtype guard
            return None
        if isinstance(converted, datetime):
            return _coerce_datetime(converted)
    return None


def _seconds_to_next_event(reference: datetime, events: Sequence[datetime]) -> float | None:
    deltas: list[float] = []
    for event in events:
        delta = (event - reference).total_seconds()
        if delta >= 0.0:
            deltas.append(delta)
    if not deltas:
        return None
    return min(deltas)


def _normalise_macro_events(source: Sequence[object] | object | None) -> list[datetime]:
    if source is None:
        return []

    stack: list[object] = [source]
    seen: set[int] = set()
    collected: list[datetime] = []

    while stack:
        item = stack.pop()
        if item is None:
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)

        coerced = _coerce_datetime(item)
        if coerced is not None:
            collected.append(coerced)
            continue

        if isinstance(item, Mapping):
            for key in ("timestamp", "ts", "time", "event_time", "datetime", "start"):
                if key in item:
                    stack.append(item[key])
            for key in ("events", "macro_events", "items", "upcoming"):
                if key in item:
                    stack.append(item[key])
            continue

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            stack.extend(item)
            continue

    unique: dict[str, datetime] = {}
    for timestamp in collected:
        fingerprint = timestamp.isoformat()
        existing = unique.get(fingerprint)
        if existing is None or timestamp < existing:
            unique[fingerprint] = timestamp
    return sorted(unique.values())


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


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
