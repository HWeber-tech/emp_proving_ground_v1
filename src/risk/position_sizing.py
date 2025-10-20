"""Canonical position sizing helpers for the risk subsystem."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any


QuantilePayload = Mapping[str, Any] | Sequence[Any] | float | int


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute the Kelly Criterion fraction for position sizing."""

    denominator = max(avg_loss, 1e-9)
    edge = max(0.0, min(1.0, win_rate))
    opportunity = avg_win / denominator
    risk = 1.0 - edge
    if opportunity == 0:
        return 0.0
    fraction = (opportunity * edge - risk) / opportunity
    return max(0.0, min(1.0, fraction))


def position_size(balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal) -> Decimal:
    """Determine the allowed position size for a given risk budget."""

    risk_amount = balance * risk_per_trade
    divisor = max(Decimal("1e-9"), stop_loss_pct)
    return risk_amount / divisor


def _coerce_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def _coerce_quantile_key(key: Any) -> float | None:
    if isinstance(key, (int, float)):
        candidate = float(key)
    elif isinstance(key, str):
        match = re.search(r"-?\d+(?:\.\d+)?", key)
        if not match:
            return None
        candidate = float(match.group(0))
        if "%" in key and candidate > 1.0:
            candidate /= 100.0
        elif candidate > 1.0:
            # Treat bare integers like 25 as percentages.
            candidate /= 100.0
    else:
        return None
    candidate = round(candidate, 2)
    if candidate in (0.25, 0.5, 0.75):
        return candidate
    return None


def normalise_quantile_triplet(payload: QuantilePayload) -> tuple[float, float, float] | None:
    """Normalise heterogeneous quantile payloads to (q25, q50, q75).

    The helper accepts mappings (with keys such as ``q25`` or ``0.25``),
    sequences of ``(quantile, value)`` pairs, or bare numeric triplets.
    Returns ``None`` when the payload cannot be parsed.
    """

    # Handle direct triplet of numeric values first.
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if len(payload) == 3 and all(_coerce_float(item) is not None for item in payload):
            values = [_coerce_float(item) for item in payload]
            if None not in values:
                q25, q50, q75 = sorted(values)  # type: ignore[arg-type]
                return (float(q25), float(q50), float(q75))

    entries: dict[float, float] = {}

    def _register(key: Any, value: Any) -> None:
        quantile = _coerce_quantile_key(key)
        val = _coerce_float(value)
        if quantile is None or val is None:
            return
        entries.setdefault(quantile, val)

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _register(key, value)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            if isinstance(item, Mapping):
                nested = normalise_quantile_triplet(item)
                if nested is not None:
                    return nested
            elif isinstance(item, Sequence) and len(item) >= 2:
                key, value = item[0], item[1]
                _register(key, value)

    if {0.25, 0.5, 0.75}.issubset(entries):
        lower = entries[0.25]
        median = entries[0.5]
        upper = entries[0.75]

        lower = float(lower)
        median = float(median)
        upper = float(upper)

        # Enforce monotonic ordering without discarding the original values.
        if median < lower:
            median = lower
        if median > upper:
            median = upper
        if lower > upper:
            lower, upper = upper, lower

        return (lower, median, upper)

    return None


def quantile_edge_ratio(payload: QuantilePayload) -> float:
    """Compute an edge-to-uncertainty ratio from quantile predictions.

    The ratio is defined as ``median / max(iqr, eps)`` where ``iqr`` is the
    interquartile range (q75 - q25). Returns ``0.0`` when the payload cannot be
    parsed or the interquartile range is effectively zero.
    """

    triplet = normalise_quantile_triplet(payload)
    if triplet is None:
        return 0.0
    q25, q50, q75 = triplet
    dispersion = max(abs(q75 - q25), 1e-9)
    ratio = q50 / dispersion
    if not math.isfinite(ratio):
        return 0.0
    return ratio


__all__ = [
    "kelly_fraction",
    "position_size",
    "normalise_quantile_triplet",
    "quantile_edge_ratio",
]
