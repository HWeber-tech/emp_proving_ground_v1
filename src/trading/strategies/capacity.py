"""Liquidity-aware capacity helpers for strategy sizing limits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterator

DEFAULT_L1_CAPACITY_RATIO = 0.02
"""Default maximum fraction of the L1 depth percentile allocated to a strategy."""

_CANDIDATE_PATHS: tuple[tuple[str, ...], ...] = (
    ("liquidity", "l1", "depth_percentiles", "p50"),
    ("liquidity", "depth_percentiles", "l1", "p50"),
    ("order_book", "l1", "depth_percentiles", "p50"),
    ("order_book", "depth_percentiles", "l1", "p50"),
    ("market_depth", "l1", "percentiles", "p50"),
    ("market_depth", "percentiles", "l1", "p50"),
    ("depth_percentiles", "l1", "p50"),
    ("depth", "l1", "p50"),
    ("l1_depth_percentiles", "p50"),
    ("l1_depth", "percentiles", "p50"),
    ("l1_depth_percentile",),
    ("l1_depth_p50",),
)


def resolve_l1_depth_cap(
    market_data: Mapping[str, Any],
    symbol: str,
    *,
    ratio: float = DEFAULT_L1_CAPACITY_RATIO,
) -> tuple[float | None, dict[str, object]]:
    """Return a (cap, metadata) tuple enforcing an L1 depth percentile budget.

    Parameters
    ----------
    market_data:
        Composite market data mapping passed to strategy ``generate_signal`` calls.
    symbol:
        Instrument identifier to inspect within ``market_data``.
    ratio:
        Fraction of the detected L1 depth percentile allocated to the strategy.

    Returns
    -------
    (cap, metadata)
        ``cap`` is ``None`` when no qualifying percentile is discovered. ``metadata``
        contains diagnostic information describing the basis for the cap when
        available, otherwise an empty ``dict``.
    """

    payload = market_data.get(symbol)
    if not isinstance(payload, Mapping):
        return None, {}

    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None, {}

    value, path = _resolve_depth_percentile(payload)
    if value is None:
        return None, {}

    cap = ratio * value
    if not math.isfinite(cap) or cap <= 0.0:
        return None, {}

    basis_path = ".".join(path)
    metadata: dict[str, object] = {
        "cap": float(cap),
        "cap_ratio": ratio,
        "basis_value": float(value),
        "basis_path": basis_path,
    }
    return cap, metadata


def _resolve_depth_percentile(payload: Mapping[str, Any]) -> tuple[float | None, tuple[str, ...]]:
    for path in _CANDIDATE_PATHS:
        value, resolved_path = _lookup_path(payload, path)
        if value is not None:
            return value, resolved_path

    for value, resolved_path in _scan_depth_percentiles(payload):
        return value, resolved_path

    return None, ()


def _lookup_path(
    mapping: Mapping[str, Any], path: tuple[str, ...]
) -> tuple[float | None, tuple[str, ...]]:
    current: Any = mapping
    resolved_keys: list[str] = []

    for element in path:
        if not isinstance(current, Mapping):
            return None, ()
        key = _match_key(current, element)
        if key is None:
            return None, ()
        resolved_keys.append(str(key))
        current = current[key]

    value = _to_positive_float(current)
    if value is not None:
        return value, tuple(resolved_keys)

    if isinstance(current, Mapping):
        for percentile_key in ("p50", "median", "0.5", "50", "p0_5", "percentile_50"):
            key = _match_key(current, percentile_key)
            if key is None:
                continue
            candidate = _to_positive_float(current[key])
            if candidate is not None:
                resolved_keys.append(str(key))
                return candidate, tuple(resolved_keys)

    return None, ()


def _scan_depth_percentiles(
    mapping: Mapping[str, Any],
    path: tuple[str, ...] = (),
) -> Iterator[tuple[float, tuple[str, ...]]]:
    for key, value in mapping.items():
        next_path = path + (str(key),)
        if isinstance(value, Mapping):
            yield from _scan_depth_percentiles(value, next_path)
            continue

        key_norm = str(key).lower()
        path_norm = ".".join(part.lower() for part in next_path)
        if "percentile" not in key_norm:
            continue
        if "l1" not in path_norm and "level1" not in path_norm and "l1" not in key_norm:
            continue

        candidate = _to_positive_float(value)
        if candidate is not None:
            yield candidate, next_path


def _match_key(mapping: Mapping[str, Any], target: str) -> str | None:
    target_norm = str(target).lower()

    for key in mapping.keys():
        candidate_norm = str(key).lower()
        if candidate_norm == target_norm:
            return str(key)

    for key in mapping.keys():
        candidate_norm = str(key).lower()
        if target_norm == "l1":
            if "l1" in candidate_norm or "level1" in candidate_norm or "lvl1" in candidate_norm:
                return str(key)
        elif target_norm == "percentiles":
            if "percentile" in candidate_norm:
                return str(key)
        elif target_norm == "percentile_50":
            if "percentile" in candidate_norm and "50" in candidate_norm:
                return str(key)
        else:
            if target_norm in candidate_norm:
                return str(key)
    return None


def _to_positive_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate) or candidate <= 0.0:
        return None
    return candidate


__all__ = ["DEFAULT_L1_CAPACITY_RATIO", "resolve_l1_depth_cap"]
