from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .market_regime import (
    MarketRegimeAssessment,
    apply_regime_adjustment,
)

from src.data_foundation.config.execution_config import ExecutionConfig


def _safe_float(value: object) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


@dataclass
class ExecContext:
    spread: float
    top_imbalance: float
    sigma_ann: float
    size_ratio: float


def estimate_slippage_bps(
    ctx: ExecContext,
    cfg: ExecutionConfig,
    *,
    regime_assessment: MarketRegimeAssessment | None = None,
) -> float:
    # Simple monotonic function consistent with test expectations
    s = max(ctx.spread, 0.0)
    imb = max(ctx.top_imbalance, 0.0)
    sig = max(ctx.sigma_ann, 0.0)
    sz = max(ctx.size_ratio, 0.0)
    base = getattr(cfg.slippage, "base_bps", 0.0)
    spread_coef = getattr(cfg.slippage, "spread_coef", 0.0)
    imbalance_coef = getattr(cfg.slippage, "imbalance_coef", 0.0)
    sigma_coef = getattr(cfg.slippage, "sigma_coef", 0.0)
    size_coef = getattr(cfg.slippage, "size_coef", 0.0)
    estimate = float(
        base + spread_coef * s + imbalance_coef * imb + sigma_coef * sig + size_coef * sz
    )
    if regime_assessment is not None:
        return apply_regime_adjustment(estimate, regime_assessment)
    return estimate


def estimate_commission_bps(cfg: ExecutionConfig) -> float:
    return max(0.0, float(cfg.fees.commission_bps))


def calculate_edge_ticks(
    delta_hat: float | int | None,
    spread_ticks: float | int | None,
    *,
    spread_floor: float | int | None = None,
) -> float:
    """Return expected edge in ticks given a normalised price delta.

    Parameters
    ----------
    delta_hat:
        Normalised price change signal (dimensionless).
    spread_ticks:
        Observed bid/ask spread expressed in ticks (or price units when tick
        size is unknown).
    spread_floor:
        Optional minimum spread floor in ticks used when the observed spread is
        narrower than a volatility-derived bound.
    """

    delta = _safe_float(delta_hat)
    spread = _safe_float(spread_ticks)
    floor = _safe_float(spread_floor) if spread_floor is not None else 0.0

    if delta is None or spread is None:
        return 0.0

    effective_spread = max(spread, floor if floor is not None else 0.0, 0.0)
    if effective_spread == 0.0:
        return 0.0
    return delta * effective_spread


def estimate_total_cost_ticks(
    spread_ticks: float | int | None,
    *,
    slippage: float | int | None = None,
    fees: float | int | None = None,
    adverse_selection_penalty: float | int | None = None,
) -> float:
    """Estimate total cost in ticks using the roadmap D.1.1 rule."""

    spread = _safe_float(spread_ticks)
    spread_component = 0.0
    if spread is not None:
        spread_component = 0.5 * max(spread, 0.0)

    total = spread_component
    for component in (slippage, fees, adverse_selection_penalty):
        value = _safe_float(component)
        if value is None:
            continue
        total += value
    return total


_EVENT_HORIZON_ALIAS_MAP: dict[str, str] = {
    "1": "ev1",
    "01": "ev1",
    "ev1": "ev1",
    "5": "ev5",
    "05": "ev5",
    "ev5": "ev5",
    "20": "ev20",
    "ev20": "ev20",
}

_TIME_HORIZON_ALIAS_MAP: dict[str, str] = {
    "100ms": "100ms",
    "0.1s": "100ms",
    "0.10s": "100ms",
    "100": "100ms",
    "0.100s": "100ms",
    "500ms": "500ms",
    "0.5s": "500ms",
    "0.50s": "500ms",
    "500": "500ms",
    "2s": "2s",
    "2.0s": "2s",
    "2.00s": "2s",
    "2000ms": "2s",
    "2000": "2s",
}

_CANONICAL_EVENT_HORIZON_ORDER: tuple[str, ...] = ("ev1", "ev5", "ev20")
_CANONICAL_TIME_HORIZON_ORDER: tuple[str, ...] = ("100ms", "500ms", "2s")


def _canonicalise_event_horizon_label(value: object) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        rounded = round(value)
        if rounded <= 0:
            return None
        if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12):
            key = str(int(rounded))
            return _EVENT_HORIZON_ALIAS_MAP.get(key, f"ev{int(rounded)}")
        return None

    text = str(value).strip()
    if not text:
        return None

    lower = text.lower()
    alias = _EVENT_HORIZON_ALIAS_MAP.get(lower)
    if alias is not None:
        return alias
    if lower.startswith("ev") and lower[2:].isdigit():
        suffix = int(lower[2:])
        if suffix > 0:
            return f"ev{suffix}"
        return None
    if lower.isdigit():
        suffix = int(lower)
        if suffix > 0:
            return f"ev{suffix}"
    return None


def _canonicalise_time_horizon_label(value: object) -> str | None:
    numeric_ms: float | None = None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        numeric = float(value)
        numeric_ms = numeric if abs(numeric) >= 50.0 else numeric * 1000.0
    else:
        text = str(value).strip().lower()
        if not text:
            return None
        alias = _TIME_HORIZON_ALIAS_MAP.get(text)
        if alias is not None:
            return alias
        if text.endswith("ms"):
            try:
                numeric_ms = float(text[:-2])
            except ValueError:
                return None
        elif text.endswith("s"):
            try:
                numeric_ms = float(text[:-1]) * 1000.0
            except ValueError:
                return None
        else:
            try:
                numeric = float(text)
            except ValueError:
                return None
            if not math.isfinite(numeric):
                return None
            numeric_ms = numeric if abs(numeric) >= 50.0 else numeric * 1000.0

    if numeric_ms is None or numeric_ms <= 0.0:
        return None

    if math.isclose(numeric_ms, 100.0, rel_tol=0.0, abs_tol=1e-9):
        return "100ms"
    if math.isclose(numeric_ms, 500.0, rel_tol=0.0, abs_tol=1e-9):
        return "500ms"
    if math.isclose(numeric_ms, 2000.0, rel_tol=0.0, abs_tol=1e-9):
        return "2s"
    return None


def _resolve_component_value(
    source: float | int | Mapping[object, object] | None,
    canonical_label: str,
    original_label: object,
) -> float | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        candidates: list[object] = [canonical_label]
        candidates.append(canonical_label.lower())
        candidates.append(canonical_label.upper())
        if isinstance(original_label, str):
            candidates.extend(
                [
                    original_label,
                    original_label.lower(),
                    original_label.upper(),
                ]
            )
        else:
            candidates.append(original_label)
        for candidate in candidates:
            if candidate in source:
                return _safe_float(source[candidate])
        return None
    return _safe_float(source)


def _order_by_canonical(
    values: Mapping[str, float],
    canonical_order: Sequence[str],
) -> dict[str, float]:
    ordered: dict[str, float] = {}
    for label in canonical_order:
        value = values.get(label)
        if value is not None:
            ordered[label] = value
    for label, value in values.items():
        if label not in ordered:
            ordered[label] = value
    return ordered


def calculate_dimensionless_delta_hat(
    mid_now: float | int | None,
    mid_future: float | int | None,
    tick_size: float | int | None,
    spread_ticks: float | int | None,
    *,
    spread_floor_ticks: float | int | None = None,
) -> float:
    """Normalise the forward mid-price change by the effective spread in ticks.

    Implements roadmap task **B.3.2**::

        delta_hat = (mid[t+H] - mid[t]) / (tick * max(spread, kσ))

    Parameters
    ----------
    mid_now:
        Observed mid-price at time ``t``.
    mid_future:
        Forward-looking mid-price at horizon ``t + H``.
    tick_size:
        Minimum price increment (``tick``) for the instrument.
    spread_ticks:
        Bid/ask spread expressed in ticks.
    spread_floor_ticks:
        Optional volatility-derived floor ``kσ`` in ticks.
    """

    mid_current = _safe_float(mid_now)
    mid_horizon = _safe_float(mid_future)
    tick = _safe_float(tick_size)
    spread = _safe_float(spread_ticks)
    spread_floor = _safe_float(spread_floor_ticks) if spread_floor_ticks is not None else None

    if (
        mid_current is None
        or mid_horizon is None
        or tick is None
        or tick <= 0.0
    ):
        return 0.0

    spread_value = spread if spread is not None else 0.0
    effective_spread_ticks = max(
        spread_value,
        spread_floor if spread_floor is not None else 0.0,
        0.0,
    )
    if effective_spread_ticks <= 0.0:
        return 0.0

    denom = tick * effective_spread_ticks
    if denom == 0.0:
        return 0.0

    return (mid_horizon - mid_current) / denom


def calculate_dual_horizon_delta_hats(
    mid_now: float | int | None,
    *,
    tick_size: float | int | None,
    event_mid_prices: Mapping[object, object] | None = None,
    wall_time_mid_prices: Mapping[object, object] | None = None,
    event_spread_ticks: float | int | Mapping[object, object] | None = None,
    wall_time_spread_ticks: float | int | Mapping[object, object] | None = None,
    event_spread_floor_ticks: float | int | Mapping[object, object] | None = None,
    wall_time_spread_floor_ticks: float | int | Mapping[object, object] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute dimensionless deltas for roadmap dual horizons (B.3.3).

    The caller supplies forward-looking mid-prices bucketed by event-time and
    wall-time horizons.  This helper normalises each horizon into the
    canonical labels ``ev1``/``ev5``/``ev20`` and ``100ms``/``500ms``/``2s``
    before evaluating :func:`calculate_dimensionless_delta_hat`.  Unknown
    horizons or non-finite inputs are ignored so vendor payloads can be passed
    through without bespoke filtering.
    """

    mid_current = _safe_float(mid_now)
    tick = _safe_float(tick_size)
    if mid_current is None or tick is None or tick <= 0.0:
        return {"event": {}, "time": {}}

    event_mid_prices = event_mid_prices or {}
    wall_time_mid_prices = wall_time_mid_prices or {}

    event_values: dict[str, float] = {}
    for original_label, mid_future in event_mid_prices.items():
        canonical = _canonicalise_event_horizon_label(original_label)
        if canonical is None:
            continue
        future_mid = _safe_float(mid_future)
        if future_mid is None:
            continue
        spread = _resolve_component_value(event_spread_ticks, canonical, original_label)
        floor = _resolve_component_value(event_spread_floor_ticks, canonical, original_label)
        delta = calculate_dimensionless_delta_hat(
            mid_current,
            future_mid,
            tick,
            spread,
            spread_floor_ticks=floor,
        )
        event_values[canonical] = delta

    time_values: dict[str, float] = {}
    for original_label, mid_future in wall_time_mid_prices.items():
        canonical = _canonicalise_time_horizon_label(original_label)
        if canonical is None:
            continue
        future_mid = _safe_float(mid_future)
        if future_mid is None:
            continue
        spread = _resolve_component_value(wall_time_spread_ticks, canonical, original_label)
        floor = _resolve_component_value(wall_time_spread_floor_ticks, canonical, original_label)
        delta = calculate_dimensionless_delta_hat(
            mid_current,
            future_mid,
            tick,
            spread,
            spread_floor_ticks=floor,
        )
        time_values[canonical] = delta

    return {
        "event": _order_by_canonical(event_values, _CANONICAL_EVENT_HORIZON_ORDER),
        "time": _order_by_canonical(time_values, _CANONICAL_TIME_HORIZON_ORDER),
    }


__all__ = [
    "ExecContext",
    "estimate_slippage_bps",
    "estimate_commission_bps",
    "calculate_dimensionless_delta_hat",
    "calculate_edge_ticks",
    "estimate_total_cost_ticks",
    "calculate_dual_horizon_delta_hats",
]
