"""Adverse selection diagnostics based on microprice drift.

This module fulfils roadmap task **D.2.2** by computing the drift in the
microprice over a fixed event horizon, conditioned on whether the event was
triggered by our action (for example, we sent an order or crossed the book).
The helpers operate on lightweight mappings so they can be used inside live
monitors as well as in offline research notebooks without requiring pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import StatisticsError, fmean, median
from types import MappingProxyType
from typing import Mapping, MutableMapping, Sequence

import math

__all__ = ["MicropriceDriftResult", "compute_microprice_drift"]


_BUY_ALIASES = {
    "b",
    "bid",
    "buy",
    "lift",
    "long",
    "take",
}
_SELL_ALIASES = {
    "a",
    "ask",
    "hit",
    "sell",
    "short",
}
_TRUE_ALIASES = {"true", "t", "yes", "y", "1"}
_FALSE_ALIASES = {"false", "f", "no", "n", "0"}


@dataclass(frozen=True, slots=True)
class _Event:
    """Normalized view of an execution event used for drift analysis."""

    microprice: float | None
    our_action: bool
    side: str | None  # "buy", "sell", or None when unknown


@dataclass(frozen=True, slots=True)
class MicropriceDriftResult:
    """Aggregate microprice drift metrics for a fixed event horizon."""

    horizon_events: int
    samples: int
    mean_drift: float
    median_drift: float | None
    positive_fraction: float | None
    mean_by_side: Mapping[str, float | None]
    count_by_side: Mapping[str, int]

    def as_dict(self) -> Mapping[str, object]:
        """Return a serialisable representation of the drift metrics."""

        return {
            "horizon_events": self.horizon_events,
            "samples": self.samples,
            "mean_drift": self.mean_drift,
            "median_drift": self.median_drift,
            "positive_fraction": self.positive_fraction,
            "mean_by_side": dict(self.mean_by_side),
            "count_by_side": dict(self.count_by_side),
        }


def compute_microprice_drift(
    events: Sequence[Mapping[str, object]],
    horizon_events: int,
    *,
    microprice_key: str = "microprice",
    our_action_key: str = "our_action",
    side_key: str = "side",
    action_key: str = "action",
) -> MicropriceDriftResult:
    """Compute microprice drift conditioned on our action over a fixed horizon.

    Parameters
    ----------
    events:
        Chronologically ordered execution or order-book events. Each mapping is
        expected to expose a microprice value and a flag indicating whether the
        event corresponded to our action (for example, we sent an order or
        crossed the spread).
    horizon_events:
        Number of subsequent events to look ahead when measuring drift. Must be
        strictly positive.
    microprice_key:
        Mapping key that stores the microprice for each event.
    our_action_key:
        Mapping key that indicates whether the event was triggered by our
        action. When missing or falsy, the helper attempts to infer the flag
        from the action string.
    side_key:
        Mapping key that stores the side of the event (e.g. "buy" or "sell").
    action_key:
        Mapping key that stores a textual action description such as "BUY" or
        "SELL". The helper uses this field both to infer the side and to detect
        whether the event belongs to us when ``our_action_key`` is absent.
    """

    if horizon_events <= 0:
        raise ValueError("horizon_events must be a positive integer")

    normalized: list[_Event] = []
    for raw in events:
        normalized.append(
            _normalize_event(
                raw,
                microprice_key=microprice_key,
                our_action_key=our_action_key,
                side_key=side_key,
                action_key=action_key,
            )
        )

    drifts: list[float] = []
    side_samples: dict[str, list[float]] = {"buy": [], "sell": [], "unknown": []}

    for idx, event in enumerate(normalized):
        if not event.our_action:
            continue
        if event.microprice is None:
            continue

        future_index = idx + horizon_events
        if future_index >= len(normalized):
            continue

        future_event = normalized[future_index]
        future_price = future_event.microprice
        if future_price is None:
            continue

        drift = future_price - event.microprice
        drifts.append(drift)

        side_label = event.side if event.side in {"buy", "sell"} else "unknown"
        side_samples[side_label].append(drift)

    samples = len(drifts)
    if samples:
        mean_drift = fmean(drifts)
        try:
            median_drift = median(drifts)
        except StatisticsError:
            median_drift = drifts[0]
        positive_fraction = sum(1 for value in drifts if value > 0.0) / samples
    else:
        mean_drift = 0.0
        median_drift = None
        positive_fraction = None

    mean_by_side: MutableMapping[str, float | None] = {}
    count_by_side: MutableMapping[str, int] = {}
    for label, collection in side_samples.items():
        count_by_side[label] = len(collection)
        if collection:
            mean_by_side[label] = fmean(collection)
        else:
            mean_by_side[label] = None

    return MicropriceDriftResult(
        horizon_events=horizon_events,
        samples=samples,
        mean_drift=float(mean_drift),
        median_drift=None if median_drift is None else float(median_drift),
        positive_fraction=positive_fraction,
        mean_by_side=MappingProxyType(dict(mean_by_side)),
        count_by_side=MappingProxyType(dict(count_by_side)),
    )


def _normalize_event(
    payload: Mapping[str, object],
    *,
    microprice_key: str,
    our_action_key: str,
    side_key: str,
    action_key: str,
) -> _Event:
    if not isinstance(payload, Mapping):
        raise TypeError("events must be mappings with microprice metadata")

    microprice = _coerce_microprice(payload.get(microprice_key))
    action_text = payload.get(action_key)
    side_value = payload.get(side_key)

    side = _normalise_side(side_value)
    if side is None:
        side = _normalise_side(action_text)
    if side is None and "trade_sign" in payload:
        side = _normalise_side(payload.get("trade_sign"))
    if side is None and "signed_quantity" in payload:
        side = _normalise_side(payload.get("signed_quantity"))
    if side is None and "quantity" in payload:
        side = _normalise_side(payload.get("quantity"))

    our_action_flag = payload.get(our_action_key)
    if our_action_flag is None:
        our_action = _infer_action_from_text(action_text)
    else:
        our_action = _coerce_bool(our_action_flag)
        if not our_action and _infer_action_from_text(action_text):
            our_action = True

    return _Event(microprice=microprice, our_action=our_action, side=side)


def _coerce_microprice(value: object) -> float | None:
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(price):
        return None
    return price


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, bool):  # bool is subclass of int, keep explicit
            return bool(value)
        if math.isnan(float(value)):
            return False
        return float(value) != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_ALIASES:
            return True
        if text in _FALSE_ALIASES:
            return False
        return False
    return False


def _infer_action_from_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    tokens = {token for token in text.replace("_", " ").split() if token}
    if tokens & (_BUY_ALIASES | _SELL_ALIASES):
        return True
    if text in {"cross", "post", "chase", "market"}:
        return True
    return False


def _normalise_side(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in _BUY_ALIASES or text.startswith("buy") or text.startswith("long"):
            return "buy"
        if text in _SELL_ALIASES or text.startswith("sell") or text.startswith("short"):
            return "sell"
        return None
    if isinstance(value, bool):
        return "buy" if value else "sell"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if numeric > 0:
            return "buy"
        if numeric < 0:
            return "sell"
        return None
    return None
