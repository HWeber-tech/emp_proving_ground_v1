"""Yield curve analytics backing the WHY sensory dimension."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class YieldSignalSnapshot:
    """Structured representation of the latest yield-curve state."""

    direction: float
    confidence: float
    slope_2s10s: float | None
    slope_5s30s: float | None
    curvature_2_10_30: float | None
    parallel_shift: float | None
    inversion_risk: float
    regime: str
    tenors: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | None]:
        payload: dict[str, float | None] = {
            "direction": float(self.direction),
            "confidence": float(self.confidence),
            "slope_2s10s": self.slope_2s10s,
            "slope_5s30s": self.slope_5s30s,
            "curvature_2_10_30": self.curvature_2_10_30,
            "parallel_shift": self.parallel_shift,
            "inversion_risk": self.inversion_risk,
            "regime": self.regime,
        }
        for tenor, value in self.tenors.items():
            payload[f"tenor_{tenor.lower()}"] = value
        return payload


@dataclass
class YieldSlopeTracker:
    """Tracks latest tenor values and computes slope/curvature features."""

    short_tenor: str = "2Y"
    long_tenor: str = "10Y"
    latest_values: dict[str, float] = field(default_factory=dict)

    def _normalize_tenor(self, tenor: str | None) -> str | None:
        if tenor is None:
            return None
        normalized = str(tenor).upper().strip()
        if normalized.endswith("YR"):
            normalized = normalized[:-2] + "Y"
        elif normalized.endswith("YEAR"):
            normalized = normalized[:-4] + "Y"
        return normalized or None

    def update(self, tenor: str | None, value: float | int | str | None) -> None:
        key = self._normalize_tenor(tenor)
        if not key or value is None:
            return
        try:
            self.latest_values[key] = float(value)
        except (TypeError, ValueError):
            # Ignore malformed updates while keeping existing state
            return

    def update_many(
        self, entries: Mapping[str, float | int | str | None] | Iterable[tuple[str, float]]
    ) -> None:
        if isinstance(entries, Mapping):
            iterator = entries.items()
        else:
            iterator = entries
        for tenor, value in iterator:
            self.update(tenor, value)

    def signal(self) -> tuple[float, float]:
        """Return a directional steepening signal and confidence."""

        slope = self.slope(self.short_tenor, self.long_tenor)
        if slope is None:
            return 0.0, 0.0
        slope_bps = abs(slope) * 10_000.0
        if slope_bps < 1.0:
            return 0.0, 0.2
        direction = 1.0 if slope > 0 else -1.0

        scaled = math.pow(slope_bps / 180.0, 0.82)
        confidence = 0.23 + (scaled * 0.62)
        if slope < 0:
            confidence *= 0.7

        confidence = _clamp(confidence, 0.25, 1.0)
        return direction, confidence

    def slope(self, short: str, long: str) -> float | None:
        short_key = self._normalize_tenor(short)
        long_key = self._normalize_tenor(long)
        if not short_key or not long_key:
            return None
        start = self.latest_values.get(short_key)
        end = self.latest_values.get(long_key)
        if start is None or end is None:
            return None
        return float(end - start)

    def slope_2s10s(self) -> float | None:
        return self.slope("2Y", "10Y")

    def slope_5s30s(self) -> float | None:
        return self.slope("5Y", "30Y")

    def curvature_2_10_30(self) -> float | None:
        y2 = self.latest_values.get("2Y")
        y10 = self.latest_values.get("10Y")
        y30 = self.latest_values.get("30Y")
        if y2 is None or y10 is None or y30 is None:
            return None
        return float(2.0 * y10 - y2 - y30)

    def parallel_shift(self) -> float | None:
        if not self.latest_values:
            return None
        vals = [v for v in self.latest_values.values() if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def snapshot(self) -> YieldSignalSnapshot:
        slope_2s10s = self.slope_2s10s()
        slope_5s30s = self.slope_5s30s()
        curvature = self.curvature_2_10_30()
        shift = self.parallel_shift()
        direction, confidence = self.signal()

        inversion_risk = 0.0
        regime = "insufficient_data"
        if slope_2s10s is not None:
            if slope_2s10s < -0.0005:
                inversion_risk = _clamp(abs(slope_2s10s) * 40.0, 0.0, 1.0)
                regime = "inverted"
            elif slope_2s10s < 0.002:
                regime = "flat"
            elif slope_2s10s < 0.01:
                regime = "modestly_steep"
            else:
                regime = "steep"

        if curvature is not None:
            confidence = max(confidence, _clamp(abs(curvature) * 2.0, 0.2, 1.0))

        tenors = {k: float(v) for k, v in sorted(self.latest_values.items())}

        return YieldSignalSnapshot(
            direction=float(direction),
            confidence=float(_clamp(confidence, 0.0, 1.0)),
            slope_2s10s=slope_2s10s,
            slope_5s30s=slope_5s30s,
            curvature_2_10_30=curvature,
            parallel_shift=shift,
            inversion_risk=float(inversion_risk),
            regime=regime,
            tenors=tenors,
        )
