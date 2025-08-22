"""
Yield signal utilities for WHY dimension.

Provides a small stateful tracker to compute common yield-curve features
such as 2s10s slope and a basic directional signal/confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class YieldSlopeTracker:
    """Tracks latest tenor values and computes slope features.

    Default tenors used for slope are '2Y' and '10Y'.
    """

    short_tenor: str = "2Y"
    long_tenor: str = "10Y"
    latest_values: Dict[str, float] = field(default_factory=dict)

    def update(self, tenor: str | None, value: float) -> None:
        if tenor is None:
            return
        try:
            self.latest_values[str(tenor).upper()] = float(value)
        except Exception:
            return

    def signal(self) -> Tuple[float, float]:
        """Return a naive directional signal and confidence.

        Signal is sign of steepening: +1 if long-short > 0, -1 if < 0, 0 if flat/unknown.
        Confidence grows with absolute slope up to a modest cap.
        """
        sl = self.slope(self.short_tenor, self.long_tenor)
        if sl is None:
            return 0.0, 0.0
        if abs(sl) < 1e-6:
            return 0.0, 0.2
        direction = 1.0 if sl > 0 else -1.0
        # Simple scaling: 0.1 slope → ~0.5 confidence, capped at 1.0
        conf = max(0.2, min(1.0, abs(sl) * 5.0))
        return direction, conf

    # Extended features
    def slope(self, short: str, long: str) -> float | None:
        s_t = short.upper()
        l_t = long.upper()
        s = self.latest_values.get(s_t)
        l = self.latest_values.get(l_t)
        if s is None or l is None:
            return None
        return float(l - s)

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
        # Standard 2-10-30 curvature: 2*10Y - 2Y - 30Y
        return float(2.0 * y10 - y2 - y30)

    def parallel_shift(self) -> float | None:
        if not self.latest_values:
            return None
        # Average level across available tenors as proxy for shift
        vals = [v for v in self.latest_values.values() if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))


