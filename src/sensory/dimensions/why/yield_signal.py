from __future__ import annotations

"""
Yield signal utilities for WHY dimension.

Provides a small stateful tracker to compute common yield-curve features
such as 2s10s slope and a basic directional signal/confidence.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class YieldSlopeTracker:
    """Tracks latest tenor values and computes slope features.

    Default tenors used for slope are '2Y' and '10Y'.
    """

    short_tenor: str = "2Y"
    long_tenor: str = "10Y"
    latest_values: Dict[str, float] = field(default_factory=dict)

    def update(self, tenor: str, value: float) -> None:
        if tenor is None:
            return
        try:
            self.latest_values[str(tenor).upper()] = float(value)
        except Exception:
            return

    def slope(self) -> Optional[float]:
        s = self.latest_values.get(self.short_tenor.upper())
        l = self.latest_values.get(self.long_tenor.upper())
        if s is None or l is None:
            return None
        return float(l - s)

    def signal(self) -> Tuple[float, float]:
        """Return a naive directional signal and confidence.

        Signal is sign of steepening: +1 if long-short > 0, -1 if < 0, 0 if flat/unknown.
        Confidence grows with absolute slope up to a modest cap.
        """
        sl = self.slope()
        if sl is None:
            return 0.0, 0.0
        if abs(sl) < 1e-6:
            return 0.0, 0.2
        direction = 1.0 if sl > 0 else -1.0
        # Simple scaling: 0.1 slope → ~0.5 confidence, capped at 1.0
        conf = max(0.2, min(1.0, abs(sl) * 5.0))
        return direction, conf


