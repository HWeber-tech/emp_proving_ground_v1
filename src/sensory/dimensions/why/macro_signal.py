from __future__ import annotations

"""
Lightweight macro proximity signal for WHY dimension in offline backtests.

Defines a simple proximity-based signal and confidence given macro event timestamps.
"""

from typing import List, Tuple


def macro_proximity_signal(minutes_since_last: float | None,
                           minutes_to_next: float | None,
                           window: float = 15.0) -> Tuple[float, float]:
    """Compute a WHY signal and confidence based on closeness to macro events.

    - If within +/- window minutes of a macro event, reduce confidence and set neutral signal.
    - Otherwise neutral signal with higher confidence.
    Returns (signal_strength in [-1,1], confidence in [0,1]).
    """
    if minutes_since_last is None and minutes_to_next is None:
        return 0.0, 0.6
    near_last = (minutes_since_last is not None and abs(minutes_since_last) <= window)
    near_next = (minutes_to_next is not None and abs(minutes_to_next) <= window)
    if near_last or near_next:
        return 0.0, 0.3
    return 0.0, 0.8


