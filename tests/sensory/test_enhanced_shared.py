from __future__ import annotations

from src.sensory.enhanced._shared import clamp


def test_clamp_swaps_inverted_bounds() -> None:
    assert clamp(0.5, 1.0, 0.0) == 0.5
