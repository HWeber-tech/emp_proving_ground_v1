from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.slow_context import resolve_slow_context_multiplier


def test_slow_context_blocks_near_macro_event() -> None:
    anchor = datetime.now(timezone.utc)
    decision = resolve_slow_context_multiplier(
        as_of=anchor,
        macro_events=[anchor + timedelta(seconds=30)],
        vix_value=12.5,
    )

    assert decision.multiplier == pytest.approx(0.0)
    assert decision.reason == "macro_event_proximity"
    assert decision.seconds_to_macro_event == pytest.approx(30.0)


def test_slow_context_applies_high_vix_multiplier() -> None:
    anchor = datetime(2024, 1, 1, tzinfo=timezone.utc)
    decision = resolve_slow_context_multiplier(
        as_of=anchor,
        macro_events=[],
        vix_value=41.2,
    )

    assert decision.multiplier == pytest.approx(0.3)
    assert decision.reason == "high_volatility"
    assert decision.vix == pytest.approx(41.2)


def test_slow_context_defaults_to_unity_multiplier() -> None:
    anchor = datetime(2024, 1, 1, tzinfo=timezone.utc)
    decision = resolve_slow_context_multiplier(
        as_of=anchor,
        macro_events=[{"timestamp": (anchor + timedelta(seconds=360)).isoformat()}],
        vix_value=18.4,
    )

    assert decision.multiplier == pytest.approx(1.0)
    assert decision.reason == "normal"
    assert decision.seconds_to_macro_event == pytest.approx(360.0)
