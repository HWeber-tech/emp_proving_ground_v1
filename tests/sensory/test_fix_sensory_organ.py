from __future__ import annotations

from datetime import datetime

import pytest

from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan


class _StubEventBus:
    async def emit(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - helper
        return None


class _NoopTaskFactory:
    def __call__(self, *_args: object, **_kwargs: object):  # pragma: no cover - helper
        raise AssertionError("task factory should not be invoked during parsing tests")


def _build_organ() -> FIXSensoryOrgan:
    return FIXSensoryOrgan(
        event_bus=_StubEventBus(),
        price_queue=None,
        config={},
        task_factory=_NoopTaskFactory(),
    )


def test_extract_market_data_prefers_best_levels_from_entries() -> None:
    organ = _build_organ()
    message = {
        b"entries": [
            {"type": b"0", "px": 100.5, "size": 1.0},
            {"type": b"0", "px": 101.0, "size": 2.0},
            {"type": b"1", "px": 101.8, "size": 3.0},
            {"type": b"1", "px": 101.2, "size": 4.5},
        ]
    }

    result = organ._extract_market_data(message)

    assert result["bid"] == pytest.approx(101.0)
    assert result["bid_size"] == pytest.approx(2.0)
    assert result["ask"] == pytest.approx(101.2)
    assert result["ask_size"] == pytest.approx(4.5)
    assert isinstance(result["timestamp"], datetime)


def test_extract_market_data_single_entry_fallback() -> None:
    organ = _build_organ()
    message = {
        269: b"1",  # Ask entry
        270: "101.42",
        271: "3.25",
    }

    result = organ._extract_market_data(message)

    assert result["ask"] == pytest.approx(101.42)
    assert result["ask_size"] == pytest.approx(3.25)
    assert "bid" not in result
    assert isinstance(result["timestamp"], datetime)
