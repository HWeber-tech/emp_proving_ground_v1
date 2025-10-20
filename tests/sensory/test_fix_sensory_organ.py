from __future__ import annotations

import pytest

from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan


class _CaptureEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    async def emit(self, name: str, payload: dict[str, object]) -> None:
        self.events.append((name, payload))


class _NoopTaskFactory:
    def __call__(self, *_args: object, **_kwargs: object):  # pragma: no cover - helper
        raise AssertionError("task factory should not be invoked during parsing tests")


def _build_organ(event_bus: _CaptureEventBus) -> FIXSensoryOrgan:
    return FIXSensoryOrgan(
        event_bus=event_bus,
        price_queue=None,
        config={},
        task_factory=_NoopTaskFactory(),
    )


@pytest.mark.asyncio
async def test_snapshot_builds_depth_levels() -> None:
    bus = _CaptureEventBus()
    organ = _build_organ(bus)
    message = {
        55: b"EUR/USD",
        34: b"123",
        b"entries": [
            {"type": b"0", "px": 100.5, "size": 1.0},
            {"type": b"0", "px": 101.1, "size": 2.5},
            {"type": b"0", "px": 100.9, "size": 1.2},
            {"type": b"1", "px": 101.4, "size": 1.8},
            {"type": b"1", "px": 101.2, "size": 3.1},
        ],
    }

    await organ._handle_market_data_snapshot(message)

    assert bus.events, "expected market data update to be emitted"
    event, payload = bus.events[-1]
    assert event == "market_data_update"
    assert payload["symbol"] == "EUR/USD"
    assert payload["bid"] == pytest.approx(101.1)
    assert payload["bid_sz"] == pytest.approx(2.5)
    assert payload["ask"] == pytest.approx(101.2)
    assert payload["ask_sz"] == pytest.approx(3.1)
    assert payload["seq"] == 123

    depth = payload["depth"]
    assert depth["L2"]["bid"] == pytest.approx(100.9)
    assert depth["L1"]["ask"] == pytest.approx(101.2)
    assert depth["L3"]["bid"] == pytest.approx(100.5)
    assert depth["L5"]["ask"] is None


@pytest.mark.asyncio
async def test_incremental_refresh_updates_snapshot() -> None:
    bus = _CaptureEventBus()
    organ = _build_organ(bus)

    snapshot = {
        55: b"EUR/USD",
        34: b"200",
        b"entries": [
            {"type": b"0", "px": 100.0, "size": 1.0},
            {"type": b"1", "px": 100.2, "size": 1.5},
            {"type": b"1", "px": 100.3, "size": 2.5},
        ],
    }
    await organ._handle_market_data_snapshot(snapshot)
    bus.events.clear()

    incremental = {
        55: b"EUR/USD",
        34: b"201",
        b"entries": [
            {"type": b"0", "px": 100.0, "size": 4.0, "action": b"1"},
            {"type": b"1", "px": 100.2, "size": 0.0, "action": b"2"},
            {"type": b"1", "px": 100.25, "size": 1.3, "action": b"0"},
        ],
    }

    await organ._handle_market_data_incremental(incremental)

    assert bus.events, "expected incremental market data update"
    _, payload = bus.events[-1]
    assert payload["bid"] == pytest.approx(100.0)
    assert payload["bid_sz"] == pytest.approx(4.0)
    assert payload["ask"] == pytest.approx(100.25)
    assert payload["ask_sz"] == pytest.approx(1.3)
    assert payload["seq"] == 201

    depth = payload["depth"]
    assert depth["L1"]["ask"] == pytest.approx(100.25)
    assert depth["L2"]["ask"] == pytest.approx(100.3)


@pytest.mark.asyncio
async def test_single_entry_snapshot_fallback() -> None:
    bus = _CaptureEventBus()
    organ = _build_organ(bus)
    message = {
        55: b"GBP/USD",
        34: "7",
        269: b"1",
        270: "101.42",
        271: "3.25",
    }

    await organ._handle_market_data_snapshot(message)

    assert bus.events
    _, payload = bus.events[-1]
    assert payload["symbol"] == "GBP/USD"
    assert payload["ask"] == pytest.approx(101.42)
    assert payload["ask_sz"] == pytest.approx(3.25)
    assert payload["bid"] is None
    assert payload["seq"] == 7

    depth = payload["depth"]
    assert depth["L1"]["ask"] == pytest.approx(101.42)
    assert depth["L1"]["bid"] is None


@pytest.mark.asyncio
async def test_snapshot_parses_numeric_entry_groups() -> None:
    bus = _CaptureEventBus()
    organ = _build_organ(bus)
    message = {
        55: b"USD/JPY",
        34: b"45",
        268: [
            {269: b"0", 270: b"150.10", 271: b"3.0"},
            {269: b"0", 270: b"149.95", 271: b"2.5"},
            {269: b"1", 270: b"150.12", 271: b"1.7"},
            {269: b"1", 270: b"150.25", 271: b"4.1"},
        ],
    }

    await organ._handle_market_data_snapshot(message)

    assert bus.events
    _, payload = bus.events[-1]
    assert payload["symbol"] == "USD/JPY"
    assert payload["bid"] == pytest.approx(150.10)
    assert payload["bid_sz"] == pytest.approx(3.0)
    assert payload["ask"] == pytest.approx(150.12)
    assert payload["ask_sz"] == pytest.approx(1.7)
    assert payload["seq"] == 45

    depth = payload["depth"]
    assert depth["L2"]["bid"] == pytest.approx(149.95)
    assert depth["L2"]["ask"] == pytest.approx(150.25)
