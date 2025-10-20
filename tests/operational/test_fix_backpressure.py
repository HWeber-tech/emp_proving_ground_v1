import asyncio

import pytest

from src.operational.fix_connection_manager import _FIXApplicationAdapter


@pytest.mark.asyncio
async def test_fix_application_adapter_backpressure_handler_triggers_on_drop() -> None:
    adapter = _FIXApplicationAdapter(session_type="quote")
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    adapter.set_message_queue(queue)

    events: list[tuple[str, bool, dict[str, object]]] = []

    def handler(session: str, active: bool, payload: dict[str, object]) -> None:
        events.append((session, active, payload))

    adapter.set_backpressure_handler(handler)

    adapter.dispatch({35: b"W"})
    assert not events

    adapter.dispatch({35: b"W"})
    assert events
    session, active, payload = events[0]
    assert session == "quote"
    assert active is True
    assert payload["capacity"] == 1
    assert payload["queue_size"] == 1
    assert payload["dropped_total"] == 1

    await queue.get()
    adapter.dispatch({35: b"W"})
    assert len(events) >= 2
    _, recovered_active, recovered_payload = events[-1]
    assert recovered_active is False
    assert recovered_payload["queue_size"] <= 1
