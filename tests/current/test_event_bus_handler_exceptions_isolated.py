import asyncio
import logging

import pytest

from src.core.event_bus import AsyncEventBus, Event


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_handler_kind", ["sync", "async"])
async def test_event_bus_handler_exceptions_isolated(
    bad_handler_kind: str, caplog: pytest.LogCaptureFixture
):
    bus = AsyncEventBus()
    await bus.start()
    try:
        ok_seen = asyncio.Event()
        bad_seen = asyncio.Event()

        async def good_handler(ev: Event) -> None:
            ok_seen.set()

        if bad_handler_kind == "sync":

            def bad_handler_sync(ev: Event) -> None:
                bad_seen.set()
                raise RuntimeError("boom-sync")

            bad_handler = bad_handler_sync
        else:

            async def bad_handler_async(ev: Event) -> None:
                bad_seen.set()
                raise RuntimeError("boom-async")

            bad_handler = bad_handler_async

        bus.subscribe("boom", good_handler)
        bus.subscribe("boom", bad_handler)  # type: ignore[arg-type]

        with caplog.at_level(logging.ERROR):
            await bus.publish(Event(type="boom", payload={"k": "v"}))
            # Ensure both handlers were invoked
            await asyncio.wait_for(ok_seen.wait(), timeout=1.0)
            await asyncio.wait_for(bad_seen.wait(), timeout=1.0)

        # No exception should bubble to the caller; good handler must still have executed
        assert any(
            "Error in handler" in rec.message
            or "Unexpected error during event fan-out" in rec.message
            for rec in caplog.records
        )
    finally:
        await bus.stop()
