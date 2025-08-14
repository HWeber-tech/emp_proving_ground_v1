from src.core.event_bus import get_global_bus


def test_core_event_bus_smoke():
    bus = get_global_bus()
    calls = []

    def handler(payload):
        calls.append(payload)

    handle = bus.subscribe("test.topic", handler)

    # First publish should invoke exactly one handler
    count = bus.publish("test.topic", {"ping": 1})
    assert count == 1
    assert calls and calls[-1] == {"ping": 1}

    # Unsubscribe and ensure no handler is invoked
    bus.unsubscribe(handle)
    calls.clear()
    count2 = bus.publish("test.topic", {"ping": 2})
    assert count2 == 0
    assert calls == []