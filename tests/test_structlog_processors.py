from structlog.processors import TimeStamper


def test_timestamper_respects_custom_key():
    timestamper = TimeStamper(key="time")
    event = {}

    timestamper(None, "", event)

    assert "time" in event
    first_value = event["time"]
    assert isinstance(first_value, str)

    timestamper(None, "", event)

    assert event["time"] == first_value
