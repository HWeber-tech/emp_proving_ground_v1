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


def test_timestamper_utc_iso_uses_z_suffix():
    timestamper = TimeStamper(utc=True)
    event: dict[str, str] = {}

    timestamper(None, "", event)

    assert event["timestamp"].endswith("Z")
