from datetime import datetime, timedelta, timezone

from structlog.processors import JSONRenderer, TimeStamper


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


def test_timestamper_accepts_uppercase_iso():
    timestamper = TimeStamper(fmt="ISO")
    event: dict[str, str] = {}

    timestamper(None, "", event)

    assert "timestamp" in event


def test_timestamper_respects_custom_now_factory():
    frozen = datetime(2023, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    calls: list[object] = []

    def fake_now() -> datetime:
        calls.append(object())
        return frozen

    timestamper = TimeStamper(utc=True, now_factory=fake_now)
    event: dict[str, str] = {}

    timestamper(None, "", event)
    timestamper(None, "", event)

    assert event["timestamp"] == "2023-01-02T03:04:05Z"
    assert len(calls) == 1


def test_timestamper_converts_non_utc_factory_values():
    offset = timezone(timedelta(hours=2))

    def fake_now() -> datetime:
        return datetime(2023, 1, 2, 3, 4, 5, tzinfo=offset)

    timestamper = TimeStamper(utc=True, now_factory=fake_now)
    event: dict[str, str] = {}

    timestamper(None, "", event)

    assert event["timestamp"] == "2023-01-02T01:04:05Z"


def test_jsonrenderer_respects_sort_keys_flag():
    renderer = JSONRenderer(sort_keys=False)
    rendered = renderer(None, "", {"b": 2, "a": 1})

    assert rendered == "{\"b\": 2, \"a\": 1}"
