from src.core.anomaly import AnomalyEvent


def test_anomaly_event_metadata_isolated_default():
    event_a = AnomalyEvent(timestamp=1, kind="a", score=0.1)
    event_b = AnomalyEvent(timestamp=2, kind="b", score=0.2)

    assert event_a.metadata == {}
    assert event_b.metadata == {}

    event_a.metadata["foo"] = "bar"

    assert "foo" not in event_b.metadata
