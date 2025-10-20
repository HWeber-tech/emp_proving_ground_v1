from datetime import datetime, timezone

from artifacts.archive import _normalise_run_id


def test_normalise_run_id_collapses_repeated_hyphens() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    result = _normalise_run_id("foo   bar", timestamp)

    assert result == "foo-bar"
