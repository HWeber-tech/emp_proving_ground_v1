from datetime import date, datetime, timezone

from src.runtime.paper_simulation import _parse_order_timestamp


def test_parse_order_timestamp_accepts_date() -> None:
    parsed = _parse_order_timestamp(date(2024, 5, 1))

    assert isinstance(parsed, datetime)
    assert parsed.tzinfo is timezone.utc
    assert parsed == datetime(2024, 5, 1, tzinfo=timezone.utc)
