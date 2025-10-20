from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts import check_secret_age


def _ts(days_ago: int, *, now: datetime) -> str:
    return (now - timedelta(days=days_ago)).isoformat()


def test_evaluate_secret_age_flags_stale_api_key() -> None:
    now = datetime(2024, 8, 1, tzinfo=timezone.utc)
    env = {
        "ALPHA_VANTAGE_API_KEY": "demo-value",
        "ALPHA_VANTAGE_API_KEY_ROTATED_AT": _ts(120, now=now),
    }

    records = check_secret_age.evaluate_secret_age(env, now=now)

    assert len(records) == 1
    record = records[0]
    assert record.name == "ALPHA_VANTAGE_API_KEY"
    assert record.kind is check_secret_age.SecretKind.api_key
    assert record.rotation_source == "ALPHA_VANTAGE_API_KEY_ROTATED_AT"
    assert record.status is check_secret_age.SecretStatus.stale
    assert record.age_days == pytest.approx(120.0)


def test_evaluate_secret_age_groups_rotation_prefix() -> None:
    now = datetime(2024, 5, 10, tzinfo=timezone.utc)
    env = {
        "LIVE_BROKER_SANDBOX_PRICE_PASSWORD": "secret-1",
        "LIVE_BROKER_SANDBOX_TRADE_PASSWORD": "secret-2",
        "LIVE_BROKER_SANDBOX_ROTATED_AT": _ts(5, now=now),
    }

    records = check_secret_age.evaluate_secret_age(env, now=now, secret_threshold_days=30)

    assert len(records) == 1
    record = records[0]
    assert record.name == "LIVE_BROKER_SANDBOX"
    assert record.kind is check_secret_age.SecretKind.secret
    assert record.status is check_secret_age.SecretStatus.ok
    assert set(record.variables) == {
        "LIVE_BROKER_SANDBOX_PRICE_PASSWORD",
        "LIVE_BROKER_SANDBOX_TRADE_PASSWORD",
    }


def test_evaluate_secret_age_handles_missing_metadata() -> None:
    now = datetime(2024, 3, 15, tzinfo=timezone.utc)
    env = {
        "NEWS_API_KEY": "token-1",
    }

    records = check_secret_age.evaluate_secret_age(env, now=now)

    assert len(records) == 1
    record = records[0]
    assert record.status is check_secret_age.SecretStatus.unknown
    assert record.rotation_source is None
    assert record.age_days is None


def test_evaluate_secret_age_marks_missing_value() -> None:
    now = datetime(2024, 6, 1, tzinfo=timezone.utc)
    env = {
        "PAPER_TRADING_API_SECRET": "   ",
        "PAPER_TRADING_API_SECRET_ROTATED_AT": _ts(10, now=now),
    }

    records = check_secret_age.evaluate_secret_age(env, now=now)

    assert len(records) == 1
    record = records[0]
    assert record.status is check_secret_age.SecretStatus.missing
    assert record.has_value is False


def test_parse_timestamp_accepts_unix_epoch() -> None:
    timestamp = check_secret_age.parse_timestamp("1700000000")
    assert timestamp is not None
    assert timestamp.tzinfo is timezone.utc

