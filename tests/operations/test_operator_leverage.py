from __future__ import annotations

from datetime import datetime, timedelta

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    from datetime import timezone

    UTC = timezone.utc

import pytest

from src.operations.operator_leverage import (
    OperatorLeverageSnapshot,
    OperatorLeverageStatus,
    evaluate_operator_leverage,
    format_operator_leverage_markdown,
)


pytestmark = pytest.mark.guardrail


def _timestamp(now: datetime, days: int) -> str:
    return (now - timedelta(days=days)).astimezone(UTC).isoformat()


def _event(
    *,
    operator: str,
    timestamp: str,
    status: str,
    quality: str | None = "pass",
    reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "operator": operator,
        "timestamp": timestamp,
        "status": status,
    }
    if quality is not None:
        payload["quality"] = {"status": quality}
    if reason is not None:
        payload["reason"] = reason
    return payload


def test_evaluate_operator_leverage_ok_status() -> None:
    now = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
    events = [
        _event(operator="alice", timestamp=_timestamp(now, 3), status="executed"),
        _event(operator="alice", timestamp=_timestamp(now, 5), status="executed"),
        _event(operator="alice", timestamp=_timestamp(now, 9), status="executed"),
        _event(operator="alice", timestamp=_timestamp(now, 12), status="executed"),
        _event(operator="bob", timestamp=_timestamp(now, 2), status="executed"),
        _event(operator="bob", timestamp=_timestamp(now, 4), status="executed"),
        _event(operator="bob", timestamp=_timestamp(now, 8), status="executed"),
        _event(operator="bob", timestamp=_timestamp(now, 11), status="executed"),
    ]

    snapshot = evaluate_operator_leverage(
        events,
        lookback_days=14,
        target_experiments_per_week=2.0,
        warn_experiments_per_week=1.0,
        target_quality_rate=0.75,
        warn_quality_rate=0.5,
        generated_at=now,
    )

    assert snapshot.status is OperatorLeverageStatus.ok
    assert snapshot.operator_count == 2
    assert snapshot.experiments_total == 8
    assert snapshot.quality_pass_rate == pytest.approx(1.0)
    assert snapshot.metadata["low_velocity_warn"] == ()
    assert snapshot.metadata["quality_missing"] == ()
    assert snapshot.experiments_per_week is not None
    assert snapshot.experiments_per_week > 1.9

    markdown = format_operator_leverage_markdown(snapshot)
    assert "Operator leverage (OK)" in markdown
    assert "alice" in markdown


def test_evaluate_operator_leverage_warn_when_velocity_below_target() -> None:
    now = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
    events = [
        _event(operator="carol", timestamp=_timestamp(now, 3), status="executed"),
        _event(operator="carol", timestamp=_timestamp(now, 10), status="executed"),
    ]

    snapshot = evaluate_operator_leverage(
        events,
        lookback_days=14,
        target_experiments_per_week=3.0,
        warn_experiments_per_week=1.0,
        generated_at=now,
    )

    assert snapshot.status is OperatorLeverageStatus.warn
    assert ("carol",) == snapshot.metadata["low_velocity_warn"]
    assert snapshot.metadata["low_velocity_fail"] == ()


def test_evaluate_operator_leverage_fail_when_quality_low() -> None:
    now = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
    events = [
        _event(
            operator="dave",
            timestamp=_timestamp(now, 2),
            status="executed",
            quality="fail",
            reason="guardrail",
        ),
        _event(
            operator="dave",
            timestamp=_timestamp(now, 6),
            status="executed",
            quality="pass",
        ),
    ]

    snapshot = evaluate_operator_leverage(
        events,
        lookback_days=14,
        target_experiments_per_week=1.0,
        warn_experiments_per_week=0.5,
        target_quality_rate=0.9,
        warn_quality_rate=0.7,
        generated_at=now,
    )

    assert snapshot.status is OperatorLeverageStatus.fail
    assert snapshot.metadata["quality_fail"] == ("dave",)
    failure_reasons = snapshot.metadata.get("top_failure_reasons", {})
    assert failure_reasons.get("guardrail") == 1


def test_evaluate_operator_leverage_fail_when_no_events() -> None:
    now = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)

    snapshot = evaluate_operator_leverage(
        [],
        generated_at=now,
    )

    assert snapshot.status is OperatorLeverageStatus.fail
    assert snapshot.experiments_total == 0
    assert snapshot.operator_count == 0
    assert snapshot.metadata["low_velocity_fail"] == ()
