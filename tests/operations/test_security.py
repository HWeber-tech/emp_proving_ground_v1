import logging
from datetime import UTC, datetime
from typing import Mapping

import pytest

from src.core.event_bus import Event
from src.operations.event_bus_failover import EventPublishError
from src.operations.security import (
    SecurityControlEvaluation,
    SecurityPolicy,
    SecurityPostureSnapshot,
    SecurityState,
    SecurityStatus,
    evaluate_security_posture,
    publish_security_posture,
)


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._running = True

    def is_running(self) -> bool:  # pragma: no cover - simple proxy
        return self._running

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


def test_evaluate_security_posture_pass() -> None:
    policy = SecurityPolicy(
        minimum_mfa_coverage=0.8,
        credential_rotation_days=90,
        secrets_rotation_days=30,
        incident_drill_interval_days=120,
        vulnerability_scan_interval_days=45,
        required_tls_versions=("TLS1.2", "TLS1.3"),
        require_intrusion_detection=True,
    )
    state = SecurityState(
        total_users=10,
        mfa_enabled_users=9,
        credential_age_days=10,
        secrets_age_days=5,
        incident_drill_age_days=30,
        vulnerability_scan_age_days=10,
        intrusion_detection_enabled=True,
        open_critical_alerts=tuple(),
        tls_versions=("TLS1.2", "TLS1.3"),
        secrets_manager_healthy=True,
    )

    now = datetime(2025, 1, 5, tzinfo=UTC)
    snapshot = evaluate_security_posture(policy, state, now=now, service="emp")

    assert snapshot.status is SecurityStatus.passed
    assert all(control.status is not SecurityStatus.fail for control in snapshot.controls)
    markdown = snapshot.to_markdown()
    assert "MFA coverage" in markdown
    payload = snapshot.as_dict()
    assert payload["service"] == "emp"
    assert payload["metadata"]["controls_evaluated"] == len(snapshot.controls)


def test_evaluate_security_posture_escalates() -> None:
    policy = SecurityPolicy(
        minimum_mfa_coverage=0.9,
        credential_rotation_days=30,
        secrets_rotation_days=15,
        incident_drill_interval_days=60,
        vulnerability_scan_interval_days=30,
        required_tls_versions=("TLS1.2", "TLS1.3"),
        require_intrusion_detection=True,
    )
    state = SecurityState(
        total_users=20,
        mfa_enabled_users=5,
        credential_age_days=120,
        secrets_age_days=40,
        incident_drill_age_days=120,
        vulnerability_scan_age_days=50,
        intrusion_detection_enabled=False,
        open_critical_alerts=("alert-1",),
        tls_versions=("TLS1.2", "TLS1.1"),
        legacy_tls_in_use=True,
        secrets_manager_healthy=False,
    )

    snapshot = evaluate_security_posture(policy, state, service="emp")

    assert snapshot.status is SecurityStatus.fail
    failing_controls = [
        control for control in snapshot.controls if control.status is SecurityStatus.fail
    ]
    assert failing_controls
    assert any("credential" in control.control for control in failing_controls)
    assert snapshot.metadata["open_critical_alerts"] == ["alert-1"]


def test_policy_and_state_from_mapping_parsing() -> None:
    mapping = {
        "SECURITY_MFA_MIN_COVERAGE": "0.85",
        "SECURITY_CREDENTIAL_ROTATION_DAYS": "45",
        "SECURITY_REQUIRED_TLS_VERSIONS": "TLS1.2,TLS1.3",
        "SECURITY_ALLOW_LEGACY_TLS": "false",
        "SECURITY_TOTAL_USERS": "50",
        "SECURITY_USERS_WITH_MFA": "40",
        "SECURITY_CREDENTIAL_AGE_DAYS": "20",
        "SECURITY_OPEN_ALERTS": "alert-a;alert-b",
        "SECURITY_TLS_VERSIONS": ["TLS1.2", "TLS1.3"],
    }

    policy = SecurityPolicy.from_mapping(mapping)
    state = SecurityState.from_mapping(mapping)

    assert policy.minimum_mfa_coverage == 0.85
    assert policy.credential_rotation_days == 45
    assert state.total_users == 50
    assert state.mfa_enabled_users == 40
    assert state.open_critical_alerts == ("alert-a", "alert-b")


def test_publish_security_posture_emits_event() -> None:
    bus = _StubEventBus()
    control = SecurityControlEvaluation(
        control="mfa",
        status=SecurityStatus.passed,
        summary="ok",
    )
    snapshot = SecurityPostureSnapshot(
        service="emp",
        generated_at=datetime(2025, 2, 1, tzinfo=UTC),
        status=SecurityStatus.passed,
        controls=(control,),
        metadata={},
    )

    publish_security_posture(bus, snapshot)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.operational.security"
    assert event.payload["status"] == snapshot.status.value


def test_publish_security_posture_falls_back_on_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _GlobalBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, Mapping[str, object], str]] = []

        def publish_sync(self, event_type: str, payload: Mapping[str, object], *, source: str) -> None:
            self.events.append((event_type, dict(payload), source))

    bus = _StubEventBus()

    def _failing_publish(_: Event) -> None:
        raise RuntimeError("primary bus offline")

    bus.publish_from_sync = _failing_publish  # type: ignore[method-assign]

    global_bus = _GlobalBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    control = SecurityControlEvaluation(
        control="mfa",
        status=SecurityStatus.passed,
        summary="ok",
    )
    snapshot = SecurityPostureSnapshot(
        service="emp",
        generated_at=datetime(2025, 2, 1, tzinfo=UTC),
        status=SecurityStatus.passed,
        controls=(control,),
        metadata={},
    )

    with caplog.at_level(logging.WARNING):
        publish_security_posture(bus, snapshot)

    assert not bus.events
    assert global_bus.events
    assert any("falling back to global bus" in message for message in caplog.messages)


def test_publish_security_posture_raises_on_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bus = _StubEventBus()

    def _unexpected(_: Event) -> None:
        raise ValueError("boom")

    bus.publish_from_sync = _unexpected  # type: ignore[method-assign]

    called: list[object] = []

    def _fail() -> None:
        called.append("called")
        raise AssertionError("global bus should not be reached")

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", _fail)

    control = SecurityControlEvaluation(
        control="mfa",
        status=SecurityStatus.passed,
        summary="ok",
    )
    snapshot = SecurityPostureSnapshot(
        service="emp",
        generated_at=datetime(2025, 2, 1, tzinfo=UTC),
        status=SecurityStatus.passed,
        controls=(control,),
        metadata={},
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EventPublishError) as exc_info:
            publish_security_posture(bus, snapshot)

    assert not called
    assert any("Unexpected error publishing security posture" in message for message in caplog.messages)
    assert exc_info.value.stage == "runtime"


def test_publish_security_posture_raises_on_global_bus_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bus = _StubEventBus()

    def _no_result(event: Event) -> None:
        bus.events.append(event)
        return None

    bus.publish_from_sync = _no_result  # type: ignore[method-assign]

    class _FailingGlobalBus:
        def publish_sync(self, event_type: str, payload: Mapping[str, object], *, source: str) -> None:
            raise RuntimeError("global bus offline")

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: _FailingGlobalBus())

    control = SecurityControlEvaluation(
        control="mfa",
        status=SecurityStatus.passed,
        summary="ok",
    )
    snapshot = SecurityPostureSnapshot(
        service="emp",
        generated_at=datetime(2025, 2, 1, tzinfo=UTC),
        status=SecurityStatus.passed,
        controls=(control,),
        metadata={},
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EventPublishError) as exc_info:
            publish_security_posture(bus, snapshot)

    assert bus.events  # primary attempt recorded but treated as failure
    assert any("Global event bus not running" in message for message in caplog.messages)
    assert exc_info.value.stage == "global"


def test_evaluate_security_posture_merges_metadata_without_mutation() -> None:
    policy = SecurityPolicy(minimum_mfa_coverage=0.0)
    state = SecurityState(total_users=0, mfa_enabled_users=0)
    metadata_input: Mapping[str, object] = {"region": "us-east-1"}

    snapshot = evaluate_security_posture(
        policy,
        state,
        metadata=metadata_input,
        service="emp",
        now=datetime(2025, 3, 1, tzinfo=UTC),
    )

    # Ensure the returned metadata contains the supplied keys plus the derived telemetry.
    assert snapshot.metadata["region"] == "us-east-1"
    assert snapshot.metadata["controls_evaluated"] == len(snapshot.controls)
    assert snapshot.metadata["mfa_coverage"] == 0.0

    # Verify the original mapping was not mutated so callers can reuse it safely.
    assert metadata_input == {"region": "us-east-1"}


def test_publish_security_posture_uses_global_bus_when_runtime_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NotRunningBus:
        def __init__(self) -> None:
            self.events: list[Event] = []

        def is_running(self) -> bool:  # pragma: no cover - simple proxy
            return False

        def publish_from_sync(self, _: Event) -> None:  # pragma: no cover - defensive
            raise AssertionError("runtime bus should not be used when not running")

    class _GlobalBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, Mapping[str, object], str]] = []

        def publish_sync(self, event_type: str, payload: Mapping[str, object], *, source: str) -> None:
            self.events.append((event_type, dict(payload), source))

    not_running_bus = _NotRunningBus()
    global_bus = _GlobalBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    control = SecurityControlEvaluation(
        control="mfa",
        status=SecurityStatus.passed,
        summary="ok",
    )
    snapshot = SecurityPostureSnapshot(
        service="emp",
        generated_at=datetime(2025, 3, 1, tzinfo=UTC),
        status=SecurityStatus.passed,
        controls=(control,),
        metadata={},
    )

    publish_security_posture(not_running_bus, snapshot)

    assert not_running_bus.events == []
    assert global_bus.events == [
        ("telemetry.operational.security", snapshot.as_dict(), "operations.security")
    ]


def test_evaluate_security_posture_records_failed_login_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[float] = []
    monkeypatch.setattr(
        "src.operations.security.operational_metrics.set_security_failed_logins",
        lambda value: recorded.append(float(value)),
    )

    policy = SecurityPolicy()
    state = SecurityState(
        total_users=5,
        mfa_enabled_users=5,
        credential_age_days=5,
        secrets_age_days=4,
        incident_drill_age_days=2,
        vulnerability_scan_age_days=3,
        intrusion_detection_enabled=True,
        failed_logins_last_hour=7,
    )

    evaluate_security_posture(policy, state, service="emp")

    assert recorded == [pytest.approx(7.0)]
