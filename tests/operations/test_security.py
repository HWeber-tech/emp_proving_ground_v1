from datetime import UTC, datetime

from src.core.event_bus import Event
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
