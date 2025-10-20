"""Security posture evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.event_bus_failover import publish_event_with_failover
import src.operational.metrics as operational_metrics


logger = logging.getLogger(__name__)


class SecurityStatus(StrEnum):
    """Severity levels exposed by the security posture snapshot."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[SecurityStatus, int] = {
    SecurityStatus.passed: 0,
    SecurityStatus.warn: 1,
    SecurityStatus.fail: 2,
}


def _combine_status(current: SecurityStatus, candidate: SecurityStatus) -> SecurityStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _coerce_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Sequence):
        items: list[str] = []
        for entry in value:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)
        return tuple(items)
    return tuple()


@dataclass(frozen=True)
class SecurityPolicy:
    """Declarative policy describing the expected security baseline."""

    minimum_mfa_coverage: float = 0.95
    credential_rotation_days: int = 90
    secrets_rotation_days: int = 30
    incident_drill_interval_days: int = 90
    vulnerability_scan_interval_days: int = 30
    required_tls_versions: tuple[str, ...] = ("TLS1.2", "TLS1.3")
    allow_legacy_tls: bool = False
    require_intrusion_detection: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "SecurityPolicy":
        mapping = mapping or {}
        minimum_mfa_coverage = _coerce_float(mapping.get("SECURITY_MFA_MIN_COVERAGE"), 0.95)
        credential_rotation_days = _coerce_int(mapping.get("SECURITY_CREDENTIAL_ROTATION_DAYS"), 90)
        secrets_rotation_days = _coerce_int(mapping.get("SECURITY_SECRETS_ROTATION_DAYS"), 30)
        incident_interval = _coerce_int(mapping.get("SECURITY_INCIDENT_DRILL_DAYS"), 90)
        vuln_interval = _coerce_int(mapping.get("SECURITY_VULNERABILITY_SCAN_DAYS"), 30)
        required_tls_versions = _coerce_tuple(mapping.get("SECURITY_REQUIRED_TLS_VERSIONS"))
        allow_legacy_tls = _coerce_bool(mapping.get("SECURITY_ALLOW_LEGACY_TLS"), False)
        require_id = _coerce_bool(mapping.get("SECURITY_REQUIRE_INTRUSION_DETECTION"), True)

        coverage = float(min(max(minimum_mfa_coverage or 0.0, 0.0), 1.0))
        tls_versions = required_tls_versions or ("TLS1.2", "TLS1.3")

        return cls(
            minimum_mfa_coverage=coverage,
            credential_rotation_days=max(credential_rotation_days, 0),
            secrets_rotation_days=max(secrets_rotation_days, 0),
            incident_drill_interval_days=max(incident_interval, 0),
            vulnerability_scan_interval_days=max(vuln_interval, 0),
            required_tls_versions=tuple(tls_versions),
            allow_legacy_tls=allow_legacy_tls,
            require_intrusion_detection=require_id,
        )


@dataclass(frozen=True)
class SecurityState:
    """Observed security posture inputs."""

    total_users: int = 0
    mfa_enabled_users: int = 0
    credential_age_days: float | None = None
    secrets_age_days: float | None = None
    incident_drill_age_days: float | None = None
    vulnerability_scan_age_days: float | None = None
    intrusion_detection_enabled: bool = False
    failed_logins_last_hour: int | None = None
    open_critical_alerts: tuple[str, ...] = field(default_factory=tuple)
    tls_versions: tuple[str, ...] = field(default_factory=tuple)
    legacy_tls_in_use: bool = False
    secrets_manager_healthy: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "SecurityState":
        mapping = mapping or {}
        total_users = _coerce_int(mapping.get("SECURITY_TOTAL_USERS"), 0)
        mfa_enabled = _coerce_int(mapping.get("SECURITY_USERS_WITH_MFA"), 0)
        credential_age = _coerce_float(mapping.get("SECURITY_CREDENTIAL_AGE_DAYS"))
        secrets_age = _coerce_float(mapping.get("SECURITY_SECRETS_AGE_DAYS"))
        incident_age = _coerce_float(mapping.get("SECURITY_INCIDENT_DRILL_AGE_DAYS"))
        vuln_age = _coerce_float(mapping.get("SECURITY_VULNERABILITY_SCAN_AGE_DAYS"))
        intrusion_detection = _coerce_bool(
            mapping.get("SECURITY_INTRUSION_DETECTION_ACTIVE"), False
        )
        alerts = _coerce_tuple(mapping.get("SECURITY_OPEN_ALERTS"))
        tls_versions = _coerce_tuple(mapping.get("SECURITY_TLS_VERSIONS"))
        legacy_tls = _coerce_bool(mapping.get("SECURITY_LEGACY_TLS_IN_USE"), False)
        secrets_healthy = _coerce_bool(mapping.get("SECURITY_SECRETS_MANAGER_HEALTHY"), True)
        failed_logins_raw = mapping.get("SECURITY_FAILED_LOGINS_LAST_HOUR")
        failed_logins = (
            max(_coerce_int(failed_logins_raw, 0), 0)
            if failed_logins_raw is not None
            else None
        )

        return cls(
            total_users=max(total_users, 0),
            mfa_enabled_users=max(mfa_enabled, 0),
            credential_age_days=credential_age,
            secrets_age_days=secrets_age,
            incident_drill_age_days=incident_age,
            vulnerability_scan_age_days=vuln_age,
            intrusion_detection_enabled=intrusion_detection,
            failed_logins_last_hour=failed_logins,
            open_critical_alerts=alerts,
            tls_versions=tls_versions,
            legacy_tls_in_use=legacy_tls,
            secrets_manager_healthy=secrets_healthy,
        )

    @property
    def mfa_coverage(self) -> float:
        if self.total_users <= 0:
            return 0.0
        return max(min(self.mfa_enabled_users / self.total_users, 1.0), 0.0)


@dataclass(frozen=True)
class SecurityControlEvaluation:
    """Evaluation result for an individual security control."""

    control: str
    status: SecurityStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "control": self.control,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SecurityPostureSnapshot:
    """Aggregated security posture snapshot."""

    service: str
    generated_at: datetime
    status: SecurityStatus
    controls: tuple[SecurityControlEvaluation, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "controls": [control.as_dict() for control in self.controls],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.controls:
            return "| Control | Status | Summary |\n| --- | --- | --- |\n"
        rows = ["| Control | Status | Summary |", "| --- | --- | --- |"]
        for control in self.controls:
            rows.append(
                f"| {control.control} | {control.status.value.upper()} | {control.summary} |"
            )
        return "\n".join(rows)


def _control(
    controls: list[SecurityControlEvaluation],
    status: SecurityStatus,
    control: str,
    summary: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    controls.append(
        SecurityControlEvaluation(
            control=control,
            status=status,
            summary=summary,
            metadata=dict(metadata or {}),
        )
    )


def _evaluate_mfa(policy: SecurityPolicy, state: SecurityState) -> SecurityControlEvaluation:
    coverage = state.mfa_coverage
    metadata = {
        "coverage": coverage,
        "minimum": policy.minimum_mfa_coverage,
        "total_users": state.total_users,
        "mfa_enabled": state.mfa_enabled_users,
    }
    if state.total_users <= 0:
        summary = "no eligible users recorded"
        status = SecurityStatus.warn if policy.minimum_mfa_coverage > 0 else SecurityStatus.passed
    elif coverage >= policy.minimum_mfa_coverage:
        summary = f"MFA coverage {coverage:.0%} meets target"
        status = SecurityStatus.passed
    elif coverage >= max(policy.minimum_mfa_coverage - 0.15, 0.0):
        summary = f"MFA coverage {coverage:.0%} below {policy.minimum_mfa_coverage:.0%} target"
        status = SecurityStatus.warn
    else:
        summary = f"MFA coverage {coverage:.0%} critically below {policy.minimum_mfa_coverage:.0%}"
        status = SecurityStatus.fail
    return SecurityControlEvaluation(
        control="mfa_coverage",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _evaluate_rotation(
    name: str,
    age_days: float | None,
    threshold_days: int,
    warn_ratio: float = 0.75,
) -> SecurityControlEvaluation:
    metadata = {
        "age_days": age_days,
        "threshold_days": threshold_days,
    }
    if threshold_days <= 0:
        return SecurityControlEvaluation(
            control=name,
            status=SecurityStatus.passed,
            summary="rotation disabled by policy",
            metadata=metadata,
        )
    if age_days is None:
        return SecurityControlEvaluation(
            control=name,
            status=SecurityStatus.warn,
            summary="rotation age unknown",
            metadata=metadata,
        )
    warn_threshold = threshold_days * warn_ratio
    if age_days <= warn_threshold:
        return SecurityControlEvaluation(
            control=name,
            status=SecurityStatus.passed,
            summary=f"rotation age {age_days:.0f}d within target",
            metadata=metadata,
        )
    if age_days <= threshold_days:
        return SecurityControlEvaluation(
            control=name,
            status=SecurityStatus.warn,
            summary=f"rotation age {age_days:.0f}d approaching {threshold_days}d limit",
            metadata=metadata,
        )
    evaluation = SecurityControlEvaluation(
        control=name,
        status=SecurityStatus.fail,
        summary=f"rotation age {age_days:.0f}d exceeds {threshold_days}d limit",
        metadata=metadata,
    )
    if name == "secrets_rotation":
        logger.warning(
            "Secrets rotation age %.0fd exceeds %dd policy limit",
            age_days,
            threshold_days,
            extra={
                "security_control": name,
                "secrets_age_days": age_days,
                "secrets_threshold_days": threshold_days,
            },
        )
    return evaluation


def _evaluate_incident_response(
    age_days: float | None,
    threshold_days: int,
) -> SecurityControlEvaluation:
    metadata = {
        "age_days": age_days,
        "threshold_days": threshold_days,
    }
    if threshold_days <= 0:
        return SecurityControlEvaluation(
            control="incident_response",
            status=SecurityStatus.passed,
            summary="incident drills disabled",
            metadata=metadata,
        )
    if age_days is None:
        return SecurityControlEvaluation(
            control="incident_response",
            status=SecurityStatus.warn,
            summary="incident drill history unavailable",
            metadata=metadata,
        )
    if age_days <= threshold_days * 0.75:
        return SecurityControlEvaluation(
            control="incident_response",
            status=SecurityStatus.passed,
            summary=f"last drill {age_days:.0f}d ago",
            metadata=metadata,
        )
    if age_days <= threshold_days:
        return SecurityControlEvaluation(
            control="incident_response",
            status=SecurityStatus.warn,
            summary=f"last drill {age_days:.0f}d ago approaching target",
            metadata=metadata,
        )
    return SecurityControlEvaluation(
        control="incident_response",
        status=SecurityStatus.fail,
        summary=f"last drill {age_days:.0f}d ago exceeds target",
        metadata=metadata,
    )


def _evaluate_intrusion_detection(
    policy: SecurityPolicy, state: SecurityState
) -> SecurityControlEvaluation:
    metadata = {
        "required": policy.require_intrusion_detection,
        "enabled": state.intrusion_detection_enabled,
        "open_alerts": list(state.open_critical_alerts),
    }
    if not policy.require_intrusion_detection:
        if state.open_critical_alerts:
            return SecurityControlEvaluation(
                control="intrusion_detection",
                status=SecurityStatus.warn,
                summary="intrusion detection optional but critical alerts open",
                metadata=metadata,
            )
        return SecurityControlEvaluation(
            control="intrusion_detection",
            status=SecurityStatus.passed,
            summary="intrusion detection optional",
            metadata=metadata,
        )
    if not state.intrusion_detection_enabled:
        return SecurityControlEvaluation(
            control="intrusion_detection",
            status=SecurityStatus.fail,
            summary="intrusion detection disabled",
            metadata=metadata,
        )
    if state.open_critical_alerts:
        return SecurityControlEvaluation(
            control="intrusion_detection",
            status=SecurityStatus.warn,
            summary="critical alerts require triage",
            metadata=metadata,
        )
    return SecurityControlEvaluation(
        control="intrusion_detection",
        status=SecurityStatus.passed,
        summary="intrusion detection active",
        metadata=metadata,
    )


def _evaluate_tls(policy: SecurityPolicy, state: SecurityState) -> SecurityControlEvaluation:
    observed = {version.upper() for version in state.tls_versions}
    required = {version.upper() for version in policy.required_tls_versions}
    missing = sorted(required - observed)
    legacy_in_use = state.legacy_tls_in_use or any(
        version in {"TLS1.0", "TLS1.1"} for version in observed
    )
    metadata = {
        "observed_versions": sorted(observed),
        "required_versions": sorted(required),
        "missing_versions": missing,
        "legacy_tls": legacy_in_use,
        "allow_legacy": policy.allow_legacy_tls,
    }
    if missing:
        return SecurityControlEvaluation(
            control="tls_configuration",
            status=SecurityStatus.fail,
            summary=f"missing TLS versions: {', '.join(missing)}",
            metadata=metadata,
        )
    if legacy_in_use and not policy.allow_legacy_tls:
        return SecurityControlEvaluation(
            control="tls_configuration",
            status=SecurityStatus.warn,
            summary="legacy TLS detected",
            metadata=metadata,
        )
    return SecurityControlEvaluation(
        control="tls_configuration",
        status=SecurityStatus.passed,
        summary="TLS configuration meets policy",
        metadata=metadata,
    )


def _record_security_metrics(state: SecurityState) -> None:
    failed_logins = state.failed_logins_last_hour
    if failed_logins is not None:
        operational_metrics.set_security_failed_logins(failed_logins)


def evaluate_security_posture(
    policy: SecurityPolicy,
    state: SecurityState,
    *,
    service: str = "emp_platform",
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SecurityPostureSnapshot:
    """Evaluate the current security posture against the supplied policy."""

    moment = now or datetime.now(tz=UTC)
    controls: list[SecurityControlEvaluation] = []

    mfa_control = _evaluate_mfa(policy, state)
    controls.append(mfa_control)

    controls.append(
        _evaluate_rotation(
            "credential_rotation",
            state.credential_age_days,
            policy.credential_rotation_days,
        )
    )
    controls.append(
        _evaluate_rotation(
            "secrets_rotation",
            state.secrets_age_days,
            policy.secrets_rotation_days,
        )
    )
    controls.append(
        _evaluate_incident_response(
            state.incident_drill_age_days,
            policy.incident_drill_interval_days,
        )
    )
    controls.append(
        _evaluate_rotation(
            "vulnerability_scan",
            state.vulnerability_scan_age_days,
            policy.vulnerability_scan_interval_days,
        )
    )
    controls.append(_evaluate_intrusion_detection(policy, state))
    controls.append(_evaluate_tls(policy, state))

    if not state.secrets_manager_healthy:
        controls.append(
            SecurityControlEvaluation(
                control="secrets_manager",
                status=SecurityStatus.warn,
                summary="secrets manager degraded",
                metadata={"healthy": False},
            )
        )

    status = SecurityStatus.passed
    for control in controls:
        status = _combine_status(status, control.status)

    combined_metadata: MutableMapping[str, object] = (
        dict(metadata) if isinstance(metadata, MutableMapping) else {}
    )
    if not combined_metadata:
        combined_metadata = {}
    combined_metadata.update(
        {
            "open_critical_alerts": list(state.open_critical_alerts),
            "tls_versions": list(state.tls_versions),
            "controls_evaluated": len(controls),
            "mfa_coverage": state.mfa_coverage,
        }
    )
    if state.failed_logins_last_hour is not None:
        combined_metadata["failed_logins_last_hour"] = state.failed_logins_last_hour

    _record_security_metrics(state)

    return SecurityPostureSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        controls=tuple(controls),
        metadata=combined_metadata,
    )


def publish_security_posture(event_bus: EventBus, snapshot: SecurityPostureSnapshot) -> None:
    """Publish the security posture snapshot onto the runtime event bus."""

    event = Event(
        type="telemetry.operational.security",
        payload=snapshot.as_dict(),
        source="operations.security",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
        "Primary event bus publish_from_sync failed; falling back to global bus",
        runtime_unexpected_message=
        "Unexpected error publishing security posture via runtime event bus",
        runtime_none_message=
        "Primary event bus publish_from_sync returned None; falling back to global bus",
        global_not_running_message=
        "Global event bus not running while publishing security posture snapshot",
        global_unexpected_message=
        "Unexpected error publishing security posture snapshot via global bus",
    )


__all__ = [
    "SecurityStatus",
    "SecurityPolicy",
    "SecurityState",
    "SecurityControlEvaluation",
    "SecurityPostureSnapshot",
    "evaluate_security_posture",
    "publish_security_posture",
]
