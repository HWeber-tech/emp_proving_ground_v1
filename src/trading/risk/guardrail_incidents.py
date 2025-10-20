"""Utilities for extracting guardrail incidents from risk decisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _normalise_status(value: object) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"violation", "warn", "warning", "ok", "info", "error"}:
            return "warning" if lowered == "warning" else lowered
    return str(value).strip().lower()


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_timestamp(candidate: object) -> datetime:
    if isinstance(candidate, datetime):
        ts = candidate
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(candidate, str):
        text = candidate.strip()
        if text:
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                parsed = datetime.fromisoformat(iso)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _is_guardrail_check(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("risk.") or lowered.startswith("policy."):
        return True
    if lowered.startswith("portfolio_risk"):
        return True
    if lowered in {"daily_drawdown", "open_positions", "confidence_floor"}:
        return True
    return "risk" in lowered


@dataclass(frozen=True)
class GuardrailCheck:
    """Serializable snapshot of a guardrail check."""

    name: str
    status: str
    value: float | None
    threshold: float | None
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class GuardrailIncident:
    """Captured guardrail incident derived from a risk decision."""

    incident_id: str
    severity: str
    symbol: str
    strategy_id: str | None
    reason: str | None
    decision_status: str
    approved: bool
    timestamp: datetime
    checks: tuple[GuardrailCheck, ...]
    overrides: bool
    metadata: Mapping[str, Any]

    def primary_check(self) -> GuardrailCheck | None:
        return self.checks[0] if self.checks else None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "reason": self.reason,
            "decision_status": self.decision_status,
            "approved": self.approved,
            "timestamp": self.timestamp.isoformat(),
            "overrides": self.overrides,
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }
        primary = self.primary_check()
        if primary is not None:
            payload["primary_check"] = primary.name
            payload.setdefault(
                "description",
                f"{primary.name} {self.severity.replace('_', ' ')}",
            )
        return payload


def extract_guardrail_incident(
    decision: Mapping[str, Any]
) -> GuardrailIncident | None:
    """Analyse ``decision`` and emit a guardrail incident when checks warrant it."""

    checks_raw = decision.get("checks")
    guardrail_checks: list[GuardrailCheck] = []
    if isinstance(checks_raw, Sequence):
        for entry in checks_raw:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or "").strip()
            if not name or not _is_guardrail_check(name):
                continue
            value = _coerce_float(entry.get("value"))
            threshold = _coerce_float(entry.get("threshold"))
            status = _normalise_status(entry.get("status"))
            if status not in {"violation", "warn"}:
                if threshold is not None and value is not None:
                    lowered_name = name.lower()
                    is_min_guardrail = (
                        ".min_" in lowered_name
                        or lowered_name.startswith("min_")
                        or lowered_name.endswith("_min")
                    )
                    if is_min_guardrail:
                        if value < threshold:
                            if threshold > 0 and value / threshold >= 0.8:
                                status = "warn"
                            else:
                                status = "violation"
                    else:
                        if value > threshold:
                            status = "violation"
                        elif threshold > 0 and value / threshold >= 0.8:
                            status = "warn"
            metadata = _coerce_mapping(entry.get("metadata"))
            guardrail_checks.append(
                GuardrailCheck(name=name, status=status, value=value, threshold=threshold, metadata=dict(metadata)),
            )

    if not guardrail_checks:
        return None

    violation_checks = [check for check in guardrail_checks if check.status == "violation"]
    warn_checks = [check for check in guardrail_checks if check.status == "warn"]

    if not violation_checks and not warn_checks:
        return None

    decision_status = str(decision.get("status") or "").strip().lower()
    approved = decision_status == "approved"
    overrides = bool(violation_checks and approved)

    severity = "violation"
    if not violation_checks or overrides:
        severity = "near_miss"

    relevant_checks: tuple[GuardrailCheck, ...]
    if severity == "violation":
        relevant_checks = tuple(violation_checks)
    else:
        preferred = violation_checks + warn_checks if overrides else warn_checks
        relevant_checks = tuple(preferred or guardrail_checks)

    timestamp = _parse_timestamp(decision.get("timestamp"))
    symbol = str(decision.get("symbol") or "UNKNOWN")
    strategy_id = decision.get("strategy_id")
    reason = decision.get("reason")

    metadata: dict[str, Any] = {
        "violations": [check.name for check in violation_checks],
        "warnings": [check.name for check in warn_checks],
        "overrides": overrides,
    }

    id_material = "|".join(
        (
            timestamp.isoformat(),
            symbol,
            severity,
            decision_status or "unknown",
            ",".join(metadata["violations"] or metadata["warnings"] or ("guardrail",)),
        )
    )
    incident_id = hashlib.sha1(id_material.encode("utf-8", "ignore")).hexdigest()

    return GuardrailIncident(
        incident_id=incident_id,
        severity=severity,
        symbol=symbol,
        strategy_id=str(strategy_id) if strategy_id is not None else None,
        reason=str(reason) if reason is not None else None,
        decision_status=decision_status or "unknown",
        approved=approved,
        timestamp=timestamp,
        checks=relevant_checks,
        overrides=overrides,
        metadata=metadata,
    )


__all__ = [
    "GuardrailCheck",
    "GuardrailIncident",
    "extract_guardrail_incident",
]
