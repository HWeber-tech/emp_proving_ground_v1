from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, cast

from src.core.event_bus import Event, EventBus
from src.core.coercion import coerce_float, coerce_int

__all__ = [
    "RiskLimitStatus",
    "RiskThresholdType",
    "RiskLimitCheck",
    "RiskTelemetrySnapshot",
    "evaluate_risk_posture",
    "format_risk_markdown",
    "publish_risk_snapshot",
]


class RiskLimitStatus(StrEnum):
    """Severity grading for a single risk check or aggregate snapshot."""

    ok = "ok"
    warn = "warn"
    alert = "alert"


class RiskThresholdType(StrEnum):
    """Describe whether a limit represents a maximum or minimum bound."""

    maximum = "maximum"
    minimum = "minimum"


@dataclass(frozen=True)
class RiskLimitCheck:
    """Single evaluated risk guardrail."""

    name: str
    value: float | None
    threshold: float | None
    threshold_type: RiskThresholdType
    status: RiskLimitStatus
    ratio: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "threshold_type": self.threshold_type.value,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.ratio is not None:
            payload["ratio"] = self.ratio
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RiskTelemetrySnapshot:
    """Aggregated risk posture emitted on ``telemetry.risk.posture``."""

    status: RiskLimitStatus
    generated_at: datetime
    checks: tuple[RiskLimitCheck, ...]
    exposures: Mapping[str, float]
    limits: Mapping[str, float]
    telemetry: Mapping[str, float]
    approval_rate: float | None
    portfolio_equity: float | None
    peak_equity: float | None
    last_decision: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "exposures": dict(self.exposures),
            "limits": dict(self.limits),
            "telemetry": dict(self.telemetry),
        }
        if self.approval_rate is not None:
            payload["approval_rate"] = self.approval_rate
        if self.portfolio_equity is not None:
            payload["portfolio_equity"] = self.portfolio_equity
        if self.peak_equity is not None:
            payload["peak_equity"] = self.peak_equity
        if self.last_decision is not None:
            payload["last_decision"] = dict(self.last_decision)
        return payload


_WARN_RATIO = 0.8


def _infer_threshold_type(name: str) -> RiskThresholdType:
    lowered = name.lower()
    if any(token in lowered for token in ("confidence", "floor", "min")):
        return RiskThresholdType.minimum
    return RiskThresholdType.maximum


def _grade_limit(
    value: float | None,
    threshold: float | None,
    *,
    threshold_type: RiskThresholdType,
) -> tuple[RiskLimitStatus, float | None]:
    if threshold is None:
        return RiskLimitStatus.ok, None

    limit = coerce_float(threshold)
    if limit is None:
        return RiskLimitStatus.ok, None

    if threshold_type is RiskThresholdType.maximum:
        current = coerce_float(value) or 0.0
        if limit <= 0:
            return RiskLimitStatus.ok, None
        ratio = current / limit
        if current >= limit:
            return RiskLimitStatus.alert, ratio
        if ratio >= _WARN_RATIO:
            return RiskLimitStatus.warn, ratio
        return RiskLimitStatus.ok, ratio

    # Minimum threshold – value should stay above the bound
    current_value = coerce_float(value)
    if current_value is None:
        return RiskLimitStatus.warn, None
    if limit <= 0:
        return RiskLimitStatus.ok, None
    ratio = current_value / limit
    if current_value < limit:
        if ratio >= _WARN_RATIO:
            return RiskLimitStatus.warn, ratio
        return RiskLimitStatus.alert, ratio
    return RiskLimitStatus.ok, ratio


def _resolve_open_positions(state: Mapping[str, object]) -> int:
    count = coerce_int(state.get("open_positions_count"))
    if count is not None:
        return max(0, count)

    positions = state.get("open_positions")
    if isinstance(positions, Mapping):
        return sum(1 for value in positions.values() if isinstance(value, Mapping))
    return 0


def _compute_exposures(open_positions: object) -> Mapping[str, float]:
    long_exposure = 0.0
    short_exposure = 0.0
    if isinstance(open_positions, Mapping):
        for payload in open_positions.values():
            if not isinstance(payload, Mapping):
                continue
            quantity = coerce_float(payload.get("quantity")) or 0.0
            current_value = coerce_float(payload.get("current_value"))
            if current_value is None:
                price = coerce_float(payload.get("last_price"))
                if price is None:
                    price = coerce_float(payload.get("avg_price")) or 0.0
                current_value = quantity * price
            if quantity >= 0:
                long_exposure += max(0.0, current_value or 0.0)
            else:
                short_exposure += abs(current_value or 0.0)
    return {"long": long_exposure, "short": short_exposure}


def _approval_rate(telemetry: Mapping[str, object]) -> float | None:
    total = coerce_float(telemetry.get("total_checks")) or 0.0
    approved = coerce_float(telemetry.get("approved")) or 0.0
    if total <= 0:
        return None
    return max(0.0, min(1.0, approved / total))


def _normalise_limits(payload: Mapping[str, object] | None) -> Mapping[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    limits: dict[str, float] = {}
    for key, value in payload.items():
        numeric = coerce_float(value)
        if numeric is not None:
            limits[str(key)] = numeric
    return limits


def _normalise_telemetry(payload: Mapping[str, object] | None) -> Mapping[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        numeric = coerce_float(value)
        if numeric is not None:
            metrics[str(key)] = numeric
    return metrics


def _sanitize_decision(decision: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if not isinstance(decision, Mapping):
        return None
    payload: MutableMapping[str, object] = {}
    for key in ("status", "reason", "symbol", "strategy_id"):
        if key in decision:
            payload[key] = decision[key]
    checks: list[Mapping[str, object]] = []
    checks_payload = decision.get("checks")
    if isinstance(checks_payload, Sequence) and not isinstance(
        checks_payload, (str, bytes, bytearray)
    ):
        for raw in checks_payload:
            if isinstance(raw, Mapping):
                checks.append(dict(raw))
    if checks:
        payload["checks"] = checks
    return dict(payload)


def _extract_decision_checks(
    decision: Mapping[str, object] | None,
    seen: set[str],
) -> Iterable[RiskLimitCheck]:
    if not isinstance(decision, Mapping):
        return ()

    extracted: list[RiskLimitCheck] = []
    checks_payload = decision.get("checks")
    if not (isinstance(checks_payload, Sequence) and not isinstance(
        checks_payload, (str, bytes, bytearray)
    )):
        return ()

    for raw in checks_payload:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name", "")).strip()
        if not name or name in seen:
            continue
        if name in {"open_positions", "daily_drawdown"}:
            continue
        value = coerce_float(raw.get("value"))
        threshold = coerce_float(raw.get("threshold"))
        if threshold is None:
            continue
        threshold_type = _infer_threshold_type(name)
        status, ratio = _grade_limit(value, threshold, threshold_type=threshold_type)
        metadata: dict[str, object] = {}
        for key in ("recommended", "requested", "summary"):
            if key in raw:
                metadata[key] = raw[key]
        extracted.append(
            RiskLimitCheck(
                name=f"decision.{name}",
                value=value,
                threshold=threshold,
                threshold_type=threshold_type,
                status=status,
                ratio=ratio,
                metadata=metadata,
            )
        )
        seen.add(name)
    return extracted


def evaluate_risk_posture(
    portfolio_state: Mapping[str, object] | None,
    risk_limits: Mapping[str, object] | None,
    *,
    last_decision: Mapping[str, object] | None = None,
) -> RiskTelemetrySnapshot:
    state: Mapping[str, object]
    if isinstance(portfolio_state, Mapping):
        state = portfolio_state
    else:
        state = {}

    limits_payload: Mapping[str, object]
    if isinstance(risk_limits, Mapping):
        limits_payload = risk_limits
    else:
        limits_payload = {}

    raw_limits = limits_payload.get("limits")
    limits = _normalise_limits(
        cast(Mapping[str, object] | None, raw_limits) if isinstance(raw_limits, Mapping) else None
    )

    raw_telemetry = limits_payload.get("telemetry")
    telemetry = _normalise_telemetry(
        cast(Mapping[str, object] | None, raw_telemetry)
        if isinstance(raw_telemetry, Mapping)
        else None
    )

    open_positions = _resolve_open_positions(state)
    drawdown = coerce_float(state.get("current_daily_drawdown")) or 0.0
    equity = coerce_float(state.get("equity"))
    peak_equity = coerce_float(state.get("peak_equity"))
    exposures = _compute_exposures(state.get("open_positions"))

    checks: list[RiskLimitCheck] = []
    seen_names: set[str] = set()

    max_positions = limits.get("max_open_positions")
    status, ratio = _grade_limit(
        open_positions, max_positions, threshold_type=RiskThresholdType.maximum
    )
    checks.append(
        RiskLimitCheck(
            name="open_positions",
            value=float(open_positions),
            threshold=max_positions,
            threshold_type=RiskThresholdType.maximum,
            status=status,
            ratio=ratio,
        )
    )
    seen_names.add("open_positions")

    max_drawdown = limits.get("max_daily_drawdown")
    drawdown_status, drawdown_ratio = _grade_limit(
        drawdown,
        max_drawdown,
        threshold_type=RiskThresholdType.maximum,
    )
    checks.append(
        RiskLimitCheck(
            name="daily_drawdown",
            value=drawdown,
            threshold=max_drawdown,
            threshold_type=RiskThresholdType.maximum,
            status=drawdown_status,
            ratio=drawdown_ratio,
        )
    )
    seen_names.add("daily_drawdown")

    checks.extend(_extract_decision_checks(last_decision, seen_names))

    aggregate_status = RiskLimitStatus.ok
    for check in checks:
        if check.status is RiskLimitStatus.alert:
            aggregate_status = RiskLimitStatus.alert
            break
        if check.status is RiskLimitStatus.warn and aggregate_status is RiskLimitStatus.ok:
            aggregate_status = RiskLimitStatus.warn

    snapshot = RiskTelemetrySnapshot(
        status=aggregate_status,
        generated_at=datetime.utcnow(),
        checks=tuple(checks),
        exposures=exposures,
        limits=limits,
        telemetry=telemetry,
        approval_rate=_approval_rate(telemetry),
        portfolio_equity=equity,
        peak_equity=peak_equity,
        last_decision=_sanitize_decision(last_decision),
    )
    return snapshot


def format_risk_markdown(snapshot: RiskTelemetrySnapshot) -> str:
    lines = [
        f"**Status:** {snapshot.status.value.upper()} | checks={len(snapshot.checks)}",
    ]
    if snapshot.portfolio_equity is not None:
        peak = snapshot.peak_equity if snapshot.peak_equity is not None else "n/a"
        lines.append(f"**Equity:** {snapshot.portfolio_equity:,.2f} (peak {peak})")
    if snapshot.approval_rate is not None:
        lines.append(f"**Approval rate:** {snapshot.approval_rate:.0%}")
    if snapshot.exposures:
        long_exposure = snapshot.exposures.get("long", 0.0)
        short_exposure = snapshot.exposures.get("short", 0.0)
        lines.append(
            "**Exposure:** long={:,.2f} short={:,.2f}".format(long_exposure, short_exposure)
        )

    for check in snapshot.checks:
        if check.threshold is None:
            continue
        operator = "≤" if check.threshold_type is RiskThresholdType.maximum else "≥"
        ratio = f" ({check.ratio:.2f})" if check.ratio is not None else ""
        value = f"{check.value:.4f}" if check.value is not None else "n/a"
        lines.append(
            f"- `{check.name}` → {check.status.value.upper()}: {value} {operator} {check.threshold}{ratio}"
        )

    if snapshot.last_decision is not None:
        reason = snapshot.last_decision.get("reason")
        status = snapshot.last_decision.get("status")
        lines.append(
            f"**Last decision:** {status} ({reason or 'no_reason'})"
        )

    return "\n".join(lines)


async def publish_risk_snapshot(
    event_bus: EventBus,
    snapshot: RiskTelemetrySnapshot,
    *,
    source: str = "risk_gateway",
) -> None:
    payload = snapshot.as_dict()
    payload["markdown"] = format_risk_markdown(snapshot)
    event = Event(type="telemetry.risk.posture", payload=payload, source=source)
    await event_bus.publish(event)
