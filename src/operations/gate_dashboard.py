"""Renderable dashboard for gate metrics vs thresholds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

try:  # Python < 3.11 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback
    from datetime import timezone

    UTC = timezone.utc  # type: ignore[misc]


class GateMetricDirection(StrEnum):
    """Desired direction for metric thresholds."""

    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class GateDashboardStatus(StrEnum):
    """Severity level for gate metrics and dashboards."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


_STATUS_ORDER: Mapping[GateDashboardStatus, int] = {
    GateDashboardStatus.OK: 0,
    GateDashboardStatus.WARN: 1,
    GateDashboardStatus.FAIL: 2,
}


def _scale(value: float | None, multiplier: float) -> float | None:
    if value is None:
        return None
    return float(value) * multiplier


def _format_numeric(
    value: float | None,
    *,
    multiplier: float,
    unit: str | None,
    format_spec: str | None,
) -> str:
    numeric = _scale(value, multiplier)
    if numeric is None:
        return "n/a"

    if format_spec:
        text = format(numeric, format_spec)
    else:
        magnitude = abs(numeric)
        if magnitude >= 10_000:
            text = f"{numeric:,.0f}"
        elif magnitude >= 1_000:
            text = f"{numeric:,.1f}"
        elif magnitude >= 100:
            text = f"{numeric:.0f}"
        elif magnitude >= 10:
            text = f"{numeric:.1f}"
        elif magnitude >= 1:
            text = f"{numeric:.2f}"
        else:
            text = f"{numeric:.3f}"

    if unit:
        suffix = unit if unit in {"%", "bp"} or unit.startswith("%") else f" {unit}"
        return f"{text}{suffix}"
    return text


def _format_threshold(
    value: float | None,
    direction: GateMetricDirection,
    *,
    multiplier: float,
    unit: str | None,
    format_spec: str | None,
) -> str:
    if value is None:
        return "n/a"
    operator = ">=" if direction is GateMetricDirection.AT_LEAST else "<="
    return f"{operator} {_format_numeric(value, multiplier=multiplier, unit=unit, format_spec=format_spec)}"


@dataclass(frozen=True)
class GateMetricVisual:
    """Visual representation of a single gate metric."""

    name: str
    value: float | None
    warn_threshold: float | None = None
    fail_threshold: float | None = None
    direction: GateMetricDirection = GateMetricDirection.AT_LEAST
    unit: str | None = None
    multiplier: float = 1.0
    format_spec: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def status(self) -> GateDashboardStatus:
        value = self.value
        if value is None:
            return GateDashboardStatus.FAIL

        if self.direction is GateMetricDirection.AT_LEAST:
            if self.fail_threshold is not None and value < float(self.fail_threshold):
                return GateDashboardStatus.FAIL
            if self.warn_threshold is not None and value < float(self.warn_threshold):
                return GateDashboardStatus.WARN
            return GateDashboardStatus.OK

        # AT_MOST
        if self.fail_threshold is not None and value > float(self.fail_threshold):
            return GateDashboardStatus.FAIL
        if self.warn_threshold is not None and value > float(self.warn_threshold):
            return GateDashboardStatus.WARN
        return GateDashboardStatus.OK

    def gauge(self, width: int = 20) -> str:
        width = max(6, int(width))
        inner_width = width - 2
        bar = ["-"] * inner_width

        value = self.value
        if value is None:
            return "[" + "?" * inner_width + "]"

        target = self.warn_threshold
        fallback = self.fail_threshold
        if target is None:
            target = fallback

        ratio: float | None = None
        if target is not None and target != 0:
            if self.direction is GateMetricDirection.AT_LEAST:
                ratio = max(0.0, min(value / target, 1.0))
            else:
                ratio = max(0.0, min((target - value) / target, 1.0))

        if ratio is None:
            pointer_pos = max(0, min(inner_width - 1, int(inner_width / 2)))
            bar[pointer_pos] = "|"
        else:
            filled = max(0, min(inner_width, int(round(inner_width * ratio))))
            for idx in range(filled):
                bar[idx] = "#"

            pointer_pos = max(0, min(inner_width - 1, int(round(inner_width * ratio))))
            bar[pointer_pos] = "|"

        if self.fail_threshold is not None:
            if self.direction is GateMetricDirection.AT_LEAST:
                breached = value < float(self.fail_threshold)
            else:
                breached = value > float(self.fail_threshold)
            if breached:
                bar[-1] = "!"

        return "[" + "".join(bar) + "]"

    @property
    def value_text(self) -> str:
        return _format_numeric(
            self.value,
            multiplier=self.multiplier,
            unit=self.unit,
            format_spec=self.format_spec,
        )

    @property
    def warn_text(self) -> str:
        return _format_threshold(
            self.warn_threshold,
            self.direction,
            multiplier=self.multiplier,
            unit=self.unit,
            format_spec=self.format_spec,
        )

    @property
    def fail_text(self) -> str:
        return _format_threshold(
            self.fail_threshold,
            self.direction,
            multiplier=self.multiplier,
            unit=self.unit,
            format_spec=self.format_spec,
        )

    def detail_line(self) -> str:
        status_text = self.status().value.upper()
        parts = [f"{self.name}: {self.value_text} [{status_text}] {self.gauge()}"]

        thresholds: list[str] = []
        if self.warn_threshold is not None:
            thresholds.append(f"warn {self.warn_text}")
        if self.fail_threshold is not None:
            thresholds.append(f"fail {self.fail_text}")
        if thresholds:
            parts.append("(" + ", ".join(thresholds) + ")")
        if self.description:
            parts.append(f"- {self.description}")
        return " ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "warn_threshold": self.warn_threshold,
            "fail_threshold": self.fail_threshold,
            "direction": self.direction.value,
            "unit": self.unit,
            "multiplier": self.multiplier,
            "format_spec": self.format_spec,
            "status": self.status().value,
            "value_display": self.value_text,
            "warn_display": self.warn_text,
            "fail_display": self.fail_text,
            "gauge": self.gauge(),
        }
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class GateDashboard:
    """Dashboard aggregating multiple gate metrics."""

    generated_at: datetime
    metrics: Sequence[GateMetricVisual]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def status(self) -> GateDashboardStatus:
        current = GateDashboardStatus.OK
        for metric in self.metrics:
            candidate = metric.status()
            if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
                current = candidate
        return current

    def status_breakdown(self) -> Mapping[GateDashboardStatus, int]:
        counts = Counter(metric.status() for metric in self.metrics)
        return {
            GateDashboardStatus.OK: counts.get(GateDashboardStatus.OK, 0),
            GateDashboardStatus.WARN: counts.get(GateDashboardStatus.WARN, 0),
            GateDashboardStatus.FAIL: counts.get(GateDashboardStatus.FAIL, 0),
        }

    def headline(self) -> str:
        counts = self.status_breakdown()
        total = len(self.metrics)
        return (
            f"Gate health {self.status().value.upper()} — "
            f"{counts[GateDashboardStatus.FAIL]} fail / "
            f"{counts[GateDashboardStatus.WARN]} warn across {total} metrics"
        )

    def panel_details(self) -> tuple[str, ...]:
        return tuple(metric.detail_line() for metric in self.metrics)

    def to_markdown(self) -> str:
        lines = [
            "# Gate dashboard",
            f"- Generated at: {self.generated_at.astimezone(UTC).isoformat()}",
            f"- Overall status: {self.status().value.upper()}",
            "",
            "| Metric | Value | Warn | Fail | Status | Gauge |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for metric in self.metrics:
            lines.append(
                "| {name} | {value} | {warn} | {fail} | {status} | `{gauge}` |".format(
                    name=metric.name,
                    value=metric.value_text,
                    warn=metric.warn_text,
                    fail=metric.fail_text,
                    status=metric.status().value.upper(),
                    gauge=metric.gauge(width=18),
                )
            )

        if not self.metrics:
            lines.append("| _none_ | n/a | n/a | n/a | OK | `[]` |")

        return "\n".join(lines).rstrip() + "\n"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status().value,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["status_breakdown"] = {
            key.value: value for key, value in self.status_breakdown().items()
        }
        return payload


def _normalise_generated_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _normalise_metadata(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def _coerce_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def build_gate_dashboard(
    metrics: Sequence[GateMetricVisual | Mapping[str, Any] | Any],
    *,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GateDashboard:
    """Construct a gate dashboard from metric payloads."""

    metric_objects: list[GateMetricVisual] = []
    for item in metrics:
        if isinstance(item, GateMetricVisual):
            metric_objects.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError("Gate dashboard metrics must be mappings or GateMetricVisual instances")

        name = str(item.get("name")) if item.get("name") is not None else None
        if not name:
            raise ValueError("Gate metric requires a 'name'")

        value_raw = item.get("value")
        warn_raw = item.get("warn", item.get("warn_threshold"))
        fail_raw = item.get("fail", item.get("fail_threshold"))
        direction_raw = item.get("direction", GateMetricDirection.AT_LEAST.value)
        unit = item.get("unit")
        multiplier = item.get("multiplier", 1.0)
        format_spec = item.get("format") or item.get("format_spec")
        description = item.get("description")
        metric_metadata = _normalise_metadata(item.get("metadata"))

        try:
            direction = GateMetricDirection(str(direction_raw))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported gate metric direction: {direction_raw}") from exc

        def _maybe_float(payload: Any) -> float | None:
            if payload is None:
                return None
            try:
                return float(payload)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Gate metric '{name}' has invalid numeric value: {payload}") from exc

        metric_objects.append(
            GateMetricVisual(
                name=name,
                value=_maybe_float(value_raw),
                warn_threshold=_maybe_float(warn_raw),
                fail_threshold=_maybe_float(fail_raw),
                direction=direction,
                unit=str(unit) if unit is not None else None,
                multiplier=float(multiplier),
                format_spec=str(format_spec) if format_spec is not None else None,
                description=str(description) if description is not None else None,
                metadata=metric_metadata,
            )
        )

    moment = generated_at or datetime.now(tz=UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)

    return GateDashboard(
        generated_at=moment,
        metrics=tuple(metric_objects),
        metadata=_normalise_metadata(metadata),
    )


def build_gate_dashboard_from_mapping(payload: Mapping[str, Any]) -> GateDashboard:
    """Coerce a generic mapping into a :class:`GateDashboard`."""

    metrics_payload = payload.get("metrics")
    metrics_sequence = _coerce_sequence(metrics_payload)

    generated_at = _normalise_generated_at(payload.get("generated_at"))
    metadata = _normalise_metadata(payload.get("metadata"))

    return build_gate_dashboard(
        metrics_sequence, generated_at=generated_at, metadata=metadata
    )


__all__ = [
    "GateDashboard",
    "GateDashboardStatus",
    "GateMetricDirection",
    "GateMetricVisual",
    "build_gate_dashboard",
    "build_gate_dashboard_from_mapping",
]
