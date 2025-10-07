"""Alerting policy helpers for risk and operations readiness.

The high-impact roadmap calls for automated alerting that escalates risk
breaches and system failures across email, SMS, and webhook channels.  This
module provides a lightweight policy engine that can be configured from the
runtime or from YAML/JSON configuration files.  The helpers avoid hard
dependencies on specific providers so unit tests and local development can use
in-memory transports while production deployments wire in SMTP, SMS gateways,
or incident management webhooks.

Example usage::

    from operations.alerts import (
        AlertEvent,
        AlertSeverity,
        build_default_alert_manager,
    )

    manager = build_default_alert_manager()
    event = AlertEvent(
        category="risk_breach",
        severity=AlertSeverity.critical,
        message="Daily VaR exceeded by 45%",
        context={"portfolio": "EMP-CORE"},
    )
    manager.dispatch(event)

The ``build_default_alert_manager`` helper loads a policy that routes risk
breaches to email/webhook channels and escalates system failures to SMS as
well.  Teams with bespoke routing requirements can call ``load_alert_policy``
with their own configuration payload and transport factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
import logging
import smtplib
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence
from urllib import request

logger = logging.getLogger(__name__)

__all__ = [
    "AlertSeverity",
    "AlertEvent",
    "AlertChannel",
    "AlertRule",
    "AlertDispatchResult",
    "AlertManager",
    "load_alert_policy",
    "build_default_alert_manager",
    "default_alert_policy_config",
]


class AlertSeverity(StrEnum):
    """Supported severities for alert routing."""

    info = "info"
    warning = "warning"
    critical = "critical"


_SEVERITY_ORDER: Mapping[AlertSeverity, int] = {
    AlertSeverity.info: 0,
    AlertSeverity.warning: 1,
    AlertSeverity.critical: 2,
}


def _coerce_severity(value: str | AlertSeverity | None, *, default: AlertSeverity) -> AlertSeverity:
    if value is None:
        return default
    if isinstance(value, AlertSeverity):
        return value
    try:
        return AlertSeverity(str(value).lower())
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown alert severity: {value!r}") from exc


def _severity_rank(severity: AlertSeverity) -> int:
    return _SEVERITY_ORDER[severity]


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True, frozen=True)
class AlertEvent:
    """Domain object representing an operational alert."""

    category: str
    severity: AlertSeverity
    message: str
    tags: tuple[str, ...] = ()
    context: Mapping[str, object] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=_now_utc)

    def contains_tags(self, required: Sequence[str]) -> bool:
        if not required:
            return True
        event_tags = set(self.tags)
        return all(tag in event_tags for tag in required)


Transport = Callable[[AlertEvent], None]


@dataclass(slots=True, frozen=True)
class AlertChannel:
    """Alert delivery channel with a minimum severity gate."""

    name: str
    transport: Transport
    channel_type: str
    min_severity: AlertSeverity = AlertSeverity.warning
    metadata: Mapping[str, object] = field(default_factory=dict)

    def should_dispatch(self, severity: AlertSeverity) -> bool:
        return _severity_rank(severity) >= _severity_rank(self.min_severity)


@dataclass(slots=True, frozen=True)
class AlertRule:
    """Routing rule combining categories, severities, and channels."""

    name: str
    categories: tuple[str, ...] = ()
    min_severity: AlertSeverity = AlertSeverity.warning
    channels: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    suppress_seconds: float | None = None

    def matches(self, event: AlertEvent) -> bool:
        if self.categories and event.category not in self.categories:
            return False
        if _severity_rank(event.severity) < _severity_rank(self.min_severity):
            return False
        if self.tags and not event.contains_tags(self.tags):
            return False
        return True


@dataclass(slots=True, frozen=True)
class AlertDispatchResult:
    """Outcome returned by :meth:`AlertManager.dispatch`."""

    event: AlertEvent
    triggered_channels: tuple[str, ...]
    suppressed_rules: tuple[str, ...] = ()
    missing_channels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class AlertManager:
    """Evaluate alert rules and invoke the matching channels."""

    def __init__(
        self,
        channels: Sequence[AlertChannel],
        rules: Sequence[AlertRule],
        *,
        default_channels: Sequence[str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        channel_map: dict[str, AlertChannel] = {}
        for channel in channels:
            if channel.name in channel_map:
                raise ValueError(f"Duplicate alert channel: {channel.name}")
            channel_map[channel.name] = channel

        self._channels = channel_map
        self._rules = tuple(rules)
        self._default_channels = tuple(default_channels or ())
        self._clock = clock or _now_utc
        self._last_dispatch: MutableMapping[tuple[str, str], datetime] = {}

    def dispatch(self, event: AlertEvent) -> AlertDispatchResult:
        """Evaluate the policy for a single event."""

        triggered: list[str] = []
        triggered_set: set[str] = set()
        suppressed: list[str] = []
        missing: dict[str, list[str]] = {}

        now = self._clock()

        for rule in self._rules:
            if not rule.matches(event):
                continue

            if rule.suppress_seconds is not None:
                key = (rule.name, event.category)
                last = self._last_dispatch.get(key)
                if last is not None:
                    delta = now - last
                    if delta < timedelta(seconds=rule.suppress_seconds):
                        suppressed.append(rule.name)
                        continue
                self._last_dispatch[key] = now

            channel_names = rule.channels or self._default_channels
            if not channel_names:
                continue

            for name in channel_names:
                channel = self._channels.get(name)
                if channel is None:
                    missing.setdefault(rule.name, []).append(name)
                    continue
                if not channel.should_dispatch(event.severity):
                    continue
                if name in triggered_set:
                    continue
                channel.transport(event)
                triggered_set.add(name)
                triggered.append(name)

        missing_payload = {
            rule_name: tuple(names)
            for rule_name, names in missing.items()
        }

        return AlertDispatchResult(
            event=event,
            triggered_channels=tuple(triggered),
            suppressed_rules=tuple(suppressed),
            missing_channels=missing_payload,
        )

    def dispatch_many(self, events: Iterable[AlertEvent]) -> list[AlertDispatchResult]:
        """Process multiple events and return their outcomes."""

        return [self.dispatch(event) for event in events]


TransportFactory = Callable[[Mapping[str, object]], Transport]


def _logging_email_transport(config: Mapping[str, object]) -> Transport:
    sender = str(config.get("sender") or "emp-alerts@example.com")
    recipients = tuple(str(item).strip() for item in config.get("recipients", ()) if str(item).strip())
    subject_template = str(config.get("subject_template") or "[{severity}] {category}")
    body_template = str(
        config.get("body_template")
        or "{message}\n\ncontext={context_json}\noccurred_at={occurred_at}"
    )

    def render(event: AlertEvent) -> tuple[str, str]:
        context_json = ""
        if event.context:
            try:
                context_json = json.dumps(event.context, sort_keys=True)
            except TypeError:
                context_json = repr(dict(event.context))
        subject = subject_template.format(
            severity=event.severity.value.upper(),
            category=event.category,
            message=event.message,
            occurred_at=event.occurred_at.isoformat(),
            context_json=context_json,
        )
        body = body_template.format(
            severity=event.severity.value,
            category=event.category,
            message=event.message,
            occurred_at=event.occurred_at.isoformat(),
            context_json=context_json,
        )
        return subject, body

    smtp_host = config.get("smtp_host")
    smtp_port = int(config.get("smtp_port", 587))
    use_tls = bool(config.get("smtp_tls", True))
    smtp_username = config.get("smtp_username")
    smtp_password = config.get("smtp_password")

    if smtp_host:
        timeout = int(config.get("smtp_timeout", 10))

        def transport(event: AlertEvent) -> None:  # pragma: no cover - exercised in integration
            subject, body = render(event)
            message = f"Subject: {subject}\nFrom: {sender}\nTo: {', '.join(recipients or (sender,))}\n\n{body}"
            with smtplib.SMTP(str(smtp_host), smtp_port, timeout=timeout) as client:
                if use_tls:
                    client.starttls()
                if smtp_username:
                    client.login(str(smtp_username), str(smtp_password or ""))
                client.sendmail(sender, list(recipients or (sender,)), message)

    else:

        def transport(event: AlertEvent) -> None:
            subject, body = render(event)
            logger.warning(
                "email_alert",
                extra={
                    "sender": sender,
                    "recipients": recipients,
                    "subject": subject,
                    "body": body,
                },
            )

    return transport


def _logging_sms_transport(config: Mapping[str, object]) -> Transport:
    sender = str(config.get("sender") or "EMP")
    recipients = tuple(str(item).strip() for item in config.get("recipients", ()) if str(item).strip())
    gateway_url = config.get("gateway_url")
    timeout = float(config.get("timeout", 5.0))

    if gateway_url:

        def transport(event: AlertEvent) -> None:  # pragma: no cover - exercised in integration
            payload = {
                "sender": sender,
                "recipients": recipients,
                "message": event.message,
                "severity": event.severity.value,
                "category": event.category,
            }
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                str(gateway_url),
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=timeout):
                pass

    else:

        def transport(event: AlertEvent) -> None:
            logger.warning(
                "sms_alert",
                extra={
                    "sender": sender,
                    "recipients": recipients,
                    "message": event.message,
                    "severity": event.severity.value,
                    "category": event.category,
                },
            )

    return transport


def _logging_webhook_transport(config: Mapping[str, object]) -> Transport:
    url = str(config.get("url") or "")
    headers = {str(k): str(v) for k, v in config.get("headers", {}).items()}
    timeout = float(config.get("timeout", 5.0))

    if url:

        def transport(event: AlertEvent) -> None:  # pragma: no cover - exercised in integration
            payload = {
                "category": event.category,
                "severity": event.severity.value,
                "message": event.message,
                "occurred_at": event.occurred_at.isoformat(),
                "tags": list(event.tags),
                "context": dict(event.context),
            }
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(str(url), data=data, headers={"Content-Type": "application/json", **headers})
            with request.urlopen(req, timeout=timeout):
                pass

    else:

        def transport(event: AlertEvent) -> None:
            logger.warning(
                "webhook_alert",
                extra={
                    "category": event.category,
                    "severity": event.severity.value,
                    "message": event.message,
                    "tags": event.tags,
                    "context": dict(event.context),
                },
            )

    return transport


def _default_transport_factories() -> Mapping[str, TransportFactory]:
    return {
        "email": _logging_email_transport,
        "sms": _logging_sms_transport,
        "webhook": _logging_webhook_transport,
    }


def _build_channels(
    channel_entries: Iterable[Mapping[str, object]],
    *,
    transport_factories: Mapping[str, TransportFactory],
) -> list[AlertChannel]:
    channels: list[AlertChannel] = []
    for entry in channel_entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Channel definitions must be mappings")
        name = entry.get("name")
        if not name:
            raise ValueError("Alert channel missing 'name'")
        channel_type = entry.get("type")
        if not channel_type:
            raise ValueError(f"Alert channel {name!r} missing 'type'")
        factory = transport_factories.get(str(channel_type))
        if factory is None:
            raise ValueError(f"Unsupported alert channel type: {channel_type!r}")
        transport = factory(entry)
        min_severity = _coerce_severity(entry.get("min_severity"), default=AlertSeverity.warning)
        channel = AlertChannel(
            name=str(name),
            transport=transport,
            channel_type=str(channel_type),
            min_severity=min_severity,
            metadata={key: value for key, value in entry.items() if key not in {"name", "type", "min_severity"}},
        )
        channels.append(channel)
    return channels


def _build_rules(rule_entries: Iterable[Mapping[str, object]]) -> list[AlertRule]:
    rules: list[AlertRule] = []
    for entry in rule_entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Alert rules must be mappings")
        name = entry.get("name")
        if not name:
            raise ValueError("Alert rule missing 'name'")
        categories = tuple(str(value) for value in entry.get("categories", ()))
        min_severity = _coerce_severity(entry.get("min_severity"), default=AlertSeverity.warning)
        channels = tuple(str(value) for value in entry.get("channels", ()))
        tags = tuple(str(value) for value in entry.get("tags", ()))
        suppress_seconds_raw = entry.get("suppress_seconds")
        suppress_seconds = float(suppress_seconds_raw) if suppress_seconds_raw is not None else None
        rules.append(
            AlertRule(
                name=str(name),
                categories=categories,
                min_severity=min_severity,
                channels=channels,
                tags=tags,
                suppress_seconds=suppress_seconds,
            )
        )
    return rules


def load_alert_policy(
    config: Mapping[str, object],
    *,
    transport_factories: Mapping[str, TransportFactory] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AlertManager:
    """Construct an :class:`AlertManager` from configuration data."""

    factories = {**_default_transport_factories(), **(transport_factories or {})}

    channels_config = config.get("channels") or ()
    rules_config = config.get("rules") or ()
    default_channels = tuple(str(value) for value in config.get("default_channels", ()))

    channels = _build_channels(channels_config, transport_factories=factories)
    rules = _build_rules(rules_config)

    return AlertManager(
        channels,
        rules,
        default_channels=default_channels,
        clock=clock,
    )


def default_alert_policy_config() -> dict[str, object]:
    """Return the default alert policy for institutional readiness."""

    return {
        "channels": [
            {
                "name": "ops-email",
                "type": "email",
                "min_severity": "warning",
            },
            {
                "name": "ops-sms",
                "type": "sms",
                "min_severity": "critical",
            },
            {
                "name": "ops-webhook",
                "type": "webhook",
                "min_severity": "warning",
            },
        ],
        "rules": [
            {
                "name": "risk-breach",
                "categories": ["risk_breach"],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 300,
            },
            {
                "name": "system-failure",
                "categories": ["system_failure"],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-sms", "ops-webhook"],
                "suppress_seconds": 120,
            },
            {
                "name": "operational-readiness",
                "categories": ["operational.readiness"],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 600,
            },
            {
                "name": "operational-component",
                "categories": [
                    "operational.system_validation",
                    "operational.incident_response",
                    "operational.operational_slos",
                    "operational.drift_sentry",
                    "understanding.drift_sentry",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 300,
            },
            {
                "name": "sensory-drift",
                "categories": [
                    "sensory.drift",
                    "sensory.drift.why",
                    "sensory.drift.what",
                    "sensory.drift.when",
                    "sensory.drift.how",
                    "sensory.drift.anomaly",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 600,
            },
            {
                "name": "system-validation",
                "categories": [
                    "system_validation.status",
                    "system_validation.check",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 300,
            },
            {
                "name": "system-validation-reliability",
                "categories": ["system_validation.reliability"],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 600,
            },
            {
                "name": "incident-response-critical",
                "categories": [
                    "incident_response.missing_runbooks",
                    "incident_response.postmortem_backlog",
                    "incident_response.roster.primary",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook", "ops-sms"],
                "suppress_seconds": 300,
            },
            {
                "name": "incident-response-reliability",
                "categories": [
                    "incident_response.mtta",
                    "incident_response.mttr",
                    "incident_response.metrics_staleness",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook", "ops-sms"],
                "suppress_seconds": 600,
            },
            {
                "name": "incident-response-notifications",
                "categories": [
                    "incident_response.status",
                    "incident_response.issue",
                    "incident_response.drill",
                    "incident_response.training",
                    "incident_response.open_incidents",
                    "incident_response.roster.secondary",
                    "incident_response.chatops",
                ],
                "min_severity": "warning",
                "channels": ["ops-email", "ops-webhook"],
                "suppress_seconds": 300,
            },
        ],
    }


def build_default_alert_manager(
    *,
    transport_factories: Mapping[str, TransportFactory] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AlertManager:
    """Instantiate the default alert manager used by the roadmap."""

    config = default_alert_policy_config()
    return load_alert_policy(config, transport_factories=transport_factories, clock=clock)
