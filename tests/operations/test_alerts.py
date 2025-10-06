"""Tests for the alerting policy helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from operations.alerts import (
    AlertEvent,
    AlertManager,
    AlertSeverity,
    build_default_alert_manager,
    default_alert_policy_config,
    load_alert_policy,
)


def _make_manager(overrides: dict[str, object] | None = None, *, clock=None):
    config = default_alert_policy_config()
    if overrides:
        config.update(overrides)

    captures: list[tuple[str, str, str]] = []

    def _capture_factory(channel_type: str):
        def factory(channel_config):
            name = channel_config["name"]

            def transport(event: AlertEvent) -> None:
                captures.append((name, channel_type, event.category))

            return transport

        return factory

    manager = load_alert_policy(
        config,
        transport_factories={
            "email": _capture_factory("email"),
            "sms": _capture_factory("sms"),
            "webhook": _capture_factory("webhook"),
        },
        clock=clock,
    )
    return manager, captures


def test_risk_breach_routes_to_email_and_webhook() -> None:
    manager, captures = _make_manager()

    event = AlertEvent(
        category="risk_breach",
        severity=AlertSeverity.warning,
        message="VaR breached",
    )
    result = manager.dispatch(event)

    assert result.triggered_channels == ("ops-email", "ops-webhook")
    assert captures == [
        ("ops-email", "email", "risk_breach"),
        ("ops-webhook", "webhook", "risk_breach"),
    ]


def test_system_failure_triggers_sms_escalation() -> None:
    manager, captures = _make_manager()

    event = AlertEvent(
        category="system_failure",
        severity=AlertSeverity.critical,
        message="Data feed offline",
    )
    result = manager.dispatch(event)

    assert set(result.triggered_channels) == {"ops-email", "ops-sms", "ops-webhook"}
    assert set(captures) == {
        ("ops-email", "email", "system_failure"),
        ("ops-sms", "sms", "system_failure"),
        ("ops-webhook", "webhook", "system_failure"),
    }


def test_alerts_respect_severity_thresholds() -> None:
    manager, captures = _make_manager()

    event = AlertEvent(
        category="system_failure",
        severity=AlertSeverity.warning,
        message="Gateway flapped",
    )
    result = manager.dispatch(event)

    assert "ops-sms" not in result.triggered_channels
    assert all(name != "ops-sms" for name, _, _ in captures)


def test_suppression_window_skips_duplicate_events() -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)

    def clock():
        return clock.current

    clock.current = now  # type: ignore[attr-defined]

    manager, captures = _make_manager(clock=clock)

    event = AlertEvent(
        category="risk_breach",
        severity=AlertSeverity.warning,
        message="PnL drawdown",
    )

    first = manager.dispatch(event)
    assert first.triggered_channels
    assert not first.suppressed_rules

    # Second dispatch happens immediately and should be suppressed.
    second = manager.dispatch(event)
    assert not second.triggered_channels
    assert second.suppressed_rules == ("risk-breach",)

    # Move beyond the suppression window and ensure alerts fire again.
    clock.current = now + timedelta(seconds=600)  # type: ignore[attr-defined]
    third = manager.dispatch(event)
    assert third.triggered_channels
    assert captures[:2] == [
        ("ops-email", "email", "risk_breach"),
        ("ops-webhook", "webhook", "risk_breach"),
    ]


def test_operational_readiness_rule_routes_alerts() -> None:
    manager, captures = _make_manager()

    event = AlertEvent(
        category="operational.readiness",
        severity=AlertSeverity.warning,
        message="Operational readiness degraded",
    )

    result = manager.dispatch(event)

    assert set(result.triggered_channels) == {"ops-email", "ops-webhook"}
    assert {channel for channel, _, _ in captures} == {"ops-email", "ops-webhook"}


def test_incident_response_critical_routes_sms() -> None:
    manager, captures = _make_manager()

    event = AlertEvent(
        category="incident_response.missing_runbooks",
        severity=AlertSeverity.critical,
        message="Missing runbooks",
    )

    result = manager.dispatch(event)

    assert set(result.triggered_channels) == {"ops-email", "ops-webhook", "ops-sms"}
    channel_names = {channel for channel, _, _ in captures}
    assert channel_names == {"ops-email", "ops-webhook", "ops-sms"}


def test_rule_tag_filters() -> None:
    config = default_alert_policy_config()
    config["channels"].append({"name": "prod-email", "type": "email"})
    config["rules"].append(
        {
            "name": "prod-only",
            "categories": ["system_failure"],
            "channels": ["prod-email"],
            "tags": ["prod"],
        }
    )

    manager, captures = _make_manager(config)

    dev_event = AlertEvent(
        category="system_failure",
        severity=AlertSeverity.critical,
        message="Dev queue stalled",
        tags=("dev",),
    )
    manager.dispatch(dev_event)
    assert all(entry[0] != "prod-email" for entry in captures)

    prod_event = AlertEvent(
        category="system_failure",
        severity=AlertSeverity.critical,
        message="Prod queue stalled",
        tags=("prod",),
    )
    captures.clear()
    manager.dispatch(prod_event)
    assert any(entry[0] == "prod-email" for entry in captures)


def test_default_manager_uses_default_policy() -> None:
    manager = build_default_alert_manager(
        transport_factories={
            "email": lambda cfg: lambda event: None,
            "sms": lambda cfg: lambda event: None,
            "webhook": lambda cfg: lambda event: None,
        }
    )

    assert isinstance(manager, AlertManager)


def test_invalid_severity_raises_value_error() -> None:
    config = default_alert_policy_config()
    config["channels"][0]["min_severity"] = "bogus"

    with pytest.raises(ValueError):
        load_alert_policy(
            config,
            transport_factories={"email": lambda cfg: lambda event: None},
        )
