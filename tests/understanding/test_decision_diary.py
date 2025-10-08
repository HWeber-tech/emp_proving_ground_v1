from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import pytest

from src.operations.event_bus_failover import EventPublishError
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.probe_registry import ProbeDefinition, ProbeRegistry
from src.thinking.adaptation.policy_router import PolicyDecision, RegimeState

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - python 3.10 fallback
    UTC = timezone.utc  # type: ignore[assignment]


def _fixed_now() -> datetime:
    return datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture()
def fixed_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    deterministic = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(
        "src.understanding.decision_diary.uuid.uuid4",
        lambda: deterministic,
    )


def test_decision_diary_record_and_reload(tmp_path, fixed_uuid) -> None:
    _ = fixed_uuid
    registry = ProbeRegistry.from_definitions(
        (
            ProbeDefinition(
                probe_id="drift.sentry",
                name="Drift Sentry",
                description="Belief drift watchdog",
                owner="governance",
                contact="governance@example.com",
                severity="warn",
                runbook="docs/operations/runbooks/drift_sentry_response.md",
                tags=("drift", "shadow"),
            ),
        ),
        generated_at=_fixed_now(),
    )
    store = DecisionDiaryStore(tmp_path / "diary.json", now=_fixed_now, probe_registry=registry)

    decision = PolicyDecision(
        tactic_id="alpha.shadow",
        parameters={"size": 1, "currency": "EURUSD"},
        selected_weight=1.25,
        guardrails={"requires_diary": True},
        rationale="Shadow review",
        experiments_applied=("volatility-boost",),
        reflection_summary={"score": 0.82},
    )
    regime = RegimeState(
        regime="calm",
        confidence=0.92,
        features={"volatility": 0.4, "skew": -0.1},
        timestamp=_fixed_now(),
    )

    entry = store.record(
        policy_id="alpha.policy",
        decision=decision,
        regime_state=regime,
        outcomes={"shadow_latency_ms": 45, "reviewer_confidence": 0.88},
        belief_state={"version": "1.0", "symbol": "EURUSD", "metadata": {"source": "replay"}},
        probes=[{"probe_id": "drift.sentry", "status": "ok"}],
        notes=["auto-captured"],
        metadata={"session": "alpha-shadow"},
    )

    assert entry.entry_id.startswith("dd-20240101T120000Z-12345678")
    assert entry.probes[0].owner == "governance"
    assert entry.probes[0].contact == "governance@example.com"
    assert entry.probes[0].severity == "warn"

    exported = json.loads(store.export_json())
    assert exported["entries"][0]["policy_id"] == "alpha.policy"
    assert exported["entries"][0]["probes"][0]["owner"] == "governance"
    assert "weight_breakdown" in exported["entries"][0]["decision"]

    markdown = store.export_markdown()
    assert "## Probe ownership" in markdown
    assert "auto-captured" in markdown

    reloaded = DecisionDiaryStore(tmp_path / "diary.json")
    reloaded_entry = reloaded.get(entry.entry_id)
    assert reloaded_entry is not None
    assert reloaded_entry.probes[0].runbook.endswith("drift_sentry_response.md")
    assert reloaded.probe_registry.get("drift.sentry").owner == "governance"

    assert store.exists(entry.entry_id)


def test_decision_diary_publish_event(tmp_path, fixed_uuid, monkeypatch) -> None:
    _ = fixed_uuid
    events: list[object] = []

    def _fake_publish_event_with_failover(event_bus, event, **_kwargs):
        events.append(event)

    monkeypatch.setattr(
        "src.understanding.decision_diary.publish_event_with_failover",
        _fake_publish_event_with_failover,
    )

    registry = ProbeRegistry()

    store = DecisionDiaryStore(
        tmp_path / "diary.json",
        now=_fixed_now,
        probe_registry=registry,
        event_bus=object(),
    )

    decision = PolicyDecision(
        tactic_id="alpha.shadow",
        parameters={"size": 1},
        selected_weight=1.0,
        guardrails={},
        rationale="",
        experiments_applied=(),
        reflection_summary={},
    )
    regime = RegimeState(
        regime="calm",
        confidence=0.5,
        features={},
        timestamp=_fixed_now(),
    )

    entry = store.record(
        policy_id="alpha.policy",
        decision=decision,
        regime_state=regime,
        outcomes={},
    )

    assert events, "expected decision diary event to be published"
    event = events[0]
    assert event.type == "governance.decision_diary.recorded"
    assert event.payload["entry"]["policy_id"] == "alpha.policy"
    assert entry.entry_id in event.payload["entry"]["entry_id"]
    assert "markdown" in event.payload
    assert "weight_breakdown" in event.payload["entry"]["decision"]


def test_decision_diary_publish_event_failure(tmp_path, fixed_uuid, monkeypatch, caplog) -> None:
    _ = fixed_uuid
    def _raise_publish(*_args, **_kwargs):
        raise EventPublishError(stage="runtime", event_type="governance.decision_diary.recorded")

    monkeypatch.setattr(
        "src.understanding.decision_diary.publish_event_with_failover",
        _raise_publish,
    )

    registry = ProbeRegistry()
    store = DecisionDiaryStore(
        tmp_path / "diary.json",
        now=_fixed_now,
        probe_registry=registry,
        event_bus=object(),
    )

    decision = PolicyDecision(
        tactic_id="alpha.shadow",
        parameters={},
        selected_weight=1.0,
        guardrails={},
        rationale="",
        experiments_applied=(),
        reflection_summary={},
    )
    regime = RegimeState(
        regime="calm",
        confidence=0.5,
        features={},
        timestamp=_fixed_now(),
    )

    caplog.set_level(logging.WARNING)
    store.record(
        policy_id="alpha.policy",
        decision=decision,
        regime_state=regime,
        outcomes={},
    )

    assert "Decision diary event publish failed" in caplog.text
