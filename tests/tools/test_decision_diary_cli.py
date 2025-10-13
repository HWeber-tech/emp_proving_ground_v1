from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.thinking.adaptation.policy_router import (
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.probe_registry import ProbeDefinition, ProbeRegistry
from tools.understanding import decision_diary_cli

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - python 3.10 fallback
    UTC = timezone.utc  # type: ignore[assignment]


@pytest.fixture()
def fixed_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    deterministic = uuid.UUID("feedface-0000-0000-0000-feedface0000")
    monkeypatch.setattr(
        "src.understanding.decision_diary.uuid.uuid4",
        lambda: deterministic,
    )


def _now() -> datetime:
    return datetime(2024, 3, 1, 15, 30, tzinfo=UTC)


def _build_store(tmp_path: Path) -> tuple[DecisionDiaryStore, ProbeRegistry, Path]:
    diary_path = tmp_path / "diary.json"
    registry = ProbeRegistry.from_definitions(
        (
            ProbeDefinition(
                probe_id="drift.sentry",
                name="Drift Sentry",
                description="Belief drift watchdog",
                owner="ops",
                contact="ops@example.com",
                severity="warn",
            ),
        ),
        generated_at=_now(),
    )
    store = DecisionDiaryStore(diary_path, now=_now, probe_registry=registry)
    decision = PolicyDecision(
        tactic_id="alpha.shadow",
        parameters={"size": 2},
        selected_weight=1.05,
        guardrails={"requires_diary": True},
        rationale="Shadow cadence",
        experiments_applied=(),
        reflection_summary={},
        decision_timestamp=_now(),
    )
    regime = RegimeState(regime="balanced", confidence=0.7, features={}, timestamp=_now())
    store.record(
        policy_id="alpha.policy",
        decision=decision,
        regime_state=regime,
        outcomes={},
        probes=[{"probe_id": "drift.sentry", "status": "ok"}],
    )
    return store, registry, diary_path


def _build_diary_with_reflections(tmp_path: Path) -> Path:
    diary_path = tmp_path / "diary_with_reflection.json"
    store = DecisionDiaryStore(diary_path, now=_now)
    router = PolicyRouter()
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="breakout",
                base_weight=1.0,
                regime_bias={"bull": 1.3},
                objectives=("alpha",),
                tags=("momentum",),
            ),
            PolicyTactic(
                tactic_id="defence",
                base_weight=0.8,
                regime_bias={"bear": 1.5},
                objectives=("stability",),
                tags=("reversion",),
            ),
        )
    )

    base = _now()
    bullish = RegimeState(
        regime="bull",
        confidence=0.75,
        features={"volatility": 0.2},
        timestamp=base,
    )
    bearish = RegimeState(
        regime="bear",
        confidence=0.7,
        features={"volatility": 0.45},
        timestamp=base + timedelta(minutes=3),
    )

    first_decision = router.route(bullish)
    second_decision = router.route(bearish)

    store.record(
        policy_id="alpha.policy",
        decision=first_decision,
        regime_state=bullish,
        outcomes={},
    )
    store.record(
        policy_id="alpha.policy",
        decision=second_decision,
        regime_state=bearish,
        outcomes={},
    )
    return diary_path


def test_cli_export_diary_json(tmp_path, capsys, fixed_uuid) -> None:
    _ = fixed_uuid
    store, registry, diary_path = _build_store(tmp_path)
    registry_path = tmp_path / "registry.json"
    registry.write(registry_path)

    exit_code = decision_diary_cli.main(
        [
            "export-diary",
            "--diary",
            str(diary_path),
            "--probe-registry",
            str(registry_path),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["entries"][0]["decision"]["tactic_id"] == "alpha.shadow"
    assert payload["entries"][0]["probes"][0]["owner"] == "ops"


def test_cli_export_probes_markdown(tmp_path, capsys, fixed_uuid) -> None:
    _ = fixed_uuid
    _, registry, _ = _build_store(tmp_path)
    registry_path = tmp_path / "registry.json"
    registry.write(registry_path)

    exit_code = decision_diary_cli.main(
        [
            "export-probes",
            "--registry",
            str(registry_path),
            "--format",
            "markdown",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Understanding loop probe registry" in captured.out
    assert "drift.sentry" in captured.out


def test_cli_summarise_reflection_markdown(tmp_path, capsys) -> None:
    diary_path = _build_diary_with_reflections(tmp_path)

    exit_code = decision_diary_cli.main(
        [
            "summarise-reflection",
            "--diary",
            str(diary_path),
            "--format",
            "markdown",
            "--max-tactics",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PolicyRouter reflection summary" in captured.out
    assert "Decisions analysed: 2" in captured.out
    assert "Top tactics" in captured.out


def test_cli_summarise_reflection_json(tmp_path, capsys) -> None:
    diary_path = _build_diary_with_reflections(tmp_path)

    exit_code = decision_diary_cli.main(
        [
            "summarise-reflection",
            "--diary",
            str(diary_path),
            "--format",
            "json",
            "--window",
            "1",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["metadata"]["total_decisions"] == 1
    assert payload["digest"]["total_decisions"] == 1
    assert payload["metadata"]["window"] == 1
