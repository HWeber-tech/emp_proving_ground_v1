from __future__ import annotations

import datetime as _datetime
import enum as _enum
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if not hasattr(_datetime, "UTC"):
    setattr(_datetime, "UTC", timezone.utc)

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = _StrEnum

if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(SRC_DIR)]
    sys.modules["src"] = src_pkg
else:
    sys.modules["src"].__path__ = [str(SRC_DIR)]  # type: ignore[attr-defined]

if "src.core" not in sys.modules:
    core_pkg = types.ModuleType("src.core")
    core_pkg.__path__ = [str(SRC_DIR / "core")]
    sys.modules["src.core"] = core_pkg
else:
    sys.modules["src.core"].__path__ = [str(SRC_DIR / "core")]  # type: ignore[attr-defined]

if "src.operations" not in sys.modules:
    ops_pkg = types.ModuleType("src.operations")
    ops_pkg.__path__ = [str(SRC_DIR / "operations")]
    sys.modules["src.operations"] = ops_pkg
else:
    sys.modules["src.operations"].__path__ = [str(SRC_DIR / "operations")]  # type: ignore[attr-defined]


core_event_bus = types.ModuleType("src.core.event_bus")


class _Event:
    def __init__(self, *, type: str, payload: Any, source: str | None = None) -> None:
        self.type = type
        self.payload = payload
        self.source = source


class _EventBus:
    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: _Event) -> int:
        return 1


class _TopicBus:
    def publish_sync(self, event_type: str, payload: Any, *, source: str | None = None) -> None:
        pass


def _get_global_bus() -> _TopicBus:
    return _TopicBus()


core_event_bus.Event = _Event
core_event_bus.EventBus = _EventBus
core_event_bus.TopicBus = _TopicBus
core_event_bus.get_global_bus = _get_global_bus

sys.modules["src.core.event_bus"] = core_event_bus


def _load_module(module_name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_load_module("src.operations.event_bus_failover", SRC_DIR / "operations" / "event_bus_failover.py")
_load_module("src.operations.observability_diary", SRC_DIR / "operations" / "observability_diary.py")

from src.operations.observability_diary import (  # noqa: E402  (module loaded dynamically)
    DecisionNarrationCapsule,
    PolicyLedgerDiff,
    build_decision_narration_capsule,
    publish_decision_narration_capsule,
)


UTC = timezone.utc


class _StubEventBus:
    def __init__(self, *, running: bool = True, result: Any = 1, record: bool = True) -> None:
        self._running = running
        self._result = result
        self._record = record
        self.events: list[Any] = []

    def publish_from_sync(self, event: Any) -> Any:  # pragma: no cover - called via helper
        if self._record:
            self.events.append(event)
        return self._result

    def is_running(self) -> bool:
        return self._running


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str | None]] = []

    def publish_sync(self, event_type: str, payload: Any, *, source: str | None = None) -> None:
        self.events.append((event_type, payload, source))


def _capsule_kwargs() -> dict[str, Any]:
    return {
        "capsule_id": "2024-06-01T00:00:00Z::loop-update",
        "window_start": "2024-06-01T00:00:00",
        "window_end": "2024-06-01T04:00:00+00:00",
        "policy_diffs": [
            {
                "policy_id": "risk.exposure",
                "change_type": "updated",
                "before": {"max_gross_exposure": 0.50},
                "after": {"max_gross_exposure": 0.40},
                "approvals": ["risk", "compliance"],
                "notes": ["tightened gross exposure"],
                "metadata": {"ticket": "AUTH-42"},
            }
        ],
        "sigma_metrics": {
            "symbol": "EURUSD",
            "sigma_before": 0.25,
            "sigma_after": 0.22,
            "sigma_target": 0.20,
            "stability_index": 0.9,
            "lookback_days": 21,
        },
        "throttle_states": [
            {
                "name": "volatility",
                "state": "reduced",
                "active": True,
                "multiplier": 0.7,
                "reason": "sigma contraction",
            },
            {
                "name": "drawdown",
                "state": "stable",
                "active": False,
                "multiplier": 1.0,
            },
        ],
        "notes": ["Ledger diff validated"],
        "metadata": {"author": "alpha", "release": "loop-42"},
        "generated_at": datetime(2024, 6, 1, 5, tzinfo=UTC),
    }


def test_build_decision_narration_capsule_normalises_inputs() -> None:
    capsule = build_decision_narration_capsule(**_capsule_kwargs())

    assert capsule.generated_at.tzinfo is UTC
    assert capsule.window_start == datetime(2024, 6, 1, 0, tzinfo=UTC)
    assert capsule.window_end == datetime(2024, 6, 1, 4, tzinfo=UTC)
    assert capsule.policy_diffs[0].approvals == ("risk", "compliance")
    assert capsule.notes == ("Ledger diff validated",)

    sigma = capsule.sigma_stability
    assert sigma.symbol == "EURUSD"
    assert sigma.sigma_before == pytest.approx(0.25)
    assert sigma.delta == pytest.approx(-0.03)

    payload = capsule.as_dict()
    assert payload["capsule_id"] == "2024-06-01T00:00:00Z::loop-update"
    assert payload["sigma_stability"]["delta"] == pytest.approx(-0.03)
    assert payload["metadata"]["release"] == "loop-42"

    markdown = capsule.to_markdown()
    assert "risk.exposure" in markdown
    assert "delta=-0.0300" in markdown


def test_publish_decision_narration_capsule_prefers_runtime_bus() -> None:
    bus = _StubEventBus()
    capsule = build_decision_narration_capsule(**_capsule_kwargs())
    topic_bus = _StubTopicBus()

    publish_decision_narration_capsule(
        bus,
        capsule,
        global_bus_factory=lambda: topic_bus,
    )

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "observability.decision_narration"
    assert event.payload["version"] == 1
    assert event.payload["capsule"]["capsule_id"] == capsule.capsule_id
    assert topic_bus.events == []


def test_publish_decision_narration_capsule_falls_back_on_runtime_none() -> None:
    bus = _StubEventBus(result=None, record=False)
    capsule = build_decision_narration_capsule(**_capsule_kwargs())
    topic_bus = _StubTopicBus()

    publish_decision_narration_capsule(
        bus,
        capsule,
        global_bus_factory=lambda: topic_bus,
    )

    assert not bus.events
    assert topic_bus.events
    event_type, payload, source = topic_bus.events[0]
    assert event_type == "observability.decision_narration"
    assert payload["capsule"]["capsule_id"] == capsule.capsule_id
    assert source == "observability.diary"


def test_policy_ledger_diff_instance_passthrough() -> None:
    diff = PolicyLedgerDiff(
      policy_id="risk.limits",
      change_type="created",
      after={"max_exposure": 0.35},
    )
    capsule = build_decision_narration_capsule(
        capsule_id="capsule-1",
        window_start=None,
        window_end=None,
        policy_diffs=[diff],
        sigma_metrics={"symbol": "UNKNOWN"},
        throttle_states=[],
    )
    assert capsule.policy_diffs == (diff,)
    assert isinstance(capsule, DecisionNarrationCapsule)
