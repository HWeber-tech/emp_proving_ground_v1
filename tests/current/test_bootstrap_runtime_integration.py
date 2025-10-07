from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Mapping

import typing as _typing
import sys
import types


def _shim_class_getitem(name: str) -> type:
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):  # pragma: no cover - simple compatibility shim
            return item

    _Placeholder.__name__ = name
    return _Placeholder


if not hasattr(_typing, "Unpack"):
    _typing.Unpack = _shim_class_getitem("Unpack")  # type: ignore[attr-defined]

if not hasattr(_typing, "Self"):
    _typing.Self = _typing.TypeVar("Self")  # type: ignore[attr-defined]

if "scipy" not in sys.modules:
    scipy_module = types.ModuleType("scipy")
    signal_module = types.ModuleType("scipy.signal")
    stats_module = types.ModuleType("scipy.stats")

    def _fake_find_peaks(*_args, **_kwargs):  # pragma: no cover - lightweight stub
        return [], {}

    def _fake_zscore(values, *_, **__):  # pragma: no cover - lightweight stub
        length = len(values) if hasattr(values, "__len__") else 0
        return [0.0] * length

    signal_module.find_peaks = _fake_find_peaks  # type: ignore[attr-defined]
    stats_module.zscore = _fake_zscore  # type: ignore[attr-defined]
    scipy_module.signal = signal_module  # type: ignore[attr-defined]
    scipy_module.stats = stats_module  # type: ignore[attr-defined]
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module
    sys.modules["scipy.stats"] = stats_module

if "simplefix" not in sys.modules:
    class _FakeFixMessage:  # pragma: no cover - lightweight stub
        def __init__(self) -> None:
            self.pairs: list[tuple[int, str]] = []

        def append_pair(self, tag: int, value: str) -> None:
            self.pairs.append((tag, value))

        def encode(self) -> bytes:
            return b""

    simplefix_module = types.ModuleType("simplefix")
    simplefix_module.FixMessage = _FakeFixMessage  # type: ignore[attr-defined]
    sys.modules["simplefix"] = simplefix_module

if "src.runtime.fix_pilot" not in sys.modules:
    fix_pilot_module = types.ModuleType("src.runtime.fix_pilot")

    class _FakeFixIntegrationPilot:  # pragma: no cover - lightweight stub
        def __init__(self, *_, **__):
            pass

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def snapshot(self) -> None:
            return None

    fix_pilot_module.FixIntegrationPilot = _FakeFixIntegrationPilot  # type: ignore[attr-defined]
    sys.modules["src.runtime.fix_pilot"] = fix_pilot_module

if "src.runtime.runtime_builder" not in sys.modules:
    runtime_builder_module = types.ModuleType("src.runtime.runtime_builder")

    class _StubRuntimeApplication:  # pragma: no cover - lightweight stub
        def __init__(self, *_, **__):
            pass

        def summary(self) -> Mapping[str, Any]:
            return {}

    class _StubRuntimeWorkload:  # pragma: no cover - lightweight stub
        def __init__(self, *_, **__):
            pass

    def _stub_build_runtime_app(*_args, **_kwargs):  # pragma: no cover - lightweight stub
        raise NotImplementedError

    async def _stub_execute_timescale_ingest(*_args, **__kwargs):  # pragma: no cover - stubbed runtime hook
        return True, None

    runtime_builder_module.RuntimeApplication = _StubRuntimeApplication  # type: ignore[attr-defined]
    runtime_builder_module.RuntimeWorkload = _StubRuntimeWorkload  # type: ignore[attr-defined]
    runtime_builder_module.build_professional_runtime_application = _stub_build_runtime_app  # type: ignore[attr-defined]
    runtime_builder_module._execute_timescale_ingest = _stub_execute_timescale_ingest  # type: ignore[attr-defined]
    runtime_builder_module.__all__ = [  # type: ignore[attr-defined]
        "RuntimeApplication",
        "RuntimeWorkload",
        "build_professional_runtime_application",
        "_execute_timescale_ingest",
    ]
    sys.modules["src.runtime.runtime_builder"] = runtime_builder_module

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.governance.system_config import (
    ConnectionProtocol,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.runtime import build_professional_predator_app
from src.runtime.bootstrap_runtime import BootstrapRuntime
from src.understanding.decision_diary import DecisionDiaryStore


@pytest.mark.asyncio()
async def test_bootstrap_runtime_build_and_run() -> None:
    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_0,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "BOOTSTRAP_TICK_INTERVAL": "0.01",
            "BOOTSTRAP_MAX_TICKS": "3",
            "BOOTSTRAP_BUY_THRESHOLD": "0.05",
            "BOOTSTRAP_SELL_THRESHOLD": "0.05",
            "BOOTSTRAP_ORDER_SIZE": "2",
            "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)
    runtime = app.sensory_organ

    async with app:
        await asyncio.sleep(0.1)
        status = runtime.status()
        assert status["ticks_processed"] >= 1
        assert status["decisions"] >= 1
        assert "telemetry" in status
        assert status["telemetry"]["equity"] >= 0
        assert status["telemetry"]["last_decision"] is not None
        assert status["vision_alignment"]["status"] in {"ready", "progressing", "gap"}

        summary = app.summary()
        assert summary["status"] == "RUNNING"
        sensory_status = summary["components"].get("sensory_status")
        assert sensory_status is not None
        assert sensory_status["ticks_processed"] >= 1
        sensors = summary["components"].get("sensors", [])
        assert {"why", "what", "when", "how", "anomaly"}.issubset(set(sensors))
        audit_entries = summary.get("sensor_audit", [])
        assert audit_entries
        assert audit_entries[0]["symbol"] == "EURUSD"
        risk_section = summary.get("risk")
        assert risk_section is not None
        assert risk_section["snapshot"]["status"] in {"ok", "warn", "alert"}

    assert runtime.status()["running"] is False
    assert app.summary()["status"] == "STOPPED"


@pytest.mark.asyncio()
async def test_bootstrap_runtime_release_posture(tmp_path) -> None:
    ledger_path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    diary_path = tmp_path / "decision_diary.json"
    diary_store = DecisionDiaryStore(diary_path)
    diary_entry = diary_store.record(
        policy_id="alpha",
        decision={
            "tactic_id": "alpha",
            "parameters": {},
            "guardrails": {},
            "rationale": "bootstrap release",
            "experiments_applied": (),
            "reflection_summary": {},
        },
        regime_state={
            "regime": "calm",
            "confidence": 1.0,
            "features": {},
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
        outcomes={},
    )
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id=diary_entry.entry_id,
    )

    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_0,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "BOOTSTRAP_TICK_INTERVAL": "0.01",
            "BOOTSTRAP_MAX_TICKS": "3",
            "BOOTSTRAP_BUY_THRESHOLD": "0.05",
            "BOOTSTRAP_SELL_THRESHOLD": "0.05",
            "BOOTSTRAP_ORDER_SIZE": "1",
            "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
            "BOOTSTRAP_STRATEGY_ID": "alpha",
            "POLICY_LEDGER_PATH": str(ledger_path),
            "DECISION_DIARY_PATH": str(diary_path),
        },
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)
    runtime = app.sensory_organ

    async with app:
        await asyncio.sleep(0.1)
        status = runtime.status()
        release_posture = status.get("release_posture")
        assert isinstance(release_posture, dict)
        assert release_posture.get("stage") == PolicyLedgerStage.PAPER.value
        summary = app.summary()
        risk_section = summary.get("risk")
        assert isinstance(risk_section, dict)
        release_summary = risk_section.get("release")
        assert isinstance(release_summary, dict)
        assert release_summary.get("stage") == PolicyLedgerStage.PAPER.value
        thresholds = release_summary.get("thresholds", {})
        assert thresholds.get("stage") == PolicyLedgerStage.PAPER.value

    assert runtime.status()["running"] is False
