from __future__ import annotations

import asyncio

import pytest

from src.governance.system_config import ConnectionProtocol, EmpEnvironment, EmpTier, RunMode, SystemConfig
from src.runtime import build_professional_predator_app
from src.runtime.bootstrap_runtime import BootstrapRuntime


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
        assert runtime.execution_engine.fills
        assert "telemetry" in status
        assert status["telemetry"]["equity"] >= 0
        assert status["telemetry"]["last_decision"] is not None
        assert status["vision_alignment"]["status"] in {"ready", "progressing", "gap"}

        summary = app.summary()
        assert summary["status"] == "RUNNING"
        sensory_status = summary["components"].get("sensory_status")
        assert sensory_status is not None
        assert sensory_status["ticks_processed"] >= 1

    assert runtime.status()["running"] is False
    assert app.summary()["status"] == "STOPPED"
