from __future__ import annotations

import pytest

from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.runtime.bootstrap_runtime import BootstrapRuntime
from src.runtime.predator_app import build_professional_predator_app


@pytest.mark.asyncio()
async def test_trade_throttle_configured_from_system_extras() -> None:
    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "TRADE_THROTTLE_ENABLED": "true",
            "TRADE_THROTTLE_NAME": "compliance_guard",
            "TRADE_THROTTLE_MAX_TRADES": "1",
            "TRADE_THROTTLE_WINDOW_SECONDS": "60",
            "TRADE_THROTTLE_MIN_SPACING_SECONDS": "30",
            "TRADE_THROTTLE_SCOPE_FIELDS": "strategy_id,symbol",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)

    throttle_snapshot = app.sensory_organ.trading_manager.get_trade_throttle_snapshot()
    assert throttle_snapshot is not None
    assert throttle_snapshot.get("name") == "compliance_guard"

    metadata = throttle_snapshot.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("max_trades") == 1
    assert metadata.get("window_seconds") == pytest.approx(60.0)
    assert metadata.get("min_spacing_seconds") == pytest.approx(30.0)
    scope = metadata.get("scope")
    assert isinstance(scope, dict)
    assert set(scope.keys()) == {"strategy_id", "symbol"}

    stats = app.sensory_organ.trading_manager.get_execution_stats()
    assert isinstance(stats.get("trade_throttle"), dict)

    await app.__aenter__()
    await app.__aexit__(None, None, None)
