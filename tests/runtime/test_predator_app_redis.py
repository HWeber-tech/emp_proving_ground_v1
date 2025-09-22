from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from src.data_foundation.cache import ManagedRedisCache
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


@pytest.mark.asyncio
async def test_professional_app_uses_configured_redis(monkeypatch) -> None:
    fake_clients: list[fakeredis.FakeRedis] = []

    def fake_configure(settings, *, factory=None, ping=True):
        client = fakeredis.FakeRedis.from_url(settings.connection_url())
        fake_clients.append(client)
        if ping:
            client.ping()
        return client

    monkeypatch.setattr("src.runtime.predator_app.configure_redis_client", fake_configure)

    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "15",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)
    assert fake_clients, "Expected configure_redis_client to be invoked"

    redis_client = app.sensory_organ.redis_client
    assert isinstance(redis_client, ManagedRedisCache)
    assert redis_client.raw_client is fake_clients[0]
    assert app.sensory_organ.trading_manager.portfolio_monitor.redis_client is redis_client
    metrics = redis_client.metrics()
    assert metrics["namespace"].startswith("emp")

    await app.__aenter__()
    await app.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_professional_app_falls_back_without_credentials(monkeypatch) -> None:
    def boom(*args, **kwargs):  # pragma: no cover - defensive, ensures function not called
        raise AssertionError("configure_redis_client should not be invoked")

    monkeypatch.setattr("src.runtime.predator_app.configure_redis_client", boom)

    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={},
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)
    redis_client = app.sensory_organ.redis_client
    assert isinstance(redis_client, ManagedRedisCache)
    assert redis_client.policy.namespace.startswith("emp")
    assert app.sensory_organ.trading_manager.portfolio_monitor.redis_client is redis_client

    await app.__aenter__()
    await app.__aexit__(None, None, None)
