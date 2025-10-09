"""Regression tests for src.orchestration.compose adapters."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.market_data import NoOpMarketDataGateway
from src.core.anomaly import NoOpAnomalyDetector
from src.core.genome import NoOpGenomeProvider
from src.risk import RiskManager
from src.orchestration.compose import (
    AdaptationServiceAdapter,
    ConfigurationProviderAdapter,
    MarketDataGatewayAdapter,
    compose_validation_adapters,
)


@pytest.mark.asyncio
async def test_market_data_gateway_adapter_uses_injected_organ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class DummyOrgan:
        def fetch_data(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((symbol, kwargs))
            return {"symbol": symbol, **kwargs}

    adapter = MarketDataGatewayAdapter(organ=DummyOrgan())

    sync_result = adapter.fetch_data("EURUSD", period="1d", interval="1h")
    assert sync_result["symbol"] == "EURUSD"
    assert calls[0][0] == "EURUSD"

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(("to_thread", args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async_result = await adapter.get_market_data(
        "EURUSD", period="5d", interval="30m", start="2024-01-01", end="2024-01-02"
    )

    assert async_result["period"] == "5d"
    assert any(entry[0] == "to_thread" for entry in calls)


@pytest.mark.asyncio
async def test_adaptation_service_adapter_normalizes_partial_payload() -> None:
    class DummyEngine:
        async def adapt_in_real_time(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace(confidence=0.75, adaptation_strength=0.5)

    adapter = AdaptationServiceAdapter(engine=DummyEngine())

    result = await adapter.adapt_in_real_time({}, {}, {})

    assert result["success"] is True
    assert result["confidence"] == pytest.approx(0.75)
    assert result["quality"] == pytest.approx(0.625)
    assert result["adaptations"] == []


@pytest.mark.asyncio
async def test_adaptation_service_adapter_handles_engine_errors() -> None:
    class FailingEngine:
        async def adapt_in_real_time(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

    adapter = AdaptationServiceAdapter(engine=FailingEngine())

    result = await adapter.adapt_in_real_time({}, {}, {})

    assert result == {"success": False, "quality": 0.0, "adaptations": [], "confidence": 0.0}


def test_configuration_provider_adapter_accessors() -> None:
    class Namespace(dict):
        pass

    class StubConfig(dict):
        system = {"api": {"endpoint": "https://example"}}

        def __getitem__(self, key: str) -> Any:
            return super().__getitem__(key)

    stub = StubConfig({"core": 42})
    stub["system"] = Namespace(stub.system)

    adapter = ConfigurationProviderAdapter(cfg=stub)

    assert adapter.get_value("core") == 42
    assert adapter.get_namespace("system") == {"api": {"endpoint": "https://example"}}
    assert adapter.get_value("missing", default="fallback") == "fallback"


def test_compose_validation_adapters_handles_missing_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {
            "src.data_integration.yfinance_gateway",
            "src.sensory.enhanced.anomaly.manipulation_detection",
            "src.intelligence.sentient_adaptation",
            "src.governance.system_config",
            "src.genome.models.genome_adapter",
        }:
            raise ImportError("missing for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    class FailingGateway:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("gateway not available")

    monkeypatch.setattr(
        "src.orchestration.compose.YahooMarketDataGateway",
        FailingGateway,
    )

    adapters = compose_validation_adapters()

    from src.orchestration.compose import (
        AdaptationServiceAdapter,
        ConfigurationProviderAdapter,
        RegimeClassifierAdapter,
    )

    assert isinstance(adapters["market_data_gateway"], NoOpMarketDataGateway)
    assert isinstance(adapters["anomaly_detector"], NoOpAnomalyDetector)
    assert isinstance(adapters["regime_classifier"], RegimeClassifierAdapter)
    assert isinstance(adapters["risk_manager"], RiskManager)
    assert isinstance(adapters["adaptation_service"], AdaptationServiceAdapter)
    assert getattr(adapters["adaptation_service"], "_engine") is None
    assert isinstance(adapters["configuration_provider"], ConfigurationProviderAdapter)
    assert adapters["configuration_provider"].get_value("missing", default="fallback") == "fallback"
    assert isinstance(adapters["genome_provider"], NoOpGenomeProvider)


def test_compose_validation_adapters_prefers_runtime_implementations() -> None:
    adapters = compose_validation_adapters()

    # YahooMarketDataGateway should be preferred when available.
    from src.data_foundation.ingest.yahoo_gateway import YahooMarketDataGateway
    from src.orchestration.compose import AnomalyDetectorAdapter, ConfigurationProviderAdapter

    assert isinstance(adapters["market_data_gateway"], YahooMarketDataGateway)
    assert isinstance(adapters["anomaly_detector"], AnomalyDetectorAdapter)
    assert isinstance(adapters["configuration_provider"], ConfigurationProviderAdapter)
