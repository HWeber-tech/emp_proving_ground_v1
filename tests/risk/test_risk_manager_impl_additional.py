from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.risk import RiskManager, create_risk_manager
from src.risk.risk_manager_impl import RiskManagerImpl
from src.trading.risk.market_regime_detector import MarketRegimeResult, RegimeLabel


class _CrashMapping(dict):
    def get(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - helper for tests
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_calculate_position_size_exception_returns_zero() -> None:
    config = RiskConfig(min_position_size=5_000, max_position_size=10_000)
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=config)

    size = await manager.calculate_position_size(_CrashMapping())

    assert size == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_calculate_position_size_returns_zero_without_budget() -> None:
    config = RiskConfig(min_position_size=1_000, max_position_size=10_000)
    manager = RiskManagerImpl(initial_balance=0, risk_config=config)

    size = await manager.calculate_position_size(
        {"symbol": "EURUSD", "confidence": 0.6, "stop_loss_pct": 0.02}
    )

    assert size == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_calculate_position_size_returns_zero_when_below_minimum_lot() -> None:
    config = RiskConfig(min_position_size=10_000, max_position_size=50_000)
    manager = RiskManagerImpl(initial_balance=1_000, risk_config=config)

    size = await manager.calculate_position_size(
        {"symbol": "EURUSD", "confidence": 0.55, "stop_loss_pct": 0.05}
    )

    assert size == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_calculate_position_size_handles_non_finite_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    monkeypatch.setattr(manager, "_compute_risk_budget", lambda: float("inf"))

    size = await manager.calculate_position_size(
        {"symbol": "EURUSD", "confidence": 0.6, "stop_loss_pct": 0.02}
    )

    assert size == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_calculate_position_size_returns_bounded_size() -> None:
    config = RiskConfig(min_position_size=1_000, max_position_size=5_000)
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=config)

    size = await manager.calculate_position_size(
        {"symbol": "EURUSD", "confidence": 0.7, "stop_loss_pct": 0.02}
    )

    assert 1_000 <= size <= 5_000


@pytest.mark.asyncio
async def test_calculate_position_size_applies_macro_event_halt() -> None:
    config = RiskConfig(min_position_size=1_000, max_position_size=100_000)
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=config)

    now = datetime.now(timezone.utc)
    size = await manager.calculate_position_size(
        {
            "symbol": "EURUSD",
            "confidence": 0.7,
            "stop_loss_pct": 0.02,
            "timestamp": now,
            "macro_events": [now + timedelta(seconds=60)],
        }
    )

    assert size == pytest.approx(0.0)
    assert manager.telemetry.get("slow_context", {}).get("reason") == "macro_event_proximity"


@pytest.mark.asyncio
async def test_calculate_position_size_applies_vix_multiplier() -> None:
    config = RiskConfig(min_position_size=1_000, max_position_size=120_000)
    baseline_manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    high_vix_manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)

    now = datetime.now(timezone.utc)
    base_signal = {
        "symbol": "EURUSD",
        "confidence": 0.7,
        "stop_loss_pct": 0.02,
        "timestamp": now,
    }

    baseline_size = await baseline_manager.calculate_position_size(base_signal)

    high_vix_signal = dict(base_signal)
    high_vix_signal["vix"] = 39.5
    adjusted_size = await high_vix_manager.calculate_position_size(high_vix_signal)

    assert baseline_size > 0.0
    assert adjusted_size == pytest.approx(baseline_size * 0.3, rel=1e-6)
    assert high_vix_manager.telemetry.get("slow_context", {}).get("reason") == "high_volatility"


@pytest.mark.asyncio
async def test_validate_position_rejects_non_positive_size() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 0.0, "entry_price": 1.2, "stop_loss_pct": 0.02}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_rejects_non_positive_entry_price() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000.0, "entry_price": 0.0, "stop_loss_pct": 0.02}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_rejects_out_of_bounds_sizes() -> None:
    config = RiskConfig(min_position_size=1_000, max_position_size=5_000)
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=config)

    below_min = await manager.validate_position(
        {"symbol": "EURUSD", "size": 100.0, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )
    above_max = await manager.validate_position(
        {"symbol": "EURUSD", "size": 10_000.0, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )

    assert below_min is False
    assert above_max is False


@pytest.mark.asyncio
async def test_validate_position_requires_stop_loss_when_mandatory() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000.0, "entry_price": 1.2}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_rejects_when_risk_budget_unavailable() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.update_account_balance(0)

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000.0, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_rejects_on_aggregate_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=RiskConfig())

    monkeypatch.setattr(manager, "_aggregate_position_risk", lambda: {})
    monkeypatch.setattr(manager.risk_manager, "assess_risk", lambda positions: 1.5)

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 1_000.0, "entry_price": 1.1, "stop_loss_pct": 0.01}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_accepts_valid_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=RiskConfig())

    monkeypatch.setattr(manager, "_aggregate_position_risk", lambda: {})
    monkeypatch.setattr(manager.risk_manager, "assess_risk", lambda positions: 0.5)

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000.0, "entry_price": 1.1, "stop_loss_pct": 0.01}
    )

    assert result is True


def test_risk_manager_impl_accepts_mapping_payload() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config={})

    assert isinstance(manager._risk_config, RiskConfig)


def test_risk_manager_impl_rejects_invalid_payload_type() -> None:
    with pytest.raises(TypeError):
        RiskManagerImpl(initial_balance=10_000, risk_config=object())


def test_risk_manager_impl_rejects_invalid_payload_values() -> None:
    with pytest.raises(ValueError):
        RiskManagerImpl(
            initial_balance=10_000,
            risk_config={"max_risk_per_trade_pct": Decimal("2.0")},
        )


def test_risk_manager_requires_risk_config() -> None:
    with pytest.raises(ValueError):
        RiskManager()


def test_risk_manager_rejects_invalid_payload_type() -> None:
    with pytest.raises(TypeError):
        RiskManager(config=object())


def test_risk_manager_rejects_invalid_payload_values() -> None:
    with pytest.raises(ValueError):
        RiskManager(config={"max_risk_per_trade_pct": Decimal("2.0")})


def test_update_slow_context_macro_freeze() -> None:
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=RiskConfig())

    baseline = manager._compute_risk_budget()
    decision = manager.update_slow_context({"macro_block": True})

    assert decision.multiplier == pytest.approx(0.0)
    assert manager._slow_context_multiplier == pytest.approx(0.0)
    assert manager._compute_risk_budget() == pytest.approx(0.0)
    assert manager.telemetry["slow_context_drivers"]["macro"] is True

    manager.update_slow_context({})
    assert manager._compute_risk_budget() == pytest.approx(baseline)


def test_update_slow_context_throttle_multiplier() -> None:
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=RiskConfig())

    baseline = manager._compute_risk_budget()
    decision = manager.update_slow_context({"volatility_throttle": True})

    assert decision.multiplier == pytest.approx(0.3)
    assert manager._slow_context_multiplier == pytest.approx(0.3)
    assert manager._compute_risk_budget() == pytest.approx(baseline * 0.3)


def test_update_slow_context_allows_explicit_override() -> None:
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=RiskConfig())

    decision = manager.update_slow_context({"size_multiplier": "0.3", "macro_block": True})

    assert decision.multiplier == pytest.approx(0.3)
    assert manager._slow_context_multiplier == pytest.approx(0.3)
    assert manager.telemetry["slow_context_multiplier"] == pytest.approx(0.3)


def test_risk_manager_facade_validate_trade_respects_limits() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        min_position_size=1_000,
        max_position_size=10_000,
    )
    manager = RiskManager(config=config, initial_balance=50_000)

    approved = manager.validate_trade(
        size=Decimal("2_000"),
        entry_price=Decimal("1.25"),
        stop_loss_pct=Decimal("0.02"),
        symbol="EURUSD",
    )
    rejected = manager.validate_trade(
        size=Decimal("500"),
        entry_price=Decimal("1.25"),
        stop_loss_pct=Decimal("0.02"),
        symbol="EURUSD",
    )

    assert approved is True
    assert rejected is False


def test_risk_manager_validate_trade_rejects_missing_stop_loss() -> None:
    manager = RiskManager(config=RiskConfig())

    result = manager.validate_trade(
        size=Decimal("1_500"),
        entry_price=Decimal("1.25"),
    )

    assert result is False


def test_risk_manager_rejects_when_risk_budget_depleted() -> None:
    manager = RiskManager(config=RiskConfig())
    manager.update_account_balance(0.0)

    result = manager.validate_trade(
        size=Decimal("1_000"),
        entry_price=Decimal("1.1"),
        stop_loss_pct=Decimal("0.02"),
    )

    assert result is False


def test_risk_manager_rejects_when_aggregate_risk_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManager(config=RiskConfig())

    # Ensure the direct risk budget check passes so the aggregate branch executes.
    result = manager.validate_trade(
        size=Decimal("1_000"),
        entry_price=Decimal("1.1"),
        stop_loss_pct=Decimal("0.01"),
        symbol="EURUSD",
    )
    assert result is True

    monkeypatch.setattr(manager.risk_manager, "assess_risk", lambda positions: 1.5)

    rejected = manager.validate_trade(
        size=Decimal("1_000"),
        entry_price=Decimal("1.1"),
        stop_loss_pct=Decimal("0.01"),
        symbol="EURUSD",
    )

    assert rejected is False


def test_risk_manager_rejects_sector_exposure_breach() -> None:
    config = RiskConfig(
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.02")},
        max_position_size=200_000,
    )
    manager = RiskManager(config=config, initial_balance=Decimal("100_000"))
    manager.add_position("EURUSD", 50_000, 1.1, stop_loss_pct=0.02)

    original = manager.risk_manager.assess_risk
    manager.risk_manager.assess_risk = lambda positions: 0.5  # type: ignore[assignment]

    result = manager.validate_trade(
        size=Decimal("75_000"),
        entry_price=Decimal("1.1"),
        stop_loss_pct=Decimal("0.02"),
        symbol="EURUSD",
    )

    manager.risk_manager.assess_risk = original  # type: ignore[assignment]

    assert result is False


def test_risk_manager_sector_budget_zero_rejects_positive_exposure() -> None:
    config = RiskConfig(
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.5")},
    )
    manager = RiskManager(config=config, initial_balance=Decimal("50_000"))
    manager.update_account_balance(0.0)

    result = manager.validate_trade(
        size=Decimal("10_000"),
        entry_price=Decimal("1.2"),
        stop_loss_pct=Decimal("0.03"),
        symbol="EURUSD",
    )

    assert result is False


def test_risk_manager_normalizes_symbol_keys() -> None:
    manager = RiskManager(config=RiskConfig())

    manager.add_position("eurusd", 50_000, 1.1, stop_loss_pct=0.02)

    assert "EURUSD" in manager.positions
    assert "eurusd" not in manager.positions

    manager.update_position_value("EuRuSd", 1.2)

    entry = manager.positions["EURUSD"]
    assert entry["current_price"] == pytest.approx(1.2)

    metrics = manager.get_position_risk("eurusd")
    assert metrics["symbol"] == "EURUSD"


def test_risk_manager_validate_trade_merges_legacy_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RiskConfig(max_position_size=500_000)
    manager = RiskManager(config=config, initial_balance=Decimal("100_000"))

    manager.positions["eurusd"] = {
        "size": 20_000.0,
        "entry_price": 1.0,
        "entry_time": datetime.now(),
        "stop_loss_pct": 0.02,
    }

    captured: dict[str, float] = {}

    def _capture(positions: Mapping[str, float]) -> float:
        captured.clear()
        captured.update(positions)
        return 0.0

    monkeypatch.setattr(manager.risk_manager, "assess_risk", _capture)

    approved = manager.validate_trade(
        size=Decimal("10_000"),
        entry_price=Decimal("1.0"),
        stop_loss_pct=Decimal("0.02"),
        symbol="EURusd",
    )

    assert approved is True
    assert set(captured.keys()) == {"EURUSD"}
    assert "EURUSD" in manager.positions


def test_risk_manager_rejects_when_trade_risk_exceeds_budget() -> None:
    manager = RiskManager(config=RiskConfig())

    rejected = manager.validate_trade(
        size=Decimal("10_000"),
        entry_price=Decimal("1.5"),
        stop_loss_pct=Decimal("0.05"),
    )

    assert rejected is False


def test_risk_manager_factory_returns_facade() -> None:
    manager = create_risk_manager(config=RiskConfig())

    assert isinstance(manager, RiskManager)


def test_risk_manager_legacy_factory_removed() -> None:
    risk_module = importlib.import_module("src.risk")

    with pytest.raises(AttributeError):
        getattr(risk_module, "get_risk_manager")


def test_risk_manager_impl_assess_risk_handles_non_numeric_inputs() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    score = manager.assess_risk({"EURUSD": object()})

    assert score == pytest.approx(0.0)


def test_risk_manager_impl_last_snapshot_returns_copy() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.assess_risk({"EURUSD": 10_000.0})

    snapshot = manager.last_snapshot
    snapshot["total_exposure"] = 0.0

    assert manager.risk_manager.last_snapshot["total_exposure"] != 0.0


def test_risk_manager_impl_update_market_regime_records_metadata() -> None:
    class StubDetector:
        def __init__(self, result: MarketRegimeResult) -> None:
            self._result = result
            self.calls: list[Mapping[str, object]] = []

        def detect_regime(self, market_data: Mapping[str, object] | list[float]) -> MarketRegimeResult:
            self.calls.append({"data": list(market_data) if isinstance(market_data, list) else market_data})
            return self._result

    result = MarketRegimeResult(
        regime=RegimeLabel("storm"),
        confidence=0.6,
        realised_volatility=0.04,
        annualised_volatility=0.2,
        sample_size=50,
        risk_multiplier=0.25,
        blocked=True,
        timestamp=datetime.now(timezone.utc),
        diagnostics={"sample_size": 50.0},
    )
    detector = StubDetector(result)
    manager = RiskManagerImpl(
        initial_balance=10_000,
        risk_config=RiskConfig(),
        market_regime_detector=detector,
    )

    resolved = manager.update_market_regime([0.01, -0.02, 0.03])

    assert resolved is result
    assert manager.telemetry["last_regime"] == "storm"
    assert manager.telemetry["regime_blocked"] is True
    assert manager._regime_risk_multiplier == pytest.approx(0.25)
    assert detector.calls


def test_risk_manager_impl_update_market_regime_fail_closed_on_errors() -> None:
    class FlakyDetector:
        def __init__(self, recovery_result: MarketRegimeResult) -> None:
            self._result = recovery_result
            self.calls = 0

        def detect_regime(
            self, _market_data: Mapping[str, object] | list[float]
        ) -> MarketRegimeResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("regime detector unavailable")
            return self._result

    recovery = MarketRegimeResult(
        regime=RegimeLabel("normal"),
        confidence=0.8,
        realised_volatility=0.01,
        annualised_volatility=0.05,
        sample_size=100,
        risk_multiplier=0.75,
        blocked=False,
        timestamp=datetime.now(timezone.utc),
        diagnostics={"sample_size": 100.0},
    )
    detector = FlakyDetector(recovery)
    manager = RiskManagerImpl(
        initial_balance=50_000,
        risk_config=RiskConfig(),
        market_regime_detector=detector,
    )

    failure = manager.update_market_regime([0.01, -0.02, 0.03])

    assert failure.regime.value == "unknown"
    assert failure.risk_multiplier == pytest.approx(0.0)
    assert failure.blocked is True
    assert manager._regime_risk_multiplier == pytest.approx(0.0)
    assert manager.telemetry["regime_blocked"] is True
    assert manager.telemetry["regime_error"] == "regime detector unavailable"

    recovered = manager.update_market_regime([0.02, -0.01, 0.04])

    assert recovered is recovery
    assert manager.telemetry["last_regime"] == "normal"
    assert "regime_error" not in manager.telemetry
    assert manager._regime_risk_multiplier == pytest.approx(recovery.risk_multiplier)


def test_risk_manager_impl_update_limits_overrides_all_supported_fields() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    manager.update_limits(
        {
            "max_risk_per_trade_pct": 0.01,
            "max_drawdown": 0.3,
            "max_total_exposure_pct": 0.4,
            "max_leverage": 5,
            "min_position_size": 500,
            "max_position_size": 2_000,
            "mandatory_stop_loss": False,
            "research_mode": True,
            "var_confidence": 0.92,
            "var_simulations": 128,
            "instrument_sector_map": {"eurusd": "fx"},
            "sector_exposure_limits": {"fx": 0.05},
        }
    )

    assert manager.config.max_position_risk == pytest.approx(0.01)
    assert manager.config.max_drawdown == pytest.approx(0.4)
    assert manager.config.max_total_exposure == pytest.approx(0.4)
    assert manager.config.max_leverage == pytest.approx(5.0)
    assert manager._min_position_size == pytest.approx(500.0)
    assert manager._max_position_size == pytest.approx(2_000.0)
    assert manager._mandatory_stop_loss is False
    assert manager._research_mode is True
    assert manager._var_confidence == pytest.approx(0.92)
    assert manager._var_simulations == 128
    assert manager._instrument_sector_map == {"EURUSD": "fx"}
    assert manager._sector_limits["fx"] == pytest.approx(0.05)


def test_risk_manager_impl_update_limits_ignores_non_positive_overrides() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    baseline_risk = manager._risk_per_trade
    baseline_drawdown = manager.config.max_drawdown
    baseline_exposure = manager.config.max_total_exposure
    baseline_leverage = manager.config.max_leverage
    baseline_min_size = manager._min_position_size
    baseline_max_size = manager._max_position_size

    manager.update_limits({"max_risk_per_trade_pct": 0.0})
    assert manager._risk_per_trade == pytest.approx(baseline_risk)

    manager.update_limits({"max_total_exposure_pct": 0.0})
    assert manager.config.max_total_exposure == pytest.approx(baseline_exposure)

    manager.update_limits({"max_drawdown": 0.0})
    assert manager.config.max_drawdown == pytest.approx(0.0)
    assert manager.config.max_total_exposure == pytest.approx(baseline_exposure)

    manager.update_limits({"max_leverage": 0.0})
    assert manager.config.max_leverage == pytest.approx(baseline_leverage)

    manager.update_limits({"min_position_size": 0.0})
    assert manager._min_position_size == pytest.approx(0.0)

    manager.update_limits({"max_position_size": 0.0})
    assert manager._max_position_size == pytest.approx(manager._min_position_size)

    manager.update_limits({"instrument_sector_map": {"eurusd": ""}})
    assert manager._instrument_sector_map == {}

    manager.update_limits({"sector_exposure_limits": {"fx": 0.0}})
    assert manager._sector_limits == {}


def test_risk_manager_impl_check_risk_thresholds_handles_zero_budget() -> None:
    config = RiskConfig(
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.5")},
    )
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    manager.add_position("EURUSD", 10_000, 1.25, stop_loss_pct=0.02)
    manager.update_account_balance(0)

    assert manager.check_risk_thresholds() is False


def test_risk_manager_impl_evaluate_portfolio_risk_handles_bad_payload() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    risk = manager.evaluate_portfolio_risk({"EURUSD": object()})

    assert risk == pytest.approx(1.0)


def test_risk_manager_impl_evaluate_portfolio_risk_fail_closed_on_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    def _boom(_: Mapping[str, float]) -> float:
        raise RuntimeError("engine boom")

    monkeypatch.setattr(manager.risk_manager, "assess_risk", _boom)

    risk = manager.evaluate_portfolio_risk({"EURUSD": 1_000.0})

    assert risk == pytest.approx(1.0)


def _build_sector_enabled_manager() -> RiskManagerImpl:
    config = RiskConfig(
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.05")},
        max_position_size=250_000,
    )
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    manager.add_position("EURUSD", 50_000, 1.2, stop_loss_pct=0.02)
    return manager


def test_get_risk_summary_includes_sector_and_market_metrics() -> None:
    manager = _build_sector_enabled_manager()

    summary = manager.get_risk_summary(
        returns=[0.01, -0.004, 0.003],
        confidence=0.95,
    )

    assert "sector_exposure" in summary
    assert summary["sector_exposure"]["FX"]["exposure"] > 0.0
    market_risk = summary["market_risk"]
    assert market_risk["confidence"] == pytest.approx(0.95)
    assert market_risk["expected_shortfall"]["historical"]["confidence"] == pytest.approx(0.95)


def test_get_risk_summary_omits_sector_when_not_configured() -> None:
    manager = RiskManagerImpl(initial_balance=25_000, risk_config=RiskConfig())

    summary = manager.get_risk_summary()

    assert "sector_exposure" not in summary


def test_get_risk_summary_skips_market_risk_when_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    manager = _build_sector_enabled_manager()

    caplog.set_level(logging.WARNING, logger="src.risk.risk_manager_impl")

    summary = manager.get_risk_summary(returns=[], confidence=0.9)

    assert "market_risk" not in summary
    assert any("Market risk computation skipped" in record.message for record in caplog.records)


def test_calculate_portfolio_risk_reports_sector_and_market_metrics() -> None:
    manager = _build_sector_enabled_manager()

    metrics = manager.calculate_portfolio_risk(
        returns=[0.006, -0.002, 0.001],
        confidence=0.9,
    )

    assert metrics["risk_amount"] > 0.0
    assert metrics["sector_exposure"]["FX"]["utilisation"] > 0.0
    market_risk = metrics["market_risk"]
    assert market_risk["confidence"] == pytest.approx(0.9)


def test_calculate_portfolio_risk_handles_empty_returns(caplog: pytest.LogCaptureFixture) -> None:
    manager = _build_sector_enabled_manager()
    caplog.set_level(logging.WARNING, logger="src.risk.risk_manager_impl")

    metrics = manager.calculate_portfolio_risk(returns=[], confidence=0.9)

    assert "market_risk" not in metrics
    assert any("Market risk computation skipped" in record.message for record in caplog.records)


def test_calculate_portfolio_risk_without_sector_or_returns() -> None:
    manager = RiskManagerImpl(initial_balance=20_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 5_000, 1.0, stop_loss_pct=0.02)

    metrics = manager.calculate_portfolio_risk()

    assert "sector_exposure" not in metrics
    assert "market_risk" not in metrics


def test_get_position_risk_handles_missing_and_tracked_symbols() -> None:
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=RiskConfig())

    assert manager.get_position_risk("EURUSD") == {}

    manager.add_position("EURUSD", 25_000, 1.0, stop_loss_pct=0.02)
    manager.update_position_value("EURUSD", 1.1)

    snapshot = manager.get_position_risk("eurusd")

    assert snapshot["symbol"] == "EURUSD"
    assert snapshot["risk_amount"] > 0.0


def test_update_account_balance_recomputes_drawdown_multiplier() -> None:
    config = RiskConfig(max_drawdown_pct=Decimal("0.25"))
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    baseline_budget = manager._compute_risk_budget()

    manager.update_account_balance(80_000)

    assert manager._drawdown_multiplier == pytest.approx(0.25)
    assert manager._compute_risk_budget() < baseline_budget * 0.3
