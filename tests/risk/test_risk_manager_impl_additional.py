from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.risk.manager import RiskManager, create_risk_manager, get_risk_manager
from src.risk.risk_manager_impl import RiskManagerImpl
from src.trading.risk.market_regime_detector import MarketRegimeResult, RegimeLabel


class _CrashMapping(dict):
    def get(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - helper for tests
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_calculate_position_size_exception_returns_configured_minimum() -> None:
    config = RiskConfig(min_position_size=5_000, max_position_size=10_000)
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=config)

    size = await manager.calculate_position_size(_CrashMapping())

    assert size == pytest.approx(5_000.0)


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


def test_risk_manager_factories_return_facade() -> None:
    manager = create_risk_manager(config=RiskConfig())
    legacy = get_risk_manager(config=RiskConfig())

    assert isinstance(manager, RiskManager)
    assert isinstance(legacy, RiskManager)


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

    assert risk == pytest.approx(0.0)
