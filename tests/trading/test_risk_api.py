from __future__ import annotations

from decimal import Decimal
import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.risk.risk_api import (
    RiskApiError,
    TradingRiskInterface,
    build_runtime_risk_metadata,
    resolve_trading_risk_config,
    resolve_trading_risk_interface,
)


class StubTradingManager:
    def __init__(self, *, risk_config: RiskConfig | None = None) -> None:
        self._risk_config = risk_config or RiskConfig(
            max_risk_per_trade_pct=Decimal("0.01"),
            max_total_exposure_pct=Decimal("0.5"),
        )


class StatusOnlyTradingManager:
    def __init__(self) -> None:
        self.status_payload = {
            "risk_config": {
                "max_risk_per_trade_pct": 0.03,
                "max_total_exposure_pct": 0.4,
                "mandatory_stop_loss": True,
            },
            "policy_limits": {"max_leverage": 5},
            "policy_research_mode": False,
            "snapshot": {"max_drawdown_pct": 0.2},
        }

    def get_risk_status(self) -> dict[str, object]:
        return dict(self.status_payload)


def test_resolve_trading_risk_config_prefers_private_attribute() -> None:
    manager = StubTradingManager(
        risk_config=RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.6"),
        )
    )

    config = resolve_trading_risk_config(manager)

    assert isinstance(config, RiskConfig)
    assert config.max_risk_per_trade_pct == Decimal("0.02")
    assert config.max_total_exposure_pct == Decimal("0.6")


def test_resolve_trading_risk_config_falls_back_to_status_payload() -> None:
    manager = StatusOnlyTradingManager()

    config = resolve_trading_risk_config(manager)
    assert config.max_risk_per_trade_pct == Decimal("0.03")
    assert config.max_total_exposure_pct == Decimal("0.4")


def test_resolve_trading_risk_config_rejects_invalid_payload() -> None:
    class InvalidStatusTradingManager:
        def get_risk_status(self) -> dict[str, object]:
            return {
                "risk_config": {
                    "max_risk_per_trade_pct": -0.5,
                    "max_total_exposure_pct": 0.2,
                }
            }

    with pytest.raises(RiskApiError, match="risk configuration is invalid"):
        resolve_trading_risk_config(InvalidStatusTradingManager())


def test_trading_risk_interface_summary_includes_policy_metadata() -> None:
    interface = TradingRiskInterface(
        config=RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.5"),
        ),
        status={
            "policy_limits": {"max_drawdown_pct": 0.25},
            "policy_research_mode": True,
            "snapshot": {"max_risk_per_trade_pct": 0.02},
        },
    )

    summary = interface.summary()

    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.02)
    assert summary["policy_limits"]["max_drawdown_pct"] == pytest.approx(0.25)
    assert summary["policy_research_mode"] is True
    assert summary["latest_snapshot"]["max_risk_per_trade_pct"] == pytest.approx(0.02)


def test_build_runtime_metadata_respects_status_payload() -> None:
    manager = StatusOnlyTradingManager()

    metadata = build_runtime_risk_metadata(manager)

    assert metadata["max_risk_per_trade_pct"] == pytest.approx(0.03)
    assert metadata["policy_limits"]["max_leverage"] == 5
    assert metadata["latest_snapshot"]["max_drawdown_pct"] == pytest.approx(0.2)


def test_resolve_trading_risk_interface_exposes_status_mapping() -> None:
    manager = StatusOnlyTradingManager()

    interface = resolve_trading_risk_interface(manager)

    assert interface.status is not None
    assert interface.status["policy_limits"]["max_leverage"] == 5


def test_risk_api_error_surfaces_metadata_and_runbook() -> None:
    error = RiskApiError("example failure", details={"manager": "StubManager"})

    metadata = error.to_metadata()

    assert metadata["message"] == "example failure"
    assert metadata["runbook"].endswith("risk_api_contract.md")
    assert metadata["details"]["manager"] == "StubManager"

