from __future__ import annotations

import logging

import pytest

from src.risk import RiskManager
from src.risk.risk_manager_impl import RiskManagerImpl
from src.trading.risk.risk_api import RiskApiError, resolve_trading_risk_config
from src.trading.trading_manager import _coerce_risk_config


def test_risk_validation_failures_log_events(caplog: pytest.LogCaptureFixture) -> None:
    invalid_payload = {"max_risk_per_trade_pct": "invalid"}

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError):
            RiskManagerImpl(initial_balance=10_000, risk_config=invalid_payload)

    assert any(
        record.message == "RiskManagerImpl rejected invalid risk_config payload"
        for record in caplog.records
    )

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError):
            RiskManager(config=invalid_payload)

    assert any(
        record.message == "RiskManager rejected invalid risk_config payload"
        for record in caplog.records
    )

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError):
            _coerce_risk_config(invalid_payload)

    assert any(
        record.message == "TradingManager rejected invalid risk_config payload"
        for record in caplog.records
    )

    caplog.clear()

    class InvalidStatusTradingManager:
        _risk_config = None

        def get_risk_status(self) -> dict[str, object]:
            return {"risk_config": invalid_payload}

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RiskApiError):
            resolve_trading_risk_config(InvalidStatusTradingManager())

    assert any(
        record.message
        == "Risk API rejected invalid trading manager risk configuration"
        for record in caplog.records
    )
