from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from src.data_integration.yfinance_gateway import YFinanceGateway


class _RaisingTicker:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise self._error


class _BadTicker:
    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("unexpected failure")


@pytest.fixture(autouse=True)
def _patch_ticker(monkeypatch):
    # Default ticker returns a simple dataframe; specific tests will override the callable.
    def default_ticker(symbol: str) -> Any:
        frame = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [10]})

        class _Ticker:
            def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
                return frame

        return _Ticker()

    monkeypatch.setattr("yfinance.Ticker", default_ticker)


def test_fetch_data_logs_expected_errors(monkeypatch, caplog) -> None:
    gateway = YFinanceGateway()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: _RaisingTicker(ValueError("bad range")))

    with caplog.at_level(logging.WARNING):
        result = gateway.fetch_data("AAPL")

    assert result is None
    assert "bad range" in caplog.text


def test_fetch_data_reraises_unexpected_errors(monkeypatch) -> None:
    gateway = YFinanceGateway()
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: _BadTicker())

    with pytest.raises(RuntimeError):
        gateway.fetch_data("AAPL")
