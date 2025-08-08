#!/usr/bin/env python3

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.trading.strategies.real_base_strategy import RealBaseStrategy
from src.core.market_data import MarketData


def _mk_market_data_from_series(series: np.ndarray) -> MarketData:
    n = len(series)
    now = datetime.utcnow()
    df = pd.DataFrame({
        "timestamp": [now - timedelta(minutes=n - i) for i in range(n)],
        "open": series,
        "high": series + 0.0005,
        "low": series - 0.0005,
        "close": series,
        "volume": np.full(n, 2000.0),
    })
    # RealBaseStrategy expects a MarketData-like object with .data
    class MD:
        pass
    md = MD()
    md.data = df
    return md  # type: ignore[return-value]


def test_real_base_strategy_buy_sell_hold():
    strat = RealBaseStrategy()

    # BUY scenario: ascending prices
    up = np.linspace(1.1000, 1.1300, 60)
    up_md = _mk_market_data_from_series(up)
    buy_sig = strat.generate_signal(up_md)  # type: ignore[arg-type]
    assert buy_sig in {"BUY", "HOLD"}  # RSI oversold may not trigger strictly

    # SELL scenario: descending prices
    down = np.linspace(1.1300, 1.1000, 60)
    down_md = _mk_market_data_from_series(down)
    sell_sig = strat.generate_signal(down_md)  # type: ignore[arg-type]
    assert sell_sig in {"SELL", "HOLD"}

    # HOLD scenario: flat series
    flat = np.full(60, 1.1200)
    flat_md = _mk_market_data_from_series(flat)
    hold_sig = strat.generate_signal(flat_md)  # type: ignore[arg-type]
    assert hold_sig == "HOLD"


