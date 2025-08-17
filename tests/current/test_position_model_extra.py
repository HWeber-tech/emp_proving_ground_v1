import time
from datetime import datetime, timedelta

from src.trading.models.position import Position


def test_position_init_aliases_and_derived():
    p = Position(symbol="EURUSD", size=1000, entry_price=1.2000)
    # Canonical fields resolved
    assert p.symbol == "EURUSD"
    assert p.quantity == 1000.0
    assert p.average_price == 1.2000
    # current_price defaults to entry/average when not provided
    assert p.current_price == 1.2000
    # Derived
    assert p.value == 1000.0 * 1.2000
    assert p.unrealized_pnl == 0.0
    assert isinstance(p.last_updated, datetime)


def test_position_update_price_recomputes_pnl_and_timestamp():
    p = Position(symbol="EURUSD", size=1000, entry_price=1.2000)
    before_ts = p.last_updated
    assert before_ts is not None
    time.sleep(0.001)  # ensure monotonic increase in timestamp
    p.update_price(1.2500)
    assert p.current_price == 1.2500
    # (1.25 - 1.20) * 1000
    assert p.unrealized_pnl == 50.0
    assert isinstance(p.last_updated, datetime)
    after_ts = p.last_updated
    assert after_ts is not None
    assert after_ts >= before_ts


def test_position_update_quantity_and_average_price():
    p = Position(symbol="EURUSD", size=1000, entry_price=1.2000)
    p.update_quantity(2000, new_average_price=1.2200)
    assert p.quantity == 2000.0
    assert p.average_price == 1.2200
    # current_price still equals average price unless explicitly changed
    assert p.current_price == 1.2200
    # Unrealized PnL with equal current and entry is zero
    assert p.unrealized_pnl == 0.0


def test_position_is_long_short_flat_flags():
    long_p = Position(symbol="EURUSD", size=10, entry_price=1.0)
    short_p = Position(symbol="EURUSD", size=-5, entry_price=1.0)
    flat_p = Position(symbol="EURUSD", size=0, entry_price=1.0)

    assert long_p.is_long is True
    assert long_p.is_short is False
    assert long_p.is_flat is False

    assert short_p.is_long is False
    assert short_p.is_short is True
    assert short_p.is_flat is False

    assert flat_p.is_long is False
    assert flat_p.is_short is False
    assert flat_p.is_flat is True


def test_position_total_pnl_and_market_value_progression():
    p = Position(symbol="EURUSD", size=100, entry_price=1.0000)
    # Move price up: unrealized pnl positive
    p.update_market_price(1.0100)
    assert p.market_value == 100 * 1.0100
    assert p.total_pnl == p.realized_pnl + p.unrealized_pnl

    # Realize some pnl and ensure timestamp updates
    before = p.last_updated
    assert before is not None
    time.sleep(0.001)
    p.add_realized_pnl(5.5)
    assert p.realized_pnl == 5.5
    after = p.last_updated
    assert after is not None
    assert after >= before