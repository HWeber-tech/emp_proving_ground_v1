from datetime import datetime

from src.trading.models.trade import Trade


def test_trade_post_init_parses_iso_timestamp_string():
    t = Trade(
        trade_id="t1",
        order_id="o1",
        symbol="EURUSD",
        side="BUY",
        quantity=100.0,
        price=1.2345,
        timestamp="2024-01-02T03:04:05",  # type: ignore[arg-type] — exercising __post_init__ string parsing
        commission=0.25,
        exchange="ECN",
    )
    # __post_init__ should parse ISO string to datetime
    assert isinstance(t.timestamp, datetime)
    # Value and net_value calculations
    assert t.value == 100.0 * 1.2345
    assert t.net_value == t.value - 0.25


def test_trade_value_net_value_with_datetime_timestamp():
    ts = datetime(2024, 1, 2, 3, 4, 5)
    t = Trade(
        trade_id="t2",
        order_id="o2",
        symbol="EURUSD",
        side="SELL",
        quantity=50.0,
        price=2.0,
        timestamp=ts,
        commission=1.0,
    )
    assert t.timestamp is ts
    assert t.value == 100.0
    assert t.net_value == 99.0


def test_trade_net_value_default_commission_zero():
    ts = datetime(2024, 1, 2, 3, 4, 5)
    t = Trade(
        trade_id="t3",
        order_id="o3",
        symbol="EURUSD",
        side="BUY",
        quantity=10.0,
        price=3.0,
        timestamp=ts,
    )
    assert t.value == 30.0
    assert t.net_value == 30.0