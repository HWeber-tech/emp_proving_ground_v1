import pytest

from src.trading.models.position import Position


def test_trading_position_smoke():
    # Construct via canonical module with default current_price = entry_price
    p = Position(symbol="EURUSD", size=1000, entry_price=1.1000)
    assert p.current_price == pytest.approx(p.entry_price)

    # Move up
    p.update_price(1.1020)
    assert p.value == pytest.approx(1000 * 1.1020)
    assert p.unrealized_pnl == pytest.approx((1.1020 - 1.1000) * 1000)

    # Move down (negative PnL)
    p.update_price(1.0980)
    assert p.value == pytest.approx(1000 * 1.0980)
    assert p.unrealized_pnl == pytest.approx((1.0980 - 1.1000) * 1000)

    # Construct a second instance to confirm class-level state stays isolated
    q = Position(symbol="EURUSD", size=1000, entry_price=1.1000)
    assert q.current_price == pytest.approx(1.1000)
    q.update_price(1.1020)
    assert q.value == pytest.approx(1000 * 1.1020)
    assert q.unrealized_pnl == pytest.approx((1.1020 - 1.1000) * 1000)
