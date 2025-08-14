import pytest

from src.trading.models.position import Position as PositionFromModule
from src.trading.models import Position as PositionFromFacade


def test_trading_position_smoke():
    # Verify re-exports resolve to the same class object
    assert PositionFromModule is PositionFromFacade

    # Construct via module import with default current_price = entry_price
    p = PositionFromModule(symbol="EURUSD", size=1000, entry_price=1.1000)
    assert p.current_price == pytest.approx(p.entry_price)

    # Move up
    p.update_price(1.1020)
    assert p.value == pytest.approx(1000 * 1.1020)
    assert p.unrealized_pnl == pytest.approx((1.1020 - 1.1000) * 1000)

    # Move down (negative PnL)
    p.update_price(1.0980)
    assert p.value == pytest.approx(1000 * 1.0980)
    assert p.unrealized_pnl == pytest.approx((1.0980 - 1.1000) * 1000)

    # Construct via facade import and verify semantics again
    q = PositionFromFacade(symbol="EURUSD", size=1000, entry_price=1.1000)
    assert q.current_price == pytest.approx(1.1000)
    q.update_price(1.1020)
    assert q.value == pytest.approx(1000 * 1.1020)
    assert q.unrealized_pnl == pytest.approx((1.1020 - 1.1000) * 1000)