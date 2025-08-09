from src.trading.monitoring.portfolio_tracker import PortfolioTracker


class DummyOrder:
    def __init__(self, symbol, side, last_qty, last_px, exec_type):
        self.symbol = symbol
        self.side = side
        self.last_qty = last_qty
        self.last_px = last_px
        self.executions = [{"exec_type": exec_type}]


def test_portfolio_tracker_buy_then_sell_updates_pnl(tmp_path, monkeypatch):
    # Force JSON store path by setting env within object by patching base_dir via monkeypatch if needed
    pt = PortfolioTracker()

    # Start flat
    assert pt.positions.get("EURUSD") is None

    # Buy 1000 @ 1.1000
    buy = DummyOrder("EURUSD", "1", 1000, 1.1000, "2")
    pt._handle_order_info(buy)
    pos = pt.positions["EURUSD"]
    assert pos.quantity == 1000
    assert round(pos.avg_price, 5) == 1.1000

    # Sell 600 @ 1.1005 (realized profit = 0.0005 * 600)
    sell = DummyOrder("EURUSD", "2", 600, 1.1005, "2")
    pt._handle_order_info(sell)
    pos = pt.positions["EURUSD"]
    assert pos.quantity == 400
    assert round(pos.realized_pnl, 5) == round(0.0005 * 600, 5)


