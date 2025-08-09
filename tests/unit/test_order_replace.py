def test_replace_builds_message_and_waits(monkeypatch):
    from src.operational.icmarkets_api import GenuineFIXManager, OrderInfo, OrderStatus

    cfg = type('C', (), {'validate_config': lambda self=None: None, 'account_number': '0000000'})()
    mgr = GenuineFIXManager(cfg)

    # Fake connection
    class DummyConn:
        def __init__(self):
            self.sent = []
        def is_connected(self):
            return True
        def send_message_and_track(self, msg, req_id=None):
            self.sent.append((msg, req_id))
            return True
    mgr.trade_connection = DummyConn()

    # Seed a limit order
    cl_id = 'ORDER_1'
    mgr.orders[cl_id] = OrderInfo(
        cl_ord_id=cl_id,
        symbol='EURUSD',
        side='1',
        order_qty=1000,
        ord_type='2',
        status=OrderStatus.NEW,
    )

    # Trigger replace; should send one message and return after timeout path (no ER updates in test)
    ok = mgr.replace_order_price(orig_cl_ord_id=cl_id, new_price=1.2345, wait_timeout=0.1)
    assert ok is False  # no confirmation path
    assert len(mgr.trade_connection.sent) == 1


