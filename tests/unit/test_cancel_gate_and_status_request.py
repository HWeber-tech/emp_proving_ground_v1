def test_cancel_gate_skips_non_cancelable_and_sends_for_pending():
    from src.operational.icmarkets_api import GenuineFIXManager, OrderInfo, OrderStatus

    cfg = type('C', (), {'validate_config': lambda self=None: None, 'account_number': '0'})()
    m = GenuineFIXManager(cfg)

    # Stub connection
    sent = []
    class DummyConn:
        def is_connected(self):
            return True
        def send_message_and_track(self, msg, req_id=None):
            sent.append((msg, req_id))
            return True
    m.trade_connection = DummyConn()

    # Non-cancelable (filled)
    cl_id_f = 'ORDER_F'
    m.orders[cl_id_f] = OrderInfo(cl_ord_id=cl_id_f, status=OrderStatus.FILLED)
    assert m.cancel_order_minimal(cl_id_f, wait_timeout=0.01) is False

    # Pending new -> should attempt send
    cl_id_p = 'ORDER_P'
    m.orders[cl_id_p] = OrderInfo(cl_ord_id=cl_id_p, status=OrderStatus.PENDING_NEW)
    _ = m.cancel_order_minimal(cl_id_p, wait_timeout=0.01)
    assert any(req for (_msg, req) in sent if isinstance(req, str) and req.startswith('CNCL_'))


