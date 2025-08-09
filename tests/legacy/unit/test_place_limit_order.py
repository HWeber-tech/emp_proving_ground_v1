def test_place_limit_order_builds_and_tracks(monkeypatch):
    from src.operational.icmarkets_api import GenuineFIXManager

    cfg = type('C', (), {'validate_config': lambda self=None: None, 'account_number': '0'})()
    m = GenuineFIXManager(cfg)

    class DummyConn:
        def is_connected(self):
            return True
        def send_message_and_track(self, msg, req_id=None):
            return True

    m.trade_connection = DummyConn()
    order = m.place_limit_order_genuine("EURUSD", "BUY", 1500, 1.23456, tif="0", timeout=0.05)
    assert order is not None

