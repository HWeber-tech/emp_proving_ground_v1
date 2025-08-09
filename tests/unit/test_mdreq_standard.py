import simplefix


def test_mdreq_standard_uses_numeric_55_and_267():
    from src.operational.icmarkets_api import GenuineFIXManager

    m = GenuineFIXManager(config=type('C', (), {'validate_config': lambda self=None: None, 'account_number': '000',})())

    class DummyConn:
        def __init__(self):
            self.last = None
        def is_connected(self):
            return True
        def send_message_and_track(self, msg, req_id=None):
            self.last = msg
            return True

    m.price_connection = DummyConn()
    ok = m._send_md_request("EURUSD", "REQ1")
    assert ok
    raw = m.price_connection.last.encode().decode('utf-8', errors='ignore')
    assert "35=V" in raw
    assert "146=1" in raw
    assert "267=2" in raw
    assert "269=0" in raw and raw.count("269=") >= 2
    # Must use 55, not 48
    assert "\x0148=" not in raw
    assert "\x0155=" in raw

