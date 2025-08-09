from time import time


def test_reconnect_backoff_progression(monkeypatch):
    from src.operational.icmarkets_api import GenuineFIXManager

    # Dummy config
    cfg = type('C', (), {'validate_config': lambda self=None: None})()
    mgr = GenuineFIXManager(cfg)
    
    # Inject dummy connections with connect() outcome
    class DummyConn:
        def __init__(self):
            self._ok = False
        def is_connected(self):
            return False
        def connect(self):
            return self._ok
    mgr.price_connection = DummyConn()
    mgr.trade_connection = DummyConn()

    # Start running to enable supervisor step
    mgr.running = True

    # Force attempts and check backoff grows when connect() fails
    s_quote = mgr._reconnect_state['quote']
    s_trade = mgr._reconnect_state['trade']
    s_quote['delay'] = 1.0
    s_trade['delay'] = 1.0
    s_quote['next'] = 0.0
    s_trade['next'] = 0.0

    mgr._supervisor_step()
    assert s_quote['delay'] >= 2.0 or s_quote['next'] > 0.0
    assert s_trade['delay'] >= 2.0 or s_trade['next'] > 0.0

    # Now let trade reconnect succeed, quote keep failing
    mgr.trade_connection._ok = True
    now_before = time()
    mgr._supervisor_step()
    # Trade should reset backoff
    assert mgr._reconnect_state['trade']['delay'] == 1.0
    assert mgr._reconnect_state['trade']['next'] == 0.0

    # Quote continues backing off
    assert mgr._reconnect_state['quote']['delay'] >= 2.0


