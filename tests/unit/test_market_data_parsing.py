from datetime import datetime


def test_snapshot_parsing_groups_build_orderbook():
    from src.operational.icmarkets_api import GenuineFIXManager, OrderBook

    mgr = GenuineFIXManager(config=type('C', (), {'validate_config': lambda self=None: None})())

    # Fabricate a parsed snapshot with pairs including 268 groups
    msg = {
        '35': 'W',
        '55': 'EURUSD',
        '268': '3',
        '__pairs__': [
            ('35', 'W'), ('55', 'EURUSD'), ('268', '3'),
            ('269', '0'), ('270', '1.1000'), ('271', '1000000'),
            ('269', '1'), ('270', '1.1002'), ('271', '1200000'),
            ('269', '2'), ('270', '1.1001'), ('271', '250000'),
        ]
    }

    # Create a minimal connection replacement just to route handler
    mgr._handle_market_data_snapshot(msg)

    ob = mgr.order_books.get('EURUSD')
    assert ob is not None
    assert len(ob.bids) == 1
    assert len(ob.asks) == 1
    assert ob.last_trade is not None


def test_incremental_refresh_updates_by_action_and_side():
    from src.operational.icmarkets_api import GenuineFIXManager, OrderBook, MarketDataEntry

    mgr = GenuineFIXManager(config=type('C', (), {'validate_config': lambda self=None: None})())
    # Seed a book
    mgr.order_books['EURUSD'] = OrderBook(symbol='EURUSD')

    # New bid and ask
    inc_msg = {
        '35': 'X',
        '55': 'EURUSD',
        '268': '2',
        '__pairs__': [
            ('35', 'X'), ('55', 'EURUSD'), ('268', '2'),
            ('279', '0'), ('269', '0'), ('270', '1.1000'), ('271', '1000000'), ('278', 'bid1'),
            ('279', '0'), ('269', '1'), ('270', '1.1002'), ('271', '1200000'), ('278', 'ask1'),
        ]
    }
    mgr._handle_market_data_incremental_refresh(inc_msg)
    ob = mgr.order_books['EURUSD']
    assert any(e.entry_id == 'bid1' for e in ob.bids)
    assert any(e.entry_id == 'ask1' for e in ob.asks)

    # Change bid size
    inc_msg2 = {
        '35': 'X',
        '55': 'EURUSD',
        '268': '1',
        '__pairs__': [
            ('35', 'X'), ('55', 'EURUSD'), ('268', '1'),
            ('279', '1'), ('269', '0'), ('270', '1.1000'), ('271', '900000'), ('278', 'bid1'),
        ]
    }
    mgr._handle_market_data_incremental_refresh(inc_msg2)
    ob = mgr.order_books['EURUSD']
    assert any(e.entry_id == 'bid1' and e.size == 900000 for e in ob.bids)

    # Delete ask
    inc_msg3 = {
        '35': 'X',
        '55': 'EURUSD',
        '268': '1',
        '__pairs__': [
            ('35', 'X'), ('55', 'EURUSD'), ('268', '1'),
            ('279', '2'), ('269', '1'), ('270', '1.1002'), ('278', 'ask1'),
        ]
    }
    mgr._handle_market_data_incremental_refresh(inc_msg3)
    ob = mgr.order_books['EURUSD']
    assert not any(e.entry_id == 'ask1' for e in ob.asks)

