import pytest


def test_replace_requires_limit_and_replaceable_state():
    from src.operational.icmarkets_api import GenuineFIXManager, OrderInfo

    cfg = type('C', (), {'validate_config': lambda self=None: None, 'account_number': '0'})()
    m = GenuineFIXManager(cfg)

    cl_id = 'ORDER_X'
    # Non-limit order
    m.orders[cl_id] = OrderInfo(cl_ord_id=cl_id, ord_type='1')
    assert m.replace_order_price(cl_id, new_price=1.2345, wait_timeout=0.01) is False

    # Limit but canceled
    m.orders[cl_id] = OrderInfo(cl_ord_id=cl_id, ord_type='2', side='1')
    # Simulate non-replaceable by clearing status default
    m.orders[cl_id].status = m.orders[cl_id].status.CANCELED
    assert m.replace_order_price(cl_id, new_price=1.2345, wait_timeout=0.01) is False


