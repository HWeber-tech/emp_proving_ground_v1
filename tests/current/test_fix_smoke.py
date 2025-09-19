import asyncio
import os


async def _drain(q: asyncio.Queue, timeout=1.0):
    msgs = []
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=0.05)
            msgs.append(msg)
        except Exception:
            await asyncio.sleep(0.01)
    return msgs


def test_fix_mock_roundtrip(monkeypatch):
    # Prefer real FIX; set to '1' if you explicitly want the simulator
    os.environ.setdefault("EMP_USE_MOCK_FIX", "0")
    from src.operational.fix_connection_manager import FIXConnectionManager

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = os.environ.get("EMP_USE_MOCK_FIX") == "1"

    mgr = FIXConnectionManager(Cfg())
    assert mgr.start_sessions()

    price_q = asyncio.Queue()
    trade_q = asyncio.Queue()

    mgr.get_application("price").set_message_queue(price_q)
    mgr.get_application("trade").set_message_queue(trade_q)

    # Send a dummy order
    initiator = mgr.get_initiator("trade")

    class Order:
        cl_ord_id = "ABC123"
        quantity = 3.0
        price = 1.5
        account = "SMOKE"
        order_type = "2"
        time_in_force = "1"
        settle_type = "0"
        settle_date = "20240105"
        trade_date = "20240104"
        order_capacity = "A"
        customer_or_firm = "1"

    assert initiator.send_message(Order())

    # Drain a bit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    price_msgs = loop.run_until_complete(_drain(price_q, timeout=1.0))
    trade_msgs = loop.run_until_complete(_drain(trade_q, timeout=1.0))
    loop.close()

    assert any(m.get(35) == b"W" for m in price_msgs)
    exec_types = {m.get(150) for m in trade_msgs if m.get(35) == b"8"}
    assert b"0" in exec_types and b"F" in exec_types
    assert any(m.get(37) for m in trade_msgs if m.get(35) == b"8")
    assert any(m.get(17) for m in trade_msgs if m.get(35) == b"8")
    fills = [m for m in trade_msgs if m.get(150) == b"F"]
    assert fills
    assert any(m.get(39) == b"2" for m in fills)
    assert any(m.get(32) == b"1.5" for m in fills)
    assert any(m.get(14) == b"3" for m in fills)
    assert any(m.get(151) == b"0" for m in fills)
    assert any(m.get(31) == b"1.5" for m in fills)
    assert all(m.get(37) for m in fills)
    assert all(m.get(17) for m in fills)
    assert any(m.get(6) == b"1.5" for m in fills)
    assert any(m.get(1) == b"SMOKE" for m in fills)
    assert any(m.get(40) == b"2" for m in fills)
    assert any(m.get(59) == b"1" for m in fills)
    assert any(m.get(63) == b"0" for m in fills)
    assert any(m.get(64) == b"20240105" for m in fills)
    assert any(m.get(75) == b"20240104" for m in fills)
    assert any(m.get(528) == b"A" for m in fills)
    assert any(m.get(204) == b"1" for m in fills)



def test_fix_mock_uses_config_defaults(monkeypatch):
    monkeypatch.setenv("EMP_USE_MOCK_FIX", "1")
    from src.operational.fix_connection_manager import FIXConnectionManager

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = True
        default_account = "CFG-ACCOUNT"
        default_order_type = "1"
        default_time_in_force = "3"
        default_commission = 0.42
        default_commission_type = "3"
        default_commission_currency = "USD"
        default_settle_type = "4"
        default_settle_date = "20240110"
        default_trade_date = "20240112"
        default_order_capacity = "W"
        default_customer_or_firm = "0"

    mgr = FIXConnectionManager(Cfg())
    assert mgr.start_sessions()

    trade_q = asyncio.Queue()
    mgr.get_application("trade").set_message_queue(trade_q)

    initiator = mgr.get_initiator("trade")

    class BareOrder:
        cl_ord_id = "CFG-DEFAULT"
        quantity = 2.0
        price = 1.1

    assert initiator.send_message(BareOrder())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        trade_msgs = loop.run_until_complete(_drain(trade_q, timeout=1.0))
    finally:
        loop.close()

    fills = [m for m in trade_msgs if m.get(35) == b"8" and m.get(150) == b"F"]
    assert fills
    assert any(m.get(1) == b"CFG-ACCOUNT" for m in fills)
    assert any(m.get(40) == b"1" for m in fills)
    assert any(m.get(59) == b"3" for m in fills)
    assert any(m.get(12) == b"0.84" for m in fills)
    assert any(m.get(13) == b"3" for m in fills)
    assert any(m.get(15) == b"USD" for m in fills)
    assert any(m.get(63) == b"4" for m in fills)
    assert any(m.get(64) == b"20240110" for m in fills)
    assert any(m.get(75) == b"20240112" for m in fills)
    assert any(m.get(528) == b"W" for m in fills)
    assert any(m.get(204) == b"0" for m in fills)

    mgr.stop_sessions()
