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
    os.environ["EMP_USE_MOCK_FIX"] = "1"
    from src.operational.fix_connection_manager import FIXConnectionManager

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = True

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
    assert initiator.send_message(Order())

    # Drain a bit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    price_msgs = loop.run_until_complete(_drain(price_q, timeout=0.5))
    trade_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    loop.close()

    assert any(m.get(35) == b"W" for m in price_msgs)
    assert any(m.get(35) == b"8" for m in trade_msgs)


