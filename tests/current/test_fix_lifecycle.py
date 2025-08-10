def test_fix_only_protocol_enforced():
    from src.governance.system_config import SystemConfig
    cfg = SystemConfig()
    assert cfg.connection_protocol == "fix"
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


def _setup_mgr():
    os.environ["EMP_USE_MOCK_FIX"] = "1"
    from src.operational.fix_connection_manager import FIXConnectionManager

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = True

    mgr = FIXConnectionManager(Cfg())
    assert mgr.start_sessions()
    trade_q = asyncio.Queue()
    mgr.get_application("trade").set_message_queue(trade_q)
    initiator = mgr.get_initiator("trade")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return mgr, initiator, trade_q, loop


def test_fix_reject_and_cancel():
    mgr, initiator, trade_q, loop = _setup_mgr()
    # Reject
    class Reject:
        cl_ord_id = "R1"
        reject = True
    assert initiator.send_message(Reject())
    rej_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    assert any(m.get(35) == b"8" and m.get(150) == b"8" for m in rej_msgs)

    # Cancel
    class Cancel:
        cl_ord_id = "C1"
        cancel = True
    assert initiator.send_message(Cancel())
    can_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    assert any(m.get(35) == b"8" and m.get(150) == b"4" for m in can_msgs)
    # Partial should also appear due to mock
    class Normal:
        cl_ord_id = "N1"
    assert initiator.send_message(Normal())
    norm_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    assert any(m.get(35) == b"8" and m.get(150) == b"1" for m in norm_msgs)
    loop.close()


