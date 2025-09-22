import asyncio
import os

from src.governance.system_config import ConnectionProtocol, SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager
from src.operational.mock_fix import MockExecutionStep


def test_fix_only_protocol_enforced():
    cfg = SystemConfig()
    assert cfg.connection_protocol == ConnectionProtocol.bootstrap
    cfg_fix = cfg.with_updated(connection_protocol="fix")
    assert cfg_fix.connection_protocol is ConnectionProtocol.fix


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
    # Set to '1' to force the internal simulator when credentials are unavailable
    os.environ.setdefault("EMP_USE_MOCK_FIX", "0")
    # moved to top-level to satisfy E402; kept import semantics

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = os.environ.get("EMP_USE_MOCK_FIX") == "1"

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
        mock_reject_reason = "99"
        mock_text = "Rejected for lifecycle"
        mock_trade_date = "20240102"

    assert initiator.send_message(Reject())
    rej_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    assert any(m.get(35) == b"8" and m.get(150) == b"8" for m in rej_msgs)
    assert any(m.get(39) == b"8" for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(37) for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(17) for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(58) == b"Rejected for lifecycle" for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(103) == b"99" for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(60) for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(52) for m in rej_msgs if m.get(35) == b"8")
    assert any(m.get(75) == b"20240102" for m in rej_msgs if m.get(35) == b"8")

    # Cancel
    class Cancel:
        cl_ord_id = "C1"
        cancel = True
        quantity = 4.0
        mock_cancel_reason = "0"
        mock_cancel_text = "Canceled for lifecycle"
        mock_transact_time = "20240102-00:00:00.000"
        mock_sending_time = "20240102-00:00:01.000"
        mock_trade_date = "20240103"

    assert initiator.send_message(Cancel())
    can_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    assert any(m.get(35) == b"8" and m.get(150) == b"4" for m in can_msgs)
    assert any(m.get(39) == b"4" for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(37) for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(17) for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(58) == b"Canceled for lifecycle" for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(60) == b"20240102-00:00:00.000" for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(52) == b"20240102-00:00:01.000" for m in can_msgs if m.get(35) == b"8")
    assert any(m.get(75) == b"20240103" for m in can_msgs if m.get(35) == b"8")

    # Partial should also appear due to mock
    class Normal:
        cl_ord_id = "N1"
        quantity = 10.0
        price = 1.25
        account = "TRADER1"
        ord_type = "2"
        time_in_force = "1"
        trade_date = "20240104"
        order_capacity = "A"
        customer_or_firm = "1"

    assert initiator.send_message(Normal())
    norm_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    partial_msgs = [m for m in norm_msgs if m.get(35) == b"8" and m.get(150) == b"1"]
    assert partial_msgs
    assert any(m.get(39) == b"1" for m in partial_msgs)
    assert any(m.get(32) == b"5" and m.get(151) == b"5" for m in partial_msgs)
    assert all(m.get(37) for m in partial_msgs)
    assert all(m.get(17) for m in partial_msgs)
    assert any(m.get(1) == b"TRADER1" for m in partial_msgs)
    assert any(m.get(40) == b"2" for m in partial_msgs)
    assert any(m.get(59) == b"1" for m in partial_msgs)
    assert any(m.get(75) == b"20240104" for m in partial_msgs)
    assert any(m.get(528) == b"A" for m in partial_msgs)
    assert any(m.get(204) == b"1" for m in partial_msgs)

    fill_msgs = [m for m in norm_msgs if m.get(35) == b"8" and m.get(150) == b"F"]
    assert fill_msgs
    assert any(m.get(1) == b"TRADER1" for m in fill_msgs)
    assert any(m.get(40) == b"2" for m in fill_msgs)
    assert any(m.get(59) == b"1" for m in fill_msgs)
    assert any(m.get(75) == b"20240104" for m in fill_msgs)
    assert any(m.get(528) == b"A" for m in fill_msgs)
    assert any(m.get(204) == b"1" for m in fill_msgs)

    class RatioPlan:
        cl_ord_id = "N2"
        quantity = 8.0
        price = 1.1
        mock_execution_plan = [
            {"exec_type": "0", "delay": 0.0},
            {"exec_type": "1", "remaining_ratio": 0.5, "delay": 0.0},
            {"exec_type": "1", "ratio": 0.25, "delay": 0.0},
            {"exec_type": "F", "delay": 0.0},
        ]

    assert initiator.send_message(RatioPlan())
    ratio_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    ratio_partials = [m for m in ratio_msgs if m.get(35) == b"8" and m.get(150) == b"1"]
    assert ratio_partials
    ratio_partial_last_qtys = {m.get(32) for m in ratio_partials}
    assert b"4" in ratio_partial_last_qtys
    assert b"2" in ratio_partial_last_qtys
    ratio_fills = [m for m in ratio_msgs if m.get(35) == b"8" and m.get(150) == b"F"]
    assert ratio_fills
    assert any(m.get(32) == b"2" for m in ratio_fills)
    assert any(m.get(14) == b"8" for m in ratio_fills)
    assert any(m.get(151) == b"0" for m in ratio_fills)

    class MetadataPlan:
        cl_ord_id = "N3"
        quantity = 6.0
        price = 1.15
        mock_execution_plan = [
            MockExecutionStep(
                "1",
                quantity=3.0,
                account="PLAN-ACC",
                order_type="3",
                time_in_force="2",
                delay=0.0,
            ),
            MockExecutionStep(
                "F",
                quantity=3.0,
                account="PLAN-FINAL",
                order_type="4",
                time_in_force="5",
                delay=0.0,
            ),
        ]

    assert initiator.send_message(MetadataPlan())
    metadata_msgs = loop.run_until_complete(_drain(trade_q, timeout=0.5))
    metadata_partials = [m for m in metadata_msgs if m.get(35) == b"8" and m.get(150) == b"1"]
    assert metadata_partials
    assert any(m.get(1) == b"PLAN-ACC" for m in metadata_partials)
    assert any(m.get(40) == b"3" for m in metadata_partials)
    assert any(m.get(59) == b"2" for m in metadata_partials)

    metadata_fills = [m for m in metadata_msgs if m.get(35) == b"8" and m.get(150) == b"F"]
    assert metadata_fills
    assert any(m.get(1) == b"PLAN-FINAL" for m in metadata_fills)
    assert any(m.get(40) == b"4" for m in metadata_fills)
    assert any(m.get(59) == b"5" for m in metadata_fills)
    loop.close()
