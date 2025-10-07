import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from src.operational.fix_connection_manager import _FIXApplicationAdapter
from src.operational.mock_fix import (
    MockExecutionStep,
    MockFIXManager,
    MockMarketDataStep,
    MockOrderBookLevel,
    MockOrderInfo,
    _coerce_optional_bool,
    _coerce_optional_fix_date,
    _coerce_optional_float,
    _coerce_optional_str,
    _coerce_order_book_level,
)


def test_mock_fix_coercers_harden_against_invalid_bytes():
    assert _coerce_optional_float(b"\xff", default=3.14) == pytest.approx(3.14)
    assert _coerce_optional_str(b"\xff", default="fallback") == "fallback"
    assert _coerce_optional_bool(b"\xff") is None
    assert _coerce_optional_fix_date(float("nan")) is None


def test_mock_fix_order_book_coercion_logs_on_failure(caplog):
    class FaultyLevel:
        @property
        def price(self):
            raise RuntimeError("boom")

        @property
        def size(self):
            return 1.0

    caplog.set_level(logging.DEBUG, logger="src.operational.mock_fix")
    assert _coerce_order_book_level(FaultyLevel()) is None
    assert any("Failed to coerce order book" in message for message in caplog.messages)


def test_mock_fix_emits_enriched_order_info_and_telemetry():
    manager = MockFIXManager(
        symbol="GBPUSD",
        market_data_interval=0.01,
        market_data_duration=0.05,
        partial_fill_ratio=0.5,
    )
    received: list = []
    manager.add_order_callback(received.append)

    try:
        assert manager.start()
        assert manager.wait_for_telemetry("market_data_snapshot", timeout=1.0)
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD1", quantity=4.0, price=1.2345)
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("exec_type") == "F",
            timeout=1.0,
        )
    finally:
        manager.stop()

    telemetry = manager.snapshot_telemetry()
    assert any(event.event == "market_data_snapshot" for event in telemetry)
    assert any(
        event.event == "order_execution" and event.details.get("exec_type") == "F"
        for event in telemetry
    )

    partials = [
        info for info in received if info.executions and info.executions[-1].get("exec_type") == "1"
    ]
    assert partials
    partial = partials[0]
    assert partial.last_qty == pytest.approx(2.0)
    assert partial.cum_qty == pytest.approx(2.0)
    assert partial.leaves_qty == pytest.approx(2.0)

    fills = [
        info for info in received if info.executions and info.executions[-1].get("exec_type") == "F"
    ]
    assert fills
    fill = fills[0]
    assert fill.last_qty == pytest.approx(2.0)
    assert fill.cum_qty == pytest.approx(4.0)
    assert fill.ord_status == "2"
    assert fill.leaves_qty == 0.0
    assert fill.order_id
    assert fill.exec_id
    assert len(fill.executions) >= 3
    assert [exec.get("exec_type") for exec in fill.executions][:3] == [
        "0",
        "1",
        "F",
    ]

    fill_event = next(
        event
        for event in telemetry
        if event.event == "order_execution" and event.details.get("exec_type") == "F"
    )
    assert fill_event.details.get("last_qty") == pytest.approx(2.0)
    assert fill_event.details.get("cum_qty") == pytest.approx(4.0)
    assert fill_event.details.get("leaves_qty") == pytest.approx(0.0)
    assert fill_event.details.get("order_id") == fill.order_id
    assert fill_event.details.get("exec_id") == fill.exec_id

    sequences = [
        event.details.get("sequence")
        for event in telemetry
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD1"
    ]
    assert sequences[:3] == [1, 2, 3]


def test_mock_fix_records_order_history_snapshots():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        partial_fill_ratio=0.5,
    )

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-HISTORY", quantity=4.0, price=2.5)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    history = manager.get_order_history("ORD-HISTORY")
    assert [record["exec_type"] for record in history] == ["0", "1", "F"]
    assert history[-1]["cum_qty"] == pytest.approx(4.0)
    assert history[-1]["leaves_qty"] == pytest.approx(0.0)
    assert history[-1]["avg_px"] == pytest.approx(2.5)

    info = manager.get_last_order_info("ORD-HISTORY")
    assert info is not None
    assert info.order_id
    assert [record.get("exec_type") for record in info.executions] == [
        "0",
        "1",
        "F",
    ]
    assert info.cum_qty == pytest.approx(4.0)
    assert info.leaves_qty == pytest.approx(0.0)

    # Returned snapshots should be isolated copies
    history[0]["exec_type"] = "Z"
    info.executions[0]["exec_type"] = "Z"  # type: ignore[index]

    refreshed_history = manager.get_order_history("ORD-HISTORY")
    assert refreshed_history[0]["exec_type"] == "0"

    refreshed_info = manager.get_last_order_info("ORD-HISTORY")
    assert refreshed_info is not None
    assert refreshed_info.executions[0]["exec_type"] == "0"

    assert manager.get_order_history("UNKNOWN") == []
    assert manager.get_last_order_info("UNKNOWN") is None


def test_mock_fix_generates_order_and_exec_ids():
    manager = MockFIXManager(market_data_interval=0.01, market_data_duration=0.05)
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-IDS", quantity=4.0, price=1.0)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    order_updates = [info for info in updates if info.cl_ord_id == "ORD-IDS"]
    assert len(order_updates) >= 3
    order_ids = {info.order_id for info in order_updates}
    assert len(order_ids) == 1
    order_id = order_ids.pop()
    assert order_id.startswith("MOCK-ORD-")
    exec_ids = [info.exec_id for info in order_updates]
    assert exec_ids[:3] == [
        f"{order_id}-EXEC-001",
        f"{order_id}-EXEC-002",
        f"{order_id}-EXEC-003",
    ]

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-IDS"
    ]
    assert {event.details.get("order_id") for event in telemetry} == {order_id}
    assert [event.details.get("exec_id") for event in telemetry][:3] == exec_ids[:3]

    received_event = next(
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_received" and event.details.get("cl_ord_id") == "ORD-IDS"
    )
    assert received_event.details.get("order_id") == order_id


def test_mock_fix_allows_configurable_id_generation():
    manager = MockFIXManager(
        market_data_interval=0.0,
        market_data_duration=0.0,
        execution_interval=0.0,
        synchronous_order_flows=True,
    )

    manager.configure_id_generation(
        order_id_prefix="UNIT",
        order_id_start=42,
        order_id_padding=4,
        exec_id_start=7,
        exec_id_prefix="EXEC",
    )

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-1", quantity=4.0, price=1.25)
        )
        assert manager.wait_for_idle(timeout=1.0)

        manager.configure_id_generation(
            order_id_prefix="ALT",
            order_id_start=120,
            order_id_padding=3,
            exec_id_start=11,
            exec_id_prefix=None,
        )
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-2", quantity=3.0, price=1.5)
        )
        assert manager.wait_for_idle(timeout=1.0)

        manager.configure_id_generation(
            order_id_prefix=None,
            order_id_start=3,
            order_id_padding=2,
            exec_id_start=2,
        )
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-3", quantity=2.0, price=1.0)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    info_first = manager.get_last_order_info("ORD-1")
    assert info_first is not None
    assert info_first.order_id == "UNIT-0042"
    assert [record["exec_id"] for record in info_first.executions][:3] == [
        "EXEC-007",
        "EXEC-008",
        "EXEC-009",
    ]

    info_second = manager.get_last_order_info("ORD-2")
    assert info_second is not None
    assert info_second.order_id == "ALT-120"
    assert [record["exec_id"] for record in info_second.executions][:3] == [
        "ALT-120-EXEC-011",
        "ALT-120-EXEC-012",
        "ALT-120-EXEC-013",
    ]

    info_third = manager.get_last_order_info("ORD-3")
    assert info_third is not None
    assert info_third.order_id == "MOCK-ORD-03"
    assert [record["exec_id"] for record in info_third.executions][:3] == [
        "MOCK-ORD-03-EXEC-002",
        "MOCK-ORD-03-EXEC-003",
        "MOCK-ORD-03-EXEC-004",
    ]


def test_mock_fix_emits_timestamp_metadata():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        synchronous_order_flows=True,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-TIME",
                quantity=2.0,
                price=1.2,
                mock_transact_time="20240101-00:00:00.000",
                mock_sending_time="20240101-00:00:01.000",
                mock_execution_plan=[
                    {"exec_type": "0", "delay": 0.0},
                    {
                        "exec_type": "1",
                        "quantity": 1.0,
                        "delay": 0.0,
                        "transact_time": "20240101-00:00:02.000",
                    },
                    {
                        "exec_type": "F",
                        "delay": 0.0,
                        "sending_time": "20240101-00:00:03.000",
                    },
                ],
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-NOOVERRIDE", quantity=1.0, price=1.0)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    time_updates = [info for info in updates if info.cl_ord_id == "ORD-TIME"]
    assert time_updates
    ack = next(info for info in time_updates if info.executions[-1].get("exec_type") == "0")
    assert ack.transact_time == "20240101-00:00:00.000"
    assert ack.sending_time == "20240101-00:00:01.000"
    partial = next(info for info in time_updates if info.executions[-1].get("exec_type") == "1")
    assert partial.transact_time == "20240101-00:00:02.000"
    assert partial.sending_time == partial.transact_time
    fill = next(info for info in time_updates if info.executions[-1].get("exec_type") == "F")
    assert fill.transact_time == "20240101-00:00:00.000"
    assert fill.sending_time == "20240101-00:00:03.000"

    telemetry = manager.snapshot_telemetry()
    time_events = [
        event
        for event in telemetry
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-TIME"
    ]
    assert any(
        event.details.get("exec_type") == "1"
        and event.details.get("transact_time") == "20240101-00:00:02.000"
        for event in time_events
    )
    assert any(
        event.details.get("exec_type") == "F"
        and event.details.get("sending_time") == "20240101-00:00:03.000"
        for event in time_events
    )

    default_updates = [info for info in updates if info.cl_ord_id == "ORD-NOOVERRIDE"]
    assert default_updates
    assert all(isinstance(info.transact_time, str) for info in default_updates)
    assert all(info.transact_time for info in default_updates)
    assert all(isinstance(info.sending_time, str) for info in default_updates)
    assert all(info.sending_time for info in default_updates)


def test_mock_fix_applies_order_capacity_and_customer_flags():
    manager = MockFIXManager(
        market_data_interval=0.0,
        market_data_duration=0.0,
        synchronous_order_flows=True,
        default_order_capacity="A",
        default_customer_or_firm="1",
    )
    updates: list[MockOrderInfo] = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-CAP-DEF", quantity=2.0, price=1.1)
        )
        assert manager.wait_for_idle(timeout=1.0)

        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-CAP-OVR",
                quantity=1.0,
                price=1.5,
                mock_order_capacity="G",
                mock_customer_or_firm="0",
                mock_execution_plan=[
                    {"exec_type": "0", "delay": 0.0},
                    {"exec_type": "F", "delay": 0.0},
                ],
            )
        )
        assert manager.wait_for_idle(timeout=1.0)

        manager.configure_order_defaults(order_capacity="R", customer_or_firm="2")
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-CAP-CONF", quantity=1.0, price=2.0)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    default_info = manager.get_last_order_info("ORD-CAP-DEF")
    assert default_info is not None
    assert default_info.order_capacity == "A"
    assert default_info.customer_or_firm == "1"
    assert default_info.executions[-1].get("order_capacity") == "A"
    assert default_info.executions[-1].get("customer_or_firm") == "1"

    override_info = manager.get_last_order_info("ORD-CAP-OVR")
    assert override_info is not None
    assert override_info.order_capacity == "G"
    assert override_info.customer_or_firm == "0"
    assert override_info.executions[-1].get("order_capacity") == "G"
    assert override_info.executions[-1].get("customer_or_firm") == "0"

    configured_info = manager.get_last_order_info("ORD-CAP-CONF")
    assert configured_info is not None
    assert configured_info.order_capacity == "R"
    assert configured_info.customer_or_firm == "2"

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution"
        and event.details.get("cl_ord_id")
        in {
            "ORD-CAP-DEF",
            "ORD-CAP-OVR",
            "ORD-CAP-CONF",
        }
    ]
    assert telemetry
    assert any(
        event.details.get("order_capacity") == "A" and event.details.get("customer_or_firm") == "1"
        for event in telemetry
        if event.details.get("cl_ord_id") == "ORD-CAP-DEF"
    )
    assert any(
        event.details.get("order_capacity") == "G" and event.details.get("customer_or_firm") == "0"
        for event in telemetry
        if event.details.get("cl_ord_id") == "ORD-CAP-OVR"
    )
    assert any(
        event.details.get("order_capacity") == "R" and event.details.get("customer_or_firm") == "2"
        for event in telemetry
        if event.details.get("cl_ord_id") == "ORD-CAP-CONF"
    )

    history = manager.get_order_history("ORD-CAP-OVR")
    assert history
    assert history[-1].get("order_capacity") == "G"
    assert history[-1].get("customer_or_firm") == "0"


def test_mock_fix_tracks_order_metadata_fields():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        synchronous_order_flows=True,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-META",
                quantity=3.0,
                price=1.11,
                account="ACCT-123",
                order_type="2",
                time_in_force="3",
            )
        )
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-META-ALIAS",
                quantity=1.0,
                price=1.2,
                mock_account="ALIAS",
                mock_order_type="1",
                mock_time_in_force="0",
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    primary_updates = [info for info in updates if info.cl_ord_id == "ORD-META"]
    assert primary_updates
    assert {info.account for info in primary_updates} == {"ACCT-123"}
    assert {info.order_type for info in primary_updates} == {"2"}
    assert {info.time_in_force for info in primary_updates} == {"3"}

    alias_updates = [info for info in updates if info.cl_ord_id == "ORD-META-ALIAS"]
    assert alias_updates
    assert {info.account for info in alias_updates} == {"ALIAS"}
    assert {info.order_type for info in alias_updates} == {"1"}
    assert {info.time_in_force for info in alias_updates} == {"0"}

    history = manager.get_order_history("ORD-META")
    assert history
    assert history[0].get("account") == "ACCT-123"
    assert history[-1].get("order_type") == "2"
    assert history[-1].get("time_in_force") == "3"

    alias_history = manager.get_order_history("ORD-META-ALIAS")
    assert alias_history
    assert alias_history[0].get("account") == "ALIAS"
    assert alias_history[-1].get("order_type") == "1"
    assert alias_history[-1].get("time_in_force") == "0"

    telemetry = manager.snapshot_telemetry()
    received = next(
        event
        for event in telemetry
        if event.event == "order_received" and event.details.get("cl_ord_id") == "ORD-META"
    )
    assert received.details.get("account") == "ACCT-123"
    assert received.details.get("order_type") == "2"
    assert received.details.get("time_in_force") == "3"

    fill_event = next(
        event
        for event in telemetry
        if event.event == "order_execution"
        and event.details.get("cl_ord_id") == "ORD-META"
        and event.details.get("exec_type") == "F"
    )
    assert fill_event.details.get("account") == "ACCT-123"
    assert fill_event.details.get("order_type") == "2"
    assert fill_event.details.get("time_in_force") == "3"

    completion_event = next(
        event
        for event in telemetry
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "ORD-META"
    )
    assert completion_event.details.get("account") == "ACCT-123"
    assert completion_event.details.get("order_type") == "2"
    assert completion_event.details.get("time_in_force") == "3"


def test_mock_fix_applies_default_order_metadata():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        default_account="DEFAULT-ACC",
        default_order_type="2",
        default_time_in_force="1",
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-DEFAULT", quantity=4.0, price=1.5)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    fills = [info for info in updates if info.cl_ord_id == "ORD-DEFAULT" and info.ord_status == "2"]
    assert fills
    fill = fills[-1]
    assert fill.account == "DEFAULT-ACC"
    assert fill.order_type == "2"
    assert fill.time_in_force == "1"
    assert {exec.get("account") for exec in fill.executions if exec} == {"DEFAULT-ACC"}
    assert {exec.get("order_type") for exec in fill.executions if exec} == {"2"}
    assert {exec.get("time_in_force") for exec in fill.executions if exec} == {"1"}

    history = manager.get_order_history("ORD-DEFAULT")
    assert history
    assert history[-1].get("account") == "DEFAULT-ACC"
    assert history[-1].get("order_type") == "2"
    assert history[-1].get("time_in_force") == "1"

    telemetry = manager.snapshot_telemetry()
    exec_events = [
        event
        for event in telemetry
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-DEFAULT"
    ]
    assert exec_events
    assert exec_events[-1].details.get("account") == "DEFAULT-ACC"
    assert exec_events[-1].details.get("order_type") == "2"
    assert exec_events[-1].details.get("time_in_force") == "1"

    completion = next(
        event
        for event in telemetry
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "ORD-DEFAULT"
    )
    assert completion.details.get("account") == "DEFAULT-ACC"
    assert completion.details.get("order_type") == "2"
    assert completion.details.get("time_in_force") == "1"


def test_mock_fix_configures_order_defaults():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        manager.configure_order_defaults(account="PRIME", order_type="1", time_in_force="3")
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-PRIME", quantity=2.0, price=1.1)
        )
        assert manager.wait_for_idle(timeout=1.0)
        manager.configure_order_defaults(account="SECONDARY", time_in_force=None)
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-SECOND", quantity=1.0, price=1.2)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    prime_fill = next(
        info for info in updates if info.cl_ord_id == "ORD-PRIME" and info.ord_status == "2"
    )
    assert prime_fill.account == "PRIME"
    assert prime_fill.order_type == "1"
    assert prime_fill.time_in_force == "3"

    second_fill = next(
        info for info in updates if info.cl_ord_id == "ORD-SECOND" and info.ord_status == "2"
    )
    assert second_fill.account == "SECONDARY"
    assert second_fill.order_type == "1"
    assert second_fill.time_in_force is None


def test_mock_fix_execution_plan_overrides_order_metadata():
    manager = MockFIXManager(
        synchronous_order_flows=True,
        default_account="DEF-ACC",
        default_order_type="1",
        default_time_in_force="0",
    )
    updates: list[MockOrderInfo] = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep(
            "1",
            quantity=1.0,
            account="STEP-ACC",
            order_type="4",
            time_in_force="3",
        ),
        MockExecutionStep(
            "F",
            quantity=1.0,
            account="FINAL-ACC",
            order_type="5",
            time_in_force="6",
        ),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="META-PLAN",
                quantity=2.0,
                price=1.05,
                account="INITIAL-ACC",
                order_type="2",
                time_in_force="1",
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    partial = next(
        info for info in updates if info.cl_ord_id == "META-PLAN" and info.ord_status == "1"
    )
    assert partial.account == "STEP-ACC"
    assert partial.order_type == "4"
    assert partial.time_in_force == "3"

    fill = next(
        info for info in updates if info.cl_ord_id == "META-PLAN" and info.ord_status == "2"
    )
    assert fill.account == "FINAL-ACC"
    assert fill.order_type == "5"
    assert fill.time_in_force == "6"

    history = manager.get_order_history("META-PLAN")
    assert [record["account"] for record in history] == [
        "STEP-ACC",
        "FINAL-ACC",
    ]
    assert history[0]["order_type"] == "4"
    assert history[-1]["order_type"] == "5"
    assert history[0]["time_in_force"] == "3"
    assert history[-1]["time_in_force"] == "6"

    telemetry = manager.snapshot_telemetry()
    completion = next(
        event
        for event in telemetry
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "META-PLAN"
    )
    assert completion.details.get("account") == "FINAL-ACC"
    assert completion.details.get("order_type") == "5"
    assert completion.details.get("time_in_force") == "6"


def test_mock_fix_allows_order_and_exec_id_overrides():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-CUSTOM",
                quantity=2.0,
                price=1.3,
                mock_order_id="ORDER-XYZ",
                mock_exec_id_prefix="CUSTOM",
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    custom_updates = [info for info in updates if info.cl_ord_id == "ORD-CUSTOM"]
    assert custom_updates
    assert {info.order_id for info in custom_updates} == {"ORDER-XYZ"}
    exec_ids = [info.exec_id for info in custom_updates]
    assert len(exec_ids) >= 3
    assert exec_ids[:3] == ["CUSTOM-001", "CUSTOM-002", "CUSTOM-003"]

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-CUSTOM"
    ]
    assert {event.details.get("order_id") for event in telemetry} == {"ORDER-XYZ"}
    assert [event.details.get("exec_id") for event in telemetry][:3] == exec_ids[:3]


def test_fix_application_adapter_tracks_metrics():
    adapter = _FIXApplicationAdapter("trade")
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    adapter.set_message_queue(queue)

    adapter.dispatch({35: b"8"})
    adapter.dispatch({35: b"8"})

    metrics = adapter.get_queue_metrics()
    assert metrics["delivered"] == 1
    assert metrics["dropped"] == 1


def test_mock_fix_wait_utilities():
    manager = MockFIXManager(market_data_interval=0.01, market_data_duration=0.05)
    try:
        assert manager.start()
        assert manager.wait_for_telemetry("market_data_snapshot", timeout=1.0)
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD2", quantity=1.0, price=1.1)
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("exec_type") == "F",
            timeout=1.0,
        )
        assert manager.wait_for_telemetry("nonexistent", timeout=0.01) is False, (
            "Unexpected telemetry for nonexistent event"
        )
    finally:
        manager.stop()


def test_mock_fix_supports_manual_updates_with_auto_complete_disabled():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.05,
        execution_interval=0.05,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-MANUAL",
                quantity=4.0,
                price=1.25,
                mock_auto_complete=False,
            )
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-MANUAL"
            and event.details.get("exec_type") == "0",
            timeout=1.0,
        )
        assert manager.wait_for_idle(timeout=1.0)

        active = manager.snapshot_active_orders()
        assert {info.cl_ord_id for info in active} == {"ORD-MANUAL"}
        assert active[0].ord_status == "0"
        assert active[0].leaves_qty == pytest.approx(4.0)

        assert manager.emit_order_update(
            "ORD-MANUAL",
            "1",
            quantity=1.5,
            price=1.3,
            text="manual partial",
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-MANUAL"
            and event.details.get("exec_type") == "1",
            timeout=1.0,
        )

        partial = next(
            info for info in updates if info.cl_ord_id == "ORD-MANUAL" and info.ord_status == "1"
        )
        assert partial.last_qty == pytest.approx(1.5)
        assert partial.leaves_qty == pytest.approx(2.5)
        assert partial.text == "manual partial"

        assert manager.complete_order(
            "ORD-MANUAL",
            quantity=2.5,
            price=1.32,
            text="done",
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-MANUAL"
            and event.details.get("exec_type") == "F",
            timeout=1.0,
        )
        assert manager.wait_for_telemetry(
            "order_complete",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-MANUAL",
            timeout=1.0,
        )
    finally:
        manager.stop()

    telemetry = manager.snapshot_telemetry()
    received = next(
        event
        for event in telemetry
        if event.event == "order_received" and event.details.get("cl_ord_id") == "ORD-MANUAL"
    )
    assert received.details.get("auto_complete") is False

    history = manager.get_order_history("ORD-MANUAL")
    assert [record["exec_type"] for record in history] == ["0", "1", "F"]
    assert history[-1]["cum_qty"] == pytest.approx(4.0)
    assert history[-1]["leaves_qty"] == pytest.approx(0.0)
    assert history[-1]["last_px"] == pytest.approx(1.32)
    assert history[-1]["text"] == "done"

    info = manager.get_last_order_info("ORD-MANUAL")
    assert info is not None
    assert info.ord_status == "2"
    assert info.cum_qty == pytest.approx(4.0)
    assert info.leaves_qty == pytest.approx(0.0)


def test_mock_fix_manual_updates_override_order_metadata():
    manager = MockFIXManager(
        synchronous_order_flows=True,
        default_account="DEF-ACC",
        default_order_type="1",
        default_time_in_force="0",
    )
    updates: list[MockOrderInfo] = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="MANUAL-META",
                quantity=3.0,
                price=1.2,
                mock_auto_complete=False,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)

        assert manager.emit_order_update(
            "MANUAL-META",
            "1",
            quantity=1.0,
            price=1.25,
            account="MANUAL-ACC",
            order_type="3",
            time_in_force="4",
        )
        assert manager.emit_order_update(
            "MANUAL-META",
            "F",
            quantity=2.0,
            price=1.3,
            account="FINAL-ACC",
            order_type="5",
            time_in_force="6",
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    partial = next(
        info for info in updates if info.cl_ord_id == "MANUAL-META" and info.ord_status == "1"
    )
    assert partial.account == "MANUAL-ACC"
    assert partial.order_type == "3"
    assert partial.time_in_force == "4"

    fill = next(
        info for info in updates if info.cl_ord_id == "MANUAL-META" and info.ord_status == "2"
    )
    assert fill.account == "FINAL-ACC"
    assert fill.order_type == "5"
    assert fill.time_in_force == "6"

    history = manager.get_order_history("MANUAL-META")
    partial_record = next(record for record in history if record["exec_type"] == "1")
    fill_record = next(record for record in history if record["exec_type"] == "F")
    assert partial_record["account"] == "MANUAL-ACC"
    assert partial_record["order_type"] == "3"
    assert partial_record["time_in_force"] == "4"
    assert fill_record["account"] == "FINAL-ACC"
    assert fill_record["order_type"] == "5"
    assert fill_record["time_in_force"] == "6"

    telemetry = manager.snapshot_telemetry()
    completion = next(
        event
        for event in telemetry
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "MANUAL-META"
    )
    assert completion.details.get("account") == "FINAL-ACC"
    assert completion.details.get("order_type") == "5"
    assert completion.details.get("time_in_force") == "6"


def test_mock_fix_allows_order_replacement_with_execution_plan():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.05,
        execution_interval=0.01,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-ORIG",
                quantity=4.0,
                price=1.1,
                mock_auto_complete=False,
            )
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-ORIG"
            and event.details.get("exec_type") == "0",
            timeout=1.0,
        )
        assert manager.wait_for_idle(timeout=1.0)

        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-NEW",
                original_cl_ord_id="ORD-ORIG",
                replace=True,
                quantity=4.0,
                price=1.2,
                mock_execution_plan=[
                    MockExecutionStep(exec_type="1", quantity=1.0, delay=0.0, price=1.21),
                    MockExecutionStep(exec_type="F", quantity=3.0, delay=0.0, price=1.25),
                ],
            )
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-NEW"
            and event.details.get("exec_type") == "5",
            timeout=1.0,
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-NEW"
            and event.details.get("exec_type") == "F",
            timeout=1.0,
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    history = manager.get_order_history("ORD-NEW")
    assert [record["exec_type"] for record in history] == ["0", "5", "1", "F"]
    assert history[-1]["cum_qty"] == pytest.approx(4.0)
    assert history[-2]["last_qty"] == pytest.approx(1.0)

    replace_updates = [info for info in updates if info.cl_ord_id == "ORD-NEW"]
    assert [info.executions[-1].get("exec_type") for info in replace_updates[:3]] == ["5", "1", "F"]

    telemetry = manager.snapshot_telemetry()
    replace_received = next(
        event
        for event in telemetry
        if event.event == "order_received" and event.details.get("cl_ord_id") == "ORD-NEW"
    )
    assert replace_received.details.get("replace") is True
    assert replace_received.details.get("original_cl_ord_id") == "ORD-ORIG"

    assert manager.get_order_history("ORD-ORIG") == []
    info = manager.get_last_order_info("ORD-NEW")
    assert info is not None
    assert info.order_px == pytest.approx(1.2)
    assert info.last_px == pytest.approx(1.25)
    assert info.ord_status == "2"


def test_mock_fix_propagates_text_and_reasons():
    manager = MockFIXManager(market_data_interval=0.01, market_data_duration=0.02)
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-REJECT",
                reject=True,
                mock_reject_reason="99",
                mock_text="Rejected for testing",
            )
        )
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-CANCEL",
                cancel=True,
                quantity=1.0,
                mock_cancel_reason="0",
                mock_cancel_text="Canceled for testing",
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") in {"ORD-REJECT", "ORD-CANCEL"},
            count=2,
            timeout=1.0,
        )
    finally:
        manager.stop()

    reject_info = next(info for info in updates if info.cl_ord_id == "ORD-REJECT")
    assert reject_info.ord_rej_reason == "99"
    assert reject_info.text == "Rejected for testing"
    assert reject_info.avg_px == pytest.approx(0.0)

    cancel_info = next(info for info in updates if info.cl_ord_id == "ORD-CANCEL")
    assert cancel_info.cancel_reason == "0"
    assert cancel_info.text == "Canceled for testing"

    telemetry = [
        event for event in manager.snapshot_telemetry() if event.event == "order_execution"
    ]
    reject_event = next(
        event for event in telemetry if event.details.get("cl_ord_id") == "ORD-REJECT"
    )
    assert reject_event.details.get("ord_rej_reason") == "99"
    assert reject_event.details.get("text") == "Rejected for testing"
    cancel_event = next(
        event for event in telemetry if event.details.get("cl_ord_id") == "ORD-CANCEL"
    )
    assert cancel_event.details.get("cancel_reason") == "0"
    assert cancel_event.details.get("text") == "Canceled for testing"


def test_mock_fix_allows_per_order_partial_ratio_override():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.05,
        partial_fill_ratio=0.9,
    )
    received: list = []
    manager.add_order_callback(received.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-RATIO",
                quantity=8.0,
                price=1.111,
                mock_partial_fill_ratio=0.25,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-RATIO"
            and event.details.get("exec_type") == "F",
            timeout=1.0,
        )
    finally:
        manager.stop()

    partials = [
        info
        for info in received
        if info.cl_ord_id == "ORD-RATIO"
        and info.executions
        and info.executions[-1].get("exec_type") == "1"
    ]
    assert partials
    partial = partials[0]
    assert partial.last_qty == pytest.approx(2.0)
    assert partial.cum_qty == pytest.approx(2.0)

    telemetry = manager.snapshot_telemetry()
    ratio_events = [
        event.details.get("partial_fill_ratio")
        for event in telemetry
        if event.event == "order_received" and event.details.get("cl_ord_id") == "ORD-RATIO"
    ]
    assert ratio_events
    assert ratio_events[0] == pytest.approx(0.25)


def test_mock_fix_custom_execution_plan_controls_flow():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.05,
        execution_interval=0.01,
    )
    received: list = []
    manager.add_order_callback(received.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0),
        MockExecutionStep(exec_type="1", quantity=1.5, delay=0.0),
        {"exec_type": "1", "quantity": 0.5, "delay": 0.0},
        ("F", 1.0, 0.0),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-PLAN",
                quantity=3.0,
                price=1.25,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    order_updates = [info for info in received if info.cl_ord_id == "ORD-PLAN"]
    assert [info.ord_status for info in order_updates] == ["0", "1", "1", "2"]

    partial_qtys = [
        info.last_qty
        for info in order_updates
        if info.executions and info.executions[-1].get("exec_type") == "1"
    ]
    assert len(partial_qtys) == 2
    assert partial_qtys[0] == pytest.approx(1.5)
    assert partial_qtys[1] == pytest.approx(0.5)

    fill = next(
        info
        for info in order_updates
        if info.executions and info.executions[-1].get("exec_type") == "F"
    )
    assert fill.last_qty == pytest.approx(1.0)
    assert fill.cum_qty == pytest.approx(3.0)
    assert fill.leaves_qty == pytest.approx(0.0)

    exec_sequence = [
        event.details.get("exec_type")
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-PLAN"
    ]
    assert exec_sequence[:4] == ["0", "1", "1", "F"]


def test_mock_fix_tracks_average_price():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0),
        MockExecutionStep(exec_type="1", quantity=2.0, delay=0.0, price=1.0),
        MockExecutionStep(exec_type="F", quantity=2.0, delay=0.0, price=1.2),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-AVG",
                quantity=4.0,
                price=1.1,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    partial = next(
        info for info in updates if info.cl_ord_id == "ORD-AVG" and info.ord_status == "1"
    )
    assert partial.avg_px == pytest.approx(1.0)
    fill = next(info for info in updates if info.cl_ord_id == "ORD-AVG" and info.ord_status == "2")
    assert fill.avg_px == pytest.approx(1.1)

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-AVG"
    ]
    partial_event = next(event for event in telemetry if event.details.get("exec_type") == "1")
    assert partial_event.details.get("avg_px") == pytest.approx(1.0)
    fill_event = next(event for event in telemetry if event.details.get("exec_type") == "F")
    assert fill_event.details.get("avg_px") == pytest.approx(1.1)


def test_mock_fix_tracks_commission_metadata():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        partial_fill_ratio=0.5,
        default_commission=0.25,
        default_commission_type="ABS",
        default_commission_currency="USD",
    )
    updates: list[MockOrderInfo] = []
    manager.add_order_callback(updates.append)
    telemetry_snapshot: list = []

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-COMM", quantity=4.0, price=1.0)
        )
        assert manager.wait_for_idle(timeout=1.0)
        telemetry_snapshot = manager.snapshot_telemetry()
    finally:
        manager.stop()

    partial = next(
        info for info in updates if info.cl_ord_id == "ORD-COMM" and info.ord_status == "1"
    )
    assert partial.last_commission == pytest.approx(0.25)
    assert partial.cum_commission == pytest.approx(0.25)
    assert partial.comm_type == "ABS"
    assert partial.currency == "USD"

    fill = next(info for info in updates if info.cl_ord_id == "ORD-COMM" and info.ord_status == "2")
    assert fill.last_commission == pytest.approx(0.25)
    assert fill.cum_commission == pytest.approx(0.50)
    assert fill.comm_type == "ABS"
    assert fill.currency == "USD"

    history = manager.get_order_history("ORD-COMM")
    assert history
    assert history[-1]["cum_commission"] == pytest.approx(0.50)
    assert history[-1]["comm_type"] == "ABS"
    assert history[-1]["currency"] == "USD"

    fill_event = next(
        event
        for event in telemetry_snapshot
        if event.event == "order_execution"
        and event.details.get("cl_ord_id") == "ORD-COMM"
        and event.details.get("exec_type") == "F"
    )
    assert fill_event.details.get("cum_commission") == pytest.approx(0.50)
    assert fill_event.details.get("comm_type") == "ABS"
    assert fill_event.details.get("currency") == "USD"


def test_mock_fix_commission_overrides_and_config_updates():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
    )
    updates: list[MockOrderInfo] = []
    manager.add_order_callback(updates.append)
    telemetry_snapshot: list = []

    plan = [
        MockExecutionStep(
            exec_type="1",
            quantity=5.0,
            commission=0.4,
            comm_type="STEP",
            currency="EUR",
        ),
        MockExecutionStep(exec_type="F", quantity=5.0, commission=0.6),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="PLAN-COMM",
                quantity=10.0,
                price=2.0,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
        manager.configure_order_defaults(
            commission=0.75,
            commission_type="3",
            commission_currency="JPY",
        )
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="DEFAULT-COMM", quantity=4.0, price=1.1)
        )
        assert manager.wait_for_idle(timeout=1.0)
        telemetry_snapshot = manager.snapshot_telemetry()
    finally:
        manager.stop()

    plan_fill = next(
        info for info in updates if info.cl_ord_id == "PLAN-COMM" and info.ord_status == "2"
    )
    assert plan_fill.last_commission == pytest.approx(0.6)
    assert plan_fill.cum_commission == pytest.approx(1.0)
    assert plan_fill.comm_type == "STEP"
    assert plan_fill.currency == "EUR"

    plan_history = manager.get_order_history("PLAN-COMM")
    assert plan_history
    assert plan_history[-1]["cum_commission"] == pytest.approx(1.0)

    plan_complete = next(
        event
        for event in telemetry_snapshot
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "PLAN-COMM"
    )
    assert plan_complete.details.get("cum_commission") == pytest.approx(1.0)
    assert plan_complete.details.get("last_commission") == pytest.approx(0.6)
    assert plan_complete.details.get("comm_type") == "STEP"
    assert plan_complete.details.get("currency") == "EUR"

    default_fill = next(
        info for info in updates if info.cl_ord_id == "DEFAULT-COMM" and info.ord_status == "2"
    )
    assert default_fill.last_commission == pytest.approx(0.75)
    assert default_fill.cum_commission == pytest.approx(1.5)


def test_mock_fix_tracks_settlement_metadata():
    manager = MockFIXManager(
        synchronous_order_flows=True,
        default_settle_type="0",
        default_settle_date="20240102",
    )
    received: list[MockOrderInfo] = []
    manager.add_order_callback(received.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="SET1", quantity=4.0, price=1.5)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    assert received
    final_info = received[-1]
    assert final_info.settle_type == "0"
    assert final_info.settle_date == "20240102"

    info = manager.get_last_order_info("SET1")
    assert info is not None
    assert info.settle_type == "0"
    assert info.settle_date == "20240102"

    history = manager.get_order_history("SET1")
    assert history
    assert history[-1]["settle_type"] == "0"
    assert history[-1]["settle_date"] == "20240102"

    telemetry = manager.snapshot_telemetry()
    exec_events = [event for event in telemetry if event.event == "order_execution"]
    assert exec_events
    assert any(event.details.get("settle_type") == "0" for event in exec_events)
    assert any(event.details.get("settle_date") == "20240102" for event in exec_events)

    complete_events = [event for event in telemetry if event.event == "order_complete"]
    assert complete_events
    assert complete_events[-1].details.get("settle_type") == "0"
    assert complete_events[-1].details.get("settle_date") == "20240102"


def test_mock_fix_settlement_overrides_and_config_updates():
    manager = MockFIXManager(
        synchronous_order_flows=True,
        default_settle_type="2",
        default_settle_date="20240201",
    )

    try:
        assert manager.start()

        override_msg = SimpleNamespace(
            cl_ord_id="SET-OVR",
            quantity=5.0,
            price=1.25,
            mock_settle_type="3",
            mock_settle_date="20240315",
        )
        assert manager.trade_connection.send_message_and_track(override_msg)
        assert manager.wait_for_idle(timeout=1.0)

        override_info = manager.get_last_order_info("SET-OVR")
        assert override_info is not None
        assert override_info.settle_type == "3"
        assert override_info.settle_date == "20240315"

        override_history = manager.get_order_history("SET-OVR")
        assert override_history
        assert override_history[-1]["settle_type"] == "3"
        assert override_history[-1]["settle_date"] == "20240315"

        manual_msg = SimpleNamespace(
            cl_ord_id="SET-MAN",
            quantity=6.0,
            price=1.4,
            mock_auto_complete=False,
        )
        assert manager.trade_connection.send_message_and_track(manual_msg)
        assert manager.wait_for_idle(timeout=1.0)

        assert manager.emit_order_update(
            "SET-MAN",
            "1",
            quantity=2.0,
            settle_type="5",
            settle_date="20240505",
        )
        assert manager.emit_order_update(
            "SET-MAN",
            "F",
            quantity=4.0,
            settle_type="7",
            settle_date="20240507",
        )
        assert manager.wait_for_idle(timeout=1.0)

        manual_info = manager.get_last_order_info("SET-MAN")
        assert manual_info is not None
        assert manual_info.settle_type == "7"
        assert manual_info.settle_date == "20240507"

        manual_history = manager.get_order_history("SET-MAN")
        assert manual_history
        assert any(entry.get("settle_type") == "5" for entry in manual_history)
        assert manual_history[-1]["settle_type"] == "7"
        assert manual_history[-1]["settle_date"] == "20240507"

        manager.configure_order_defaults(settle_type="9", settle_date="20240606")

        default_msg = SimpleNamespace(
            cl_ord_id="SET-DEF",
            quantity=3.0,
            price=1.35,
        )
        assert manager.trade_connection.send_message_and_track(default_msg)
        assert manager.wait_for_idle(timeout=1.0)

        default_info = manager.get_last_order_info("SET-DEF")
        assert default_info is not None
        assert default_info.settle_type == "9"
        assert default_info.settle_date == "20240606"
    finally:
        manager.stop()


def test_mock_fix_supports_fill_price_override_on_message():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
    )
    received: list = []
    manager.add_order_callback(received.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-PX",
                quantity=4.0,
                price=1.2,
                mock_fill_price=1.45,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-PX"
            and event.details.get("exec_type") == "F",
            timeout=1.0,
        )
    finally:
        manager.stop()

    partial = next(
        info for info in received if info.cl_ord_id == "ORD-PX" and info.ord_status == "1"
    )
    assert partial.last_px == pytest.approx(1.45)
    fill = next(info for info in received if info.cl_ord_id == "ORD-PX" and info.ord_status == "2")
    assert fill.last_px == pytest.approx(1.45)

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-PX"
    ]
    partial_event = next(event for event in telemetry if event.details.get("exec_type") == "1")
    assert partial_event.details.get("last_px") == pytest.approx(1.45)
    fill_event = next(event for event in telemetry if event.details.get("exec_type") == "F")
    assert fill_event.details.get("last_px") == pytest.approx(1.45)


def test_mock_fix_execution_plan_allows_price_overrides():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
    )
    received: list = []
    manager.add_order_callback(received.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0),
        MockExecutionStep(exec_type="1", quantity=1.0, delay=0.0, price=1.42),
        MockExecutionStep(exec_type="F", quantity=1.0, delay=0.0, price=1.5),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-PXPLAN",
                quantity=2.0,
                price=1.3,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    updates = [info for info in received if info.cl_ord_id == "ORD-PXPLAN"]
    partial = next(info for info in updates if info.ord_status == "1")
    assert partial.last_px == pytest.approx(1.42)
    fill = next(info for info in updates if info.ord_status == "2")
    assert fill.last_px == pytest.approx(1.5)

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-PXPLAN"
    ]
    partial_event = next(event for event in telemetry if event.details.get("exec_type") == "1")
    assert partial_event.details.get("last_px") == pytest.approx(1.42)
    fill_event = next(event for event in telemetry if event.details.get("exec_type") == "F")
    assert fill_event.details.get("last_px") == pytest.approx(1.5)


def test_mock_fix_execution_plan_supports_id_overrides():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0, order_id="PLAN-ORDER", exec_id="PLAN-NEW"),
        MockExecutionStep(exec_type="F", quantity=2.0, delay=0.0, exec_id="PLAN-FILL"),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-PLAN-IDS",
                quantity=2.0,
                price=1.2,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    plan_updates = [info for info in updates if info.cl_ord_id == "ORD-PLAN-IDS"]
    assert len(plan_updates) == 2
    assert [info.order_id for info in plan_updates] == ["PLAN-ORDER", "PLAN-ORDER"]
    assert [info.exec_id for info in plan_updates] == ["PLAN-NEW", "PLAN-FILL"]

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-PLAN-IDS"
    ]
    exec_ids = [event.details.get("exec_id") for event in telemetry]
    assert exec_ids[:2] == ["PLAN-NEW", "PLAN-FILL"]
    assert {event.details.get("order_id") for event in telemetry} == {"PLAN-ORDER"}


def test_mock_fix_execution_plan_supports_ratio_steps():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0),
        MockExecutionStep(exec_type="1", delay=0.0, remaining_ratio=0.5),
        {"exec_type": "1", "ratio": 0.25, "delay": 0.0},
        MockExecutionStep(exec_type="F", delay=0.0),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-RATIO-PLAN",
                quantity=8.0,
                price=1.2,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    order_updates = [info for info in updates if info.cl_ord_id == "ORD-RATIO-PLAN"]
    assert [info.ord_status for info in order_updates] == ["0", "1", "1", "2"]

    partials = [info for info in order_updates if info.ord_status == "1"]
    assert len(partials) == 2
    first_partial, second_partial = partials
    assert first_partial.last_qty == pytest.approx(4.0)
    assert first_partial.cum_qty == pytest.approx(4.0)
    assert first_partial.leaves_qty == pytest.approx(4.0)
    assert second_partial.last_qty == pytest.approx(2.0)
    assert second_partial.cum_qty == pytest.approx(6.0)
    assert second_partial.leaves_qty == pytest.approx(2.0)

    fill = next(info for info in order_updates if info.ord_status == "2")
    assert fill.last_qty == pytest.approx(2.0)
    assert fill.cum_qty == pytest.approx(8.0)
    assert fill.leaves_qty == pytest.approx(0.0)

    history = manager.get_order_history("ORD-RATIO-PLAN")
    assert [record["exec_type"] for record in history] == ["0", "1", "1", "F"]
    assert history[-1]["cum_qty"] == pytest.approx(8.0)
    assert history[-1]["leaves_qty"] == pytest.approx(0.0)

    telemetry = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_execution" and event.details.get("cl_ord_id") == "ORD-RATIO-PLAN"
    ]
    partial_qtys = [
        event.details.get("last_qty")
        for event in telemetry
        if event.details.get("exec_type") == "1"
    ]
    assert partial_qtys[:2] == pytest.approx([4.0, 2.0])
    fill_event = next(event for event in telemetry if event.details.get("exec_type") == "F")
    assert fill_event.details.get("last_qty") == pytest.approx(2.0)


def test_mock_fix_execution_plan_supports_repeated_steps():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.01,
        synchronous_order_flows=True,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep(exec_type="0", delay=0.0),
        MockExecutionStep(exec_type="1", quantity=1.5, repeat=2, delay=0.0),
        {"exec_type": "1", "remaining_ratio": 0.5, "times": 2, "delay": 0.0},
        MockExecutionStep(exec_type="F", delay=0.0),
    ]

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="ORD-REPEAT-PLAN",
                quantity=6.0,
                price=2.0,
                mock_execution_plan=plan,
            )
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    order_updates = [info for info in updates if info.cl_ord_id == "ORD-REPEAT-PLAN"]
    exec_types = [info.executions[-1].get("exec_type") for info in order_updates if info.executions]
    assert exec_types == ["0", "1", "1", "1", "1", "F"]

    partials = [info for info in order_updates if info.executions[-1].get("exec_type") == "1"]
    assert [partial.last_qty for partial in partials] == [
        pytest.approx(1.5),
        pytest.approx(1.5),
        pytest.approx(1.5),
        pytest.approx(0.75),
    ]
    assert [partial.cum_qty for partial in partials] == [
        pytest.approx(1.5),
        pytest.approx(3.0),
        pytest.approx(4.5),
        pytest.approx(5.25),
    ]

    fill = next(info for info in order_updates if info.executions[-1].get("exec_type") == "F")
    assert fill.last_qty == pytest.approx(0.75)
    assert fill.cum_qty == pytest.approx(6.0)
    assert fill.leaves_qty == pytest.approx(0.0)

    history = manager.get_order_history("ORD-REPEAT-PLAN")
    assert [record["exec_type"] for record in history] == [
        "0",
        "1",
        "1",
        "1",
        "1",
        "F",
    ]
    assert history[-1]["cum_qty"] == pytest.approx(6.0)
    assert history[-1]["leaves_qty"] == pytest.approx(0.0)


def test_mock_fix_market_data_plan_emits_custom_levels():
    plan = [
        MockMarketDataStep(
            bids=[(1.0, 100.0), MockOrderBookLevel(price=0.99, size=90.0)],
            asks=[(1.01, 120.0)],
            delay=0.0,
        ),
        {"bids": [(1.02, 80.0)], "asks": [(1.03, 70.0)], "delay": 0.0},
    ]
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.01,
        market_data_plan=plan,
    )
    snapshots: list = []
    manager.add_market_data_callback(lambda symbol, book: snapshots.append((symbol, book)))

    try:
        assert manager.start()
        assert manager.wait_for_telemetry("market_data_snapshot", count=2, timeout=1.0)
        assert manager.wait_for_telemetry("market_data_complete", timeout=1.0)
    finally:
        manager.stop()

    assert len(snapshots) >= 2
    sym0, book0 = snapshots[0]
    assert sym0 == "EURUSD"
    assert book0.bids[0].price == pytest.approx(1.0)
    assert book0.asks[0].price == pytest.approx(1.01)
    _, book1 = snapshots[1]
    assert book1.bids[0].price == pytest.approx(1.02)

    telemetry = [
        event for event in manager.snapshot_telemetry() if event.event == "market_data_snapshot"
    ]
    assert telemetry[0].details.get("snapshot_index") == 0
    assert telemetry[0].details.get("bid") == pytest.approx(1.0)
    assert telemetry[0].details.get("plan_index") == 0
    assert telemetry[1].details.get("snapshot_index") == 1
    assert telemetry[1].details.get("bid") == pytest.approx(1.02)

    complete_events = [
        event for event in manager.snapshot_telemetry() if event.event == "market_data_complete"
    ]
    assert complete_events
    assert complete_events[0].details.get("steps") == 2


def test_mock_fix_market_data_plan_can_loop():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
    )
    manager.configure_market_data_plan(
        [MockMarketDataStep(bids=[(1.0, 50.0)], asks=[(1.01, 40.0)], delay=0.0)],
        loop=True,
    )
    snapshots: list = []
    manager.add_market_data_callback(lambda symbol, book: snapshots.append((symbol, book)))

    try:
        assert manager.start()
        assert manager.wait_for_telemetry("market_data_complete", timeout=1.0)
        assert manager.wait_for_telemetry("market_data_snapshot", count=3, timeout=1.0)
    finally:
        manager.stop()

    assert len(snapshots) >= 3
    telemetry = [
        event for event in manager.snapshot_telemetry() if event.event == "market_data_snapshot"
    ]
    assert len(telemetry) >= 3
    assert all(event.details.get("plan_index") == 0 for event in telemetry[:3])

    loop_events = [
        event for event in manager.snapshot_telemetry() if event.event == "market_data_complete"
    ]
    assert loop_events
    assert any(event.details.get("loop") for event in loop_events)


def test_mock_fix_tracks_active_orders_and_completion_events():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        execution_interval=0.1,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="ORD-ACTIVE", quantity=4.0, price=1.25)
        )
        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-ACTIVE"
            and event.details.get("exec_type") == "0",
            timeout=1.0,
        )

        assert "ORD-ACTIVE" in manager.list_active_order_ids()

        snapshots = manager.snapshot_active_orders()
        active_snapshot = next(info for info in snapshots if info.cl_ord_id == "ORD-ACTIVE")
        assert active_snapshot.ord_status == "0"
        assert active_snapshot.orig_qty == pytest.approx(4.0)
        assert active_snapshot.leaves_qty == pytest.approx(4.0)
        assert active_snapshot.order_px == pytest.approx(1.25)
        assert active_snapshot.order_id

        assert manager.wait_for_telemetry(
            "order_execution",
            predicate=lambda event: event.details.get("cl_ord_id") == "ORD-ACTIVE"
            and event.details.get("exec_type") == "1",
            timeout=1.0,
        )

        partial_snapshot = next(
            info for info in manager.snapshot_active_orders() if info.cl_ord_id == "ORD-ACTIVE"
        )
        assert partial_snapshot.ord_status == "1"
        assert partial_snapshot.leaves_qty == pytest.approx(2.0)

        partial_snapshot.leaves_qty = -1.0
        refreshed_snapshot = next(
            info for info in manager.snapshot_active_orders() if info.cl_ord_id == "ORD-ACTIVE"
        )
        assert refreshed_snapshot.leaves_qty == pytest.approx(2.0)

        assert manager.wait_for_order_completion("ORD-ACTIVE", timeout=1.0)
        assert manager.wait_for_idle(timeout=1.0)
        assert not manager.list_active_order_ids()
        assert not manager.snapshot_active_orders()
    finally:
        manager.stop()

    completion_events = [
        event
        for event in manager.snapshot_telemetry()
        if event.event == "order_complete" and event.details.get("cl_ord_id") == "ORD-ACTIVE"
    ]
    assert completion_events
    completion = completion_events[0]
    assert completion.details.get("final_status") == "2"
    assert completion.details.get("cum_qty") == pytest.approx(4.0)
    assert completion.details.get("leaves_qty") == pytest.approx(0.0)
    assert completion.details.get("last_exec_type") == "F"
    assert completion.details.get("avg_px") == pytest.approx(1.25)


def test_mock_fix_supports_synchronous_order_flows():
    manager = MockFIXManager(
        market_data_interval=0.01,
        market_data_duration=0.02,
        partial_fill_ratio=0.5,
        execution_interval=0.05,
        synchronous_order_flows=True,
    )
    updates: list = []
    manager.add_order_callback(updates.append)

    plan = [
        MockExecutionStep("0"),
        MockExecutionStep("1", quantity=3.0, delay=0.05),
        MockExecutionStep("F", quantity=2.0, delay=0.05),
    ]

    try:
        assert manager.start()
        start = time.perf_counter()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(
                cl_ord_id="SYNC-ORDER",
                quantity=5.0,
                price=1.23,
                mock_execution_plan=plan,
            )
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 0.09
        assert manager.wait_for_idle(timeout=0.01)
    finally:
        manager.stop()

    exec_types = [info.executions[-1].get("exec_type") for info in updates]
    assert exec_types == ["0", "1", "F"]

    history = manager.get_order_history("SYNC-ORDER")
    assert [record["exec_type"] for record in history] == ["0", "1", "F"]
    assert history[-1]["cum_qty"] == pytest.approx(5.0)
    assert history[-1]["leaves_qty"] == pytest.approx(0.0)

    info = manager.get_last_order_info("SYNC-ORDER")
    assert info is not None
    assert info.cum_qty == pytest.approx(5.0)
    assert info.leaves_qty == pytest.approx(0.0)


def test_mock_fix_tracks_trade_date_metadata():
    manager = MockFIXManager(
        synchronous_order_flows=True,
        default_trade_date="20240107",
    )
    received: list[MockOrderInfo] = []
    manager.add_order_callback(received.append)

    try:
        assert manager.start()
        assert manager.trade_connection.send_message_and_track(
            SimpleNamespace(cl_ord_id="TRADE-DATE", quantity=5.0, price=1.25)
        )
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    assert received
    info = received[-1]
    assert info.trade_date == "20240107"

    history = manager.get_order_history("TRADE-DATE")
    assert history
    assert history[-1]["trade_date"] == "20240107"

    telemetry = manager.snapshot_telemetry()
    exec_events = [event for event in telemetry if event.event == "order_execution"]
    assert exec_events
    assert any(event.details.get("trade_date") == "20240107" for event in exec_events)

    completion = [event for event in telemetry if event.event == "order_complete"]
    assert completion
    assert completion[-1].details.get("trade_date") == "20240107"


def test_mock_fix_trade_date_overrides_and_config_updates():
    manager = MockFIXManager(synchronous_order_flows=True)

    try:
        assert manager.start()

        override_msg = SimpleNamespace(
            cl_ord_id="TD-OVERRIDE",
            quantity=4.0,
            price=1.4,
            mock_trade_date="20240109",
        )
        assert manager.trade_connection.send_message_and_track(override_msg)
        assert manager.wait_for_idle(timeout=1.0)

        override_info = manager.get_last_order_info("TD-OVERRIDE")
        assert override_info is not None
        assert override_info.trade_date == "20240109"

        override_history = manager.get_order_history("TD-OVERRIDE")
        assert override_history
        assert override_history[-1]["trade_date"] == "20240109"

        manager.configure_order_defaults(trade_date="20240110")

        plan = [
            MockExecutionStep("0", delay=0.0),
            MockExecutionStep("1", quantity=3.0, trade_date="20240111", delay=0.0),
        ]
        plan_msg = SimpleNamespace(
            cl_ord_id="TD-PLAN",
            quantity=3.0,
            price=1.3,
            mock_execution_plan=plan,
            mock_auto_complete=False,
        )
        assert manager.trade_connection.send_message_and_track(plan_msg)
        assert manager.wait_for_idle(timeout=1.0)

        plan_info = manager.get_last_order_info("TD-PLAN")
        assert plan_info is not None
        assert plan_info.trade_date == "20240111"

        plan_history = manager.get_order_history("TD-PLAN")
        assert plan_history
        assert plan_history[-1]["trade_date"] == "20240111"

        assert manager.emit_order_update(
            "TD-PLAN",
            "F",
            quantity=0.0,
            trade_date="20240112",
        )
        final_info = manager.get_last_order_info("TD-PLAN")
        assert final_info is not None
        assert final_info.trade_date == "20240112"

        default_msg = SimpleNamespace(
            cl_ord_id="TD-DEFAULT",
            quantity=2.0,
            price=1.05,
        )
        assert manager.trade_connection.send_message_and_track(default_msg)
        assert manager.wait_for_idle(timeout=1.0)
    finally:
        manager.stop()

    default_info = manager.get_last_order_info("TD-DEFAULT")
    assert default_info is not None
    assert default_info.trade_date == "20240110"

    telemetry = manager.snapshot_telemetry()
    assert any(
        event.event == "order_execution"
        and event.details.get("cl_ord_id") == "TD-PLAN"
        and event.details.get("trade_date") in {"20240111", "20240112"}
        for event in telemetry
    )
