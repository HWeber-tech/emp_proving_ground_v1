import logging
import types

import pytest

import src.operational.fix_connection_manager as fix_manager


class DummyConfig:
    environment = "test"
    account_number = "0000"
    password = "secret"


def test_fix_manager_handles_failed_mock_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FailingMock:
        def __init__(self, *_, **__):
            return None

        def add_market_data_callback(self, _cb):
            return None

        def add_order_callback(self, _cb):
            return None

        def start(self) -> bool:
            return False

    monkeypatch.setenv("EMP_USE_MOCK_FIX", "1")
    monkeypatch.setattr(fix_manager, "MockFIXManager", FailingMock)
    caplog.set_level(logging.ERROR)

    manager = fix_manager.FIXConnectionManager(DummyConfig())
    assert manager.start_sessions() is False
    assert any("Failed to start FIX Manager sessions" in message for message in caplog.messages)


def test_initiator_adapter_rejects_without_trade_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = fix_manager._FIXInitiatorAdapter(types.SimpleNamespace(trade_connection=None))
    caplog.set_level(logging.ERROR)
    assert adapter.send_message(object()) is False
    assert any("Trade connection not initialized" in message for message in caplog.messages)
