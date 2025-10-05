import importlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest


def _stub_domain_pkg(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid executing src.domain.__init__ which imports non-existent RiskConfig from src.core
    pkg = types.ModuleType("src.domain")
    pkg.__path__ = [str(Path("src/domain").resolve())]  # mark as namespace package
    monkeypatch.setitem(sys.modules, "src.domain", pkg)


def test_execution_report_json(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pydantic")
    _stub_domain_pkg(monkeypatch)
    dm = importlib.import_module("src.domain.models")
    ExecutionReport = getattr(dm, "ExecutionReport")

    er = ExecutionReport(
        event_id="e1",
        timestamp=datetime(2024, 1, 2, 3, 4, 5),
        source="unit",
        trade_intent_id="t1",
        action="NEW",
        status="FILLED",
        symbol="AAPL",
        side="BUY",
        quantity=1.0,
        price=100.0,
        order_id="o1",
    )
    payload = json.loads(er.json())
    assert payload["event_id"] == "e1"
    assert payload["symbol"] == "AAPL"
    assert payload["timestamp"] == "2024-01-02T03:04:05"

