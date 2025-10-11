from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.trading.execution.resource_monitor import ResourceUsageMonitor


class _StubProcess:
    def __init__(self) -> None:
        self._cpu_calls = 0

    def cpu_percent(self, interval: Any | None = None) -> float:
        self._cpu_calls += 1
        return 12.5 + self._cpu_calls

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=256 * 1024 * 1024)

    def memory_percent(self) -> float:
        return 42.0


def test_resource_monitor_with_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_process = _StubProcess()

    class _StubPsutilModule:
        def Process(self) -> _StubProcess:
            return stub_process

    monkeypatch.setitem(sys.modules, "psutil", _StubPsutilModule())
    monitor = ResourceUsageMonitor()
    snapshot = monitor.sample()
    assert snapshot["cpu_percent"] == pytest.approx(13.5)
    assert snapshot["memory_mb"] == pytest.approx(256.0)
    assert snapshot["memory_percent"] == pytest.approx(42.0)
    assert snapshot["timestamp"] is not None


def test_resource_monitor_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monitor = ResourceUsageMonitor(process=None)
    snapshot = monitor.sample()
    assert snapshot == {
        "timestamp": None,
        "cpu_percent": None,
        "memory_mb": None,
        "memory_percent": None,
    }
