"""Regression coverage for :mod:`src.governance.kill_switch`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.governance.kill_switch import KillSwitchMonitor


@pytest.mark.asyncio()
async def test_monitor_triggers_when_file_created(tmp_path: Path) -> None:
    kill_file = tmp_path / "halt.flag"
    monitor = KillSwitchMonitor(kill_file, poll_interval=0.05)

    triggered = asyncio.Event()

    async def _callback(path: Path) -> None:
        assert path == kill_file
        triggered.set()

    task = asyncio.create_task(
        monitor.run(_callback, stop_event=asyncio.Event()), name="kill-switch-test"
    )

    try:
        await asyncio.sleep(0.1)
        kill_file.write_text("halt")

        await asyncio.wait_for(triggered.wait(), timeout=2)
    finally:
        await task

    assert monitor.triggered is True


@pytest.mark.asyncio()
async def test_monitor_stops_when_event_set(tmp_path: Path) -> None:
    kill_file = tmp_path / "halt.flag"
    monitor = KillSwitchMonitor(kill_file, poll_interval=0.05)

    stop_event = asyncio.Event()
    callback_invocations: list[Path] = []

    async def _callback(path: Path) -> None:
        callback_invocations.append(path)

    task = asyncio.create_task(monitor.run(_callback, stop_event=stop_event))
    try:
        await asyncio.sleep(0.1)
        stop_event.set()
    finally:
        await task

    assert not monitor.triggered
    assert not callback_invocations


@pytest.mark.asyncio()
async def test_monitor_requires_callable_callback(tmp_path: Path) -> None:
    kill_file = tmp_path / "halt.flag"
    monitor = KillSwitchMonitor(kill_file, poll_interval=0.05)

    with pytest.raises(TypeError):
        await monitor.run(object(), stop_event=asyncio.Event())  # type: ignore[arg-type]
