from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.reports import TestReport

from src.testing.flake_telemetry import (
    FlakeTelemetrySink,
    clip_longrepr,
    resolve_output_path,
    should_record_event,
)

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

_sink: FlakeTelemetrySink | None = None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run async tests via ``asyncio.run`` when no asyncio plugin is available."""

    plugin_manager = pyfuncitem.config.pluginmanager
    if plugin_manager.has_plugin("asyncio") or plugin_manager.has_plugin("pytest_asyncio"):
        return None

    test_obj = pyfuncitem.obj
    if not (inspect.iscoroutinefunction(test_obj) or asyncio.iscoroutinefunction(test_obj)):
        return None

    provided_args = pyfuncitem.funcargs
    signature = inspect.signature(test_obj)
    bound_args = {
        name: provided_args[name] for name in signature.parameters if name in provided_args
    }

    asyncio.run(test_obj(**bound_args))
    return True


def pytest_addoption(parser: Parser) -> None:
    """Register pytest options for flake telemetry output."""

    parser.addini(
        "flake_log_path",
        "Relative path (from repo root) for pytest flake telemetry JSON output.",
        default="tests/.telemetry/flake_runs.json",
    )
    parser.addoption(
        "--flake-log-file",
        action="store",
        dest="flake_log_file",
        help="Override the path used for pytest flake telemetry JSON output.",
    )


def pytest_configure(config: Config) -> None:
    """Prepare the telemetry sink for this test session."""

    global _sink

    explicit = config.getoption("flake_log_file")
    ini_value = config.getini("flake_log_path")
    env_value = os.environ.get("PYTEST_FLAKE_LOG")
    output_path = resolve_output_path(Path(config.rootpath), explicit, ini_value, env_value)

    _sink = FlakeTelemetrySink(output_path)
    config._flake_telemetry_sink = _sink  # type: ignore[attr-defined]


def pytest_runtest_logreport(report: TestReport) -> None:
    """Record failing/xfail call outcomes for flake telemetry."""

    if _sink is None or report.when != "call":
        return

    was_xfail = bool(getattr(report, "wasxfail", False))
    if not should_record_event(report.outcome, was_xfail):
        return

    longrepr_text: str | None = None
    try:
        longrepr_text = getattr(report, "longreprtext", None)
    except Exception:
        longrepr_text = None

    if not longrepr_text and getattr(report, "longrepr", None) is not None:
        longrepr_text = str(report.longrepr)

    _sink.record_event(
        nodeid=report.nodeid,
        outcome=report.outcome,
        duration=float(getattr(report, "duration", 0.0)),
        was_xfail=was_xfail,
        longrepr=clip_longrepr(longrepr_text),
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Flush telemetry to disk and surface any write failures."""

    global _sink

    if _sink is None:
        return

    try:
        _sink.flush(exitstatus)
    except Exception as exc:  # pragma: no cover - best effort logging
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(f"[flake-telemetry] failed to write telemetry: {exc}")
    finally:
        _sink = None
