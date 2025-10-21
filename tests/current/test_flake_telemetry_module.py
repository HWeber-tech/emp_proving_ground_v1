from __future__ import annotations

import json
from pathlib import Path

from src.testing.flake_telemetry import (
    MAX_LONGREPR_LENGTH,
    FlakeTelemetrySink,
    clip_longrepr,
    resolve_output_path,
    should_record_event,
)


def test_resolve_output_path_creates_parent(tmp_path: Path) -> None:
    path = resolve_output_path(
        tmp_path, explicit=None, ini_value="artifacts/flake.json", env_value=None
    )
    assert path == tmp_path / "artifacts" / "flake.json"
    assert path.parent.exists()


def test_should_record_event_matrix() -> None:
    assert should_record_event("failed", False) is True
    assert should_record_event("error", False) is True
    assert should_record_event("FAILED ", False) is True
    assert should_record_event("passed", True) is True
    assert should_record_event("passed", False) is False
    assert should_record_event("skipped", False) is False


def test_should_record_event_handles_non_string_outcome() -> None:
    assert should_record_event(None, False) is False


def test_clip_longrepr_truncates_when_needed() -> None:
    assert clip_longrepr(None) is None
    assert clip_longrepr("short") == "short"
    assert clip_longrepr("abcdef", limit=3) == "abc… [truncated 3 chars]"


def test_clip_longrepr_clamps_negative_limits() -> None:
    assert clip_longrepr("abcdef", limit=-1) == "… [truncated 6 chars]"


def test_sink_writes_payload(tmp_path: Path) -> None:
    sink = FlakeTelemetrySink(tmp_path / "out.json")
    sink.record_event(
        nodeid="test_module::test_case",
        outcome="failed",
        duration=0.5,
        was_xfail=False,
        longrepr="trace",
    )
    payload = sink.flush(exit_status=1)

    stored = json.loads((tmp_path / "out.json").read_text())
    assert stored == payload
    assert stored["meta"]["exit_status"] == 1
    assert stored["meta"]["event_count"] == 1
    assert stored["events"] == [
        {
            "nodeid": "test_module::test_case",
            "outcome": "failed",
            "duration": 0.5,
            "was_xfail": False,
            "longrepr": "trace",
        }
    ]


def test_sink_clips_longrepr_on_record(tmp_path: Path) -> None:
    sink = FlakeTelemetrySink(tmp_path / "out.json")
    longrepr = "x" * (MAX_LONGREPR_LENGTH + 5)

    sink.record_event(
        nodeid="test_module::longrepr",
        outcome="failed",
        duration=0.1,
        was_xfail=False,
        longrepr=longrepr,
    )

    stored = sink.events[0]["longrepr"]
    assert stored == clip_longrepr(longrepr)
    assert stored != longrepr


def test_sink_normalizes_unparseable_duration(tmp_path: Path) -> None:
    sink = FlakeTelemetrySink(tmp_path / "out.json")

    sink.record_event(
        nodeid="test_module::bad_duration",
        outcome="failed",
        duration="0.5s",
        was_xfail=False,
        longrepr=None,
    )

    assert sink.events[0]["duration"] == 0.0
