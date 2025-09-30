from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_foundation.persist.jsonl_writer import JsonlWriterError, write_events_jsonl


def test_write_events_jsonl_success(tmp_path: Path) -> None:
    out_path = tmp_path / "events.jsonl"
    events = [{"symbol": "EURUSD", "price": 1.1234}, {"symbol": "GBPUSD", "price": 1.2567}]

    result = write_events_jsonl(events, str(out_path))

    assert Path(result).exists()
    contents = out_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in contents] == events


def test_write_events_jsonl_raises_for_unserialisable_payload(tmp_path: Path) -> None:
    out_path = tmp_path / "events.jsonl"
    events = [{"callback": lambda: None}]

    with pytest.raises(JsonlWriterError):
        write_events_jsonl(events, str(out_path))

    assert not out_path.exists()


def test_write_events_jsonl_raises_for_os_errors(tmp_path: Path) -> None:
    output_dir = tmp_path / "target"
    output_dir.mkdir()

    with pytest.raises(JsonlWriterError):
        write_events_jsonl([{"symbol": "EURUSD"}], str(output_dir))
