from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data_foundation.persist import parquet_writer


class _RecordingDataFrame:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def to_parquet(self, path: str, *, index: bool) -> None:  # noqa: ARG002 - compatibility shim
        destination = Path(path)
        destination.write_text(json.dumps(self._events))


class _RecordingPandas:
    def DataFrame(self, events: list[dict[str, Any]]) -> _RecordingDataFrame:  # noqa: N802 - pandas compatibility
        return _RecordingDataFrame(events)


class _FailingDataFrame:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def to_parquet(self, path: str, *, index: bool) -> None:  # noqa: ARG002 - compatibility shim
        raise self._exc


class _FailingPandas:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def DataFrame(self, events: list[dict[str, Any]]) -> _FailingDataFrame:  # noqa: N802 - pandas compatibility
        raise self._exc


def test_write_events_parquet_creates_partition_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pandas_stub = _RecordingPandas()
    monkeypatch.setattr(parquet_writer, "_pd", pandas_stub, raising=False)

    result = parquet_writer.write_events_parquet(
        events=[{"symbol": "EMP", "price": 1.23}],
        out_dir=str(tmp_path),
        partition="2023-01-01",
    )

    assert result
    path = Path(result)
    assert path.exists()
    assert path.parent == tmp_path / "partition=2023-01-01"
    assert json.loads(path.read_text()) == [{"symbol": "EMP", "price": 1.23}]


def test_write_events_parquet_logs_dataframe_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    error = TypeError("bad payload")
    pandas_stub = _FailingPandas(error)
    monkeypatch.setattr(parquet_writer, "_pd", pandas_stub, raising=False)

    caplog.set_level("ERROR")
    result = parquet_writer.write_events_parquet(
        events=[{"symbol": "EMP", "price": 1.23}],
        out_dir="/tmp/ignored",
        partition="ignored",
    )

    assert result == ""
    assert any("Failed to convert ingest events into a pandas DataFrame" in record.message for record in caplog.records)


def test_write_events_parquet_logs_serialisation_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    error = OSError("disk error")
    dataframe = _FailingDataFrame(error)

    class _StubPandas:
        def DataFrame(self, events: list[dict[str, Any]]) -> _FailingDataFrame:  # noqa: N802 - pandas compatibility
            return dataframe

    monkeypatch.setattr(parquet_writer, "_pd", _StubPandas(), raising=False)

    caplog.set_level("ERROR")
    result = parquet_writer.write_events_parquet(
        events=[{"symbol": "EMP", "price": 1.23}],
        out_dir=str(tmp_path),
        partition="2023-01-01",
    )

    assert result == ""
    assert any("Failed to persist ingest events to Parquet" in record.message for record in caplog.records)
