from __future__ import annotations

import io
import json
import logging
from typing import Iterator

import pytest

from src.observability.logging import configure_structured_logging


@pytest.fixture()
def reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    previous_propagate = root.propagate
    try:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)
        root.propagate = previous_propagate


def _last_record(stream: io.StringIO) -> dict[str, object]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    return json.loads(lines[-1])


def test_configure_structured_logging_emits_json(reset_logging: Iterator[None]) -> None:
    stream = io.StringIO()
    configure_structured_logging(
        component="runtime",
        level="INFO",
        static_fields={"deployment": "staging"},
        stream=stream,
    )

    logger = logging.getLogger("test.runtime")
    logger.info("ingest started", extra={"run_id": "abc123", "count": 3})

    record = _last_record(stream)
    assert record["component"] == "runtime"
    assert record["deployment"] == "staging"
    assert record["message"] == "ingest started"
    assert record["run_id"] == "abc123"
    assert record["count"] == 3
    assert record["level"] == "INFO"
    assert record["logger"] == "test.runtime"
    assert "timestamp" in record


def test_structured_logging_includes_exception(reset_logging: Iterator[None]) -> None:
    stream = io.StringIO()
    configure_structured_logging(component="runtime", stream=stream)

    logger = logging.getLogger("test.runtime.exc")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("ingest failed")

    record = _last_record(stream)
    assert record["message"] == "ingest failed"
    assert "ValueError" in record["exc_info"]


def test_configure_structured_logging_idempotent(reset_logging: Iterator[None]) -> None:
    first_stream = io.StringIO()
    first_handler = configure_structured_logging(component="runtime", stream=first_stream)
    second_stream = io.StringIO()
    second_handler = configure_structured_logging(component="runtime", stream=second_stream)

    root = logging.getLogger()
    assert first_handler not in root.handlers
    assert second_handler in root.handlers
    assert (
        sum(
            1
            for handler in root.handlers
            if getattr(handler, "name", "") == "emp-structured-logger"
        )
        == 1
    )
