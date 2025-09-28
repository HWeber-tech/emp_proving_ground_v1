from __future__ import annotations

import io
import json
import logging

from src.operational.structured_logging import (
    configure_structlog,
    get_logger,
    order_logging_context,
)


def _read_single_record(buffer: io.StringIO) -> dict[str, object]:
    contents = buffer.getvalue().strip().splitlines()
    assert contents, "expected at least one log record"
    return json.loads(contents[-1])


def test_order_logging_context_binds_and_unbinds_order_metadata() -> None:
    buffer = io.StringIO()
    configure_structlog(level=logging.INFO, stream=buffer)

    with order_logging_context("ORD-123") as log:
        log.info("acknowledged", status="ACK")

    record = _read_single_record(buffer)
    assert record["event"] == "acknowledged"
    assert record["order_id"] == "ORD-123"
    assert record["correlation_id"] == "ORD-123"
    assert record["status"] == "ACK"

    buffer.seek(0)
    buffer.truncate(0)

    logger = get_logger("test_structured_logging")
    logger.info("no_context")

    record = _read_single_record(buffer)
    assert record["event"] == "no_context"
    assert "order_id" not in record
    assert "correlation_id" not in record


def test_order_logging_context_supports_custom_correlation_id() -> None:
    buffer = io.StringIO()
    configure_structlog(level=logging.INFO, stream=buffer)

    with order_logging_context("ORD-456", correlation_id="chain-1") as log:
        log.info("fill")

    record = _read_single_record(buffer)
    assert record["event"] == "fill"
    assert record["order_id"] == "ORD-456"
    assert record["correlation_id"] == "chain-1"
