from __future__ import annotations

import json
import logging

from src.structlog.stdlib import ProcessorFormatter


def test_processor_formatter_handles_non_mapping_event_dict() -> None:
    formatter = ProcessorFormatter(
        processor=lambda logger, method_name, event_dict: json.dumps(event_dict, sort_keys=True)
    )

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.structlog_event_dict = "not a mapping"

    rendered = formatter.format(record)

    assert json.loads(rendered) == {"event": "hello world"}
