import logging
import sys

from structlog.stdlib import BoundLogger


class RecordingLogger(logging.Logger):
    def __init__(self) -> None:
        super().__init__("structlog-test")
        self.last_exc_info = None
        self.last_extra = None

    def _log(
        self,
        level: int,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ) -> None:
        self.last_exc_info = exc_info
        self.last_extra = extra


def passthrough_processor(_logger, _method_name, event_dict):
    return event_dict


def test_bound_logger_forwards_exc_info() -> None:
    logger = RecordingLogger()
    bound = BoundLogger(logger=logger, processor=passthrough_processor)

    exc_info = None
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
        bound.error("failure", exc_info=exc_info)

    assert logger.last_exc_info == exc_info
    assert "exc_info" not in logger.last_extra["structlog_event_dict"]
