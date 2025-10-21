"""Minimal compatibility shims mirroring :mod:`structlog.stdlib`."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, MutableMapping


class LoggerFactory:
    """Factory returning :class:`logging.Logger` instances."""

    def __call__(self, name: str | None = None) -> logging.Logger:
        return logging.getLogger(name)


def add_logger_name(
    logger: logging.Logger | None,
    _method_name: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    if logger is not None:
        event_dict.setdefault("logger", logger.name)
    return event_dict


def add_log_level(
    _logger: logging.Logger | None,
    method_name: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    event_dict.setdefault("level", method_name.upper())
    return event_dict


class ProcessorFormatter(logging.Formatter):
    """Formatter that consumes event dictionaries from bound loggers."""

    def __init__(
        self,
        *,
        processor: Callable[[logging.Logger | None, str, MutableMapping[str, object]], str],
        foreign_pre_chain: list[
            Callable[[logging.Logger | None, str, MutableMapping[str, object]], MutableMapping[str, object]]
        ] | None = None,
    ) -> None:
        super().__init__()
        self._processor = processor
        self._foreign_pre_chain = foreign_pre_chain or []

    def format(self, record: logging.LogRecord) -> str:
        raw_event_dict = getattr(record, "structlog_event_dict", None)
        if isinstance(raw_event_dict, MutableMapping):
            event_dict = dict(raw_event_dict)
        else:
            message = record.getMessage()
            if raw_event_dict is not None:
                message = str(raw_event_dict)
            event_dict = {"event": message}
        logger = event_dict.get("_logger") if isinstance(event_dict.get("_logger"), logging.Logger) else None
        method_name = str(event_dict.get("_method_name", record.levelname.lower()))
        for processor in self._foreign_pre_chain:
            event_dict = processor(logger, method_name, event_dict)
        event_dict.pop("_logger", None)
        event_dict.pop("_method_name", None)
        return self._processor(logger, method_name, event_dict)

    @staticmethod
    def wrap_for_formatter(
        logger: logging.Logger | None,
        method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> MutableMapping[str, object]:
        event_dict.setdefault("_logger", logger)
        event_dict.setdefault("_method_name", method_name)
        return event_dict


@dataclass
class BoundLogger:
    """Very small subset of :class:`structlog.stdlib.BoundLogger` functionality."""

    logger: logging.Logger
    processor: Callable[[logging.Logger | None, str, MutableMapping[str, object]], MutableMapping[str, object] | str]
    context: MutableMapping[str, object] = field(default_factory=dict)

    def bind(self, **values: object) -> "BoundLogger":
        if not values:
            return self
        merged = dict(self.context)
        merged.update(values)
        return self.__class__(logger=self.logger, processor=self.processor, context=merged)

    # ------------------------------------------------------------------
    def _should_log(self, _levelno: int) -> bool:
        return True

    def _log(self, method_name: str, levelno: int, event: str, **kwargs: object) -> None:
        if not self._should_log(levelno):  # pragma: no cover - defensive guard
            return

        kwargs = dict(kwargs)
        exc_info = kwargs.pop("exc_info", None)

        event_dict: MutableMapping[str, object] = {"event": event}
        if self.context:
            event_dict.update(self.context)
        if kwargs:
            event_dict.update(kwargs)

        processed = self.processor(self.logger, method_name, event_dict)
        if isinstance(processed, dict):
            message = processed.get("event", event)
            extra = {"structlog_event_dict": processed}
        else:
            message = processed
            extra = {"structlog_event_dict": {"event": event}}

        log_kwargs: dict[str, object] = {"extra": extra}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info

        self.logger._log(levelno, message, args=(), **log_kwargs)

    # Convenience wrappers matching the upstream API -------------------
    def debug(self, event: str, **kwargs: object) -> None:
        self._log("debug", logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: object) -> None:
        self._log("info", logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._log("warning", logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._log("error", logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: object) -> None:
        self._log("critical", logging.CRITICAL, event, **kwargs)


__all__ = [
    "add_log_level",
    "add_logger_name",
    "BoundLogger",
    "LoggerFactory",
    "ProcessorFormatter",
]
