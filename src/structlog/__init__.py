"""Minimal ``structlog`` compatibility layer used in tests.

This repository depends on :mod:`structlog` for structured logging helpers, but
the real dependency is not available inside the execution environment that runs
the kata.  To keep the public API stable—and avoid rewriting the higher level
logging helpers—we provide a tiny, well-documented shim that mimics the handful
of behaviours relied upon by the test-suite:

* ``structlog.configure`` wires a processor chain and logger factory.
* ``structlog.get_logger`` returns a bound logger instance.
* ``structlog.contextvars`` exposes helpers to bind contextual metadata.
* ``structlog.processors`` provides timestamp and JSON rendering processors.
* ``structlog.stdlib`` supplies a ``BoundLogger`` implementation alongside the
  ``ProcessorFormatter`` bridge used by :mod:`logging`.

The goal is not to be feature complete—only to support the narrow surface used
by :mod:`src.operational.structured_logging`.  The implementation intentionally
mirrors the structure of the upstream library so future contributors can swap in
the real dependency without touching call sites.
"""

from __future__ import annotations

from collections.abc import MutableMapping as MutableMappingABC
from dataclasses import dataclass
import logging
from typing import Callable, Iterable, MutableMapping

from . import contextvars, processors, stdlib

__all__ = [
    "configure",
    "get_logger",
    "make_filtering_bound_logger",
    "contextvars",
    "processors",
    "stdlib",
]


@dataclass
class _Configuration:
    processors: list[Callable[[logging.Logger | None, str, MutableMapping[str, object]], MutableMapping[str, object] | str]]
    logger_factory: Callable[[str | None], logging.Logger]
    wrapper_class: type
    cache_logger: bool


_config = _Configuration(
    processors=[],
    logger_factory=lambda name: logging.getLogger(name),
    wrapper_class=stdlib.BoundLogger,
    cache_logger=False,
)
_logger_cache: dict[str | None, stdlib.BoundLogger] = {}


def configure(
    *,
    processors: Iterable[Callable[[logging.Logger | None, str, MutableMapping[str, object]], MutableMapping[str, object] | str]] = (),
    context_class: type[MutableMapping[str, object]] | None = None,
    logger_factory: Callable[[str | None], logging.Logger] | None = None,
    wrapper_class: type | None = None,
    cache_logger_on_first_use: bool = False,
) -> None:
    """Configure the shim to mirror :func:`structlog.configure`.

    Only the arguments exercised by the repository are supported.  ``context_class``
    is accepted for signature compatibility but otherwise ignored because Python's
    built-in ``dict`` already satisfies our needs.
    """

    global _config, _logger_cache

    processor_chain = list(processors)
    if context_class is not None:
        _ = context_class  # preserve keyword compatibility without altering behaviour
    factory = logger_factory or (lambda name: logging.getLogger(name))
    wrapper = wrapper_class or stdlib.BoundLogger

    _config = _Configuration(
        processors=processor_chain,
        logger_factory=factory,
        wrapper_class=wrapper,
        cache_logger=cache_logger_on_first_use,
    )
    _logger_cache = {}


def _run_processors(
    logger: logging.Logger | None,
    method_name: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object] | str:
    """Execute the configured processor chain for a log event."""

    processed: MutableMapping[str, object] | str = event_dict
    for processor in _config.processors:
        processed = processor(logger, method_name, processed)  # type: ignore[arg-type]
        if not isinstance(processed, MutableMappingABC):
            break
    return processed


def get_logger(name: str | None = None) -> stdlib.BoundLogger:
    """Return a bound logger using the configured factory and wrapper."""

    if _config.cache_logger and name in _logger_cache:
        return _logger_cache[name]

    logger = _config.logger_factory(name)
    bound_logger = _config.wrapper_class(
        logger=logger,
        processor=_run_processors,
    )
    if _config.cache_logger:
        _logger_cache[name] = bound_logger
    return bound_logger


def make_filtering_bound_logger(level: int) -> type:
    """Return a ``BoundLogger`` variant that honours a minimum level."""

    class FilteringBoundLogger(stdlib.BoundLogger):
        _minimum_level = level

        def _should_log(self, levelno: int) -> bool:  # pragma: no cover - simple guard
            return levelno >= self._minimum_level

    return FilteringBoundLogger


def _reset_for_tests() -> None:
    """Internal helper used by tests to isolate global state."""

    configure()
