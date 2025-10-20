"""Context management helpers mirroring ``structlog.contextvars``."""

from __future__ import annotations

from contextvars import ContextVar
from typing import MutableMapping


_CONTEXT: ContextVar[MutableMapping[str, object]] = ContextVar("structlog_context", default={})


def bind_contextvars(**values: object) -> None:
    """Bind key/value pairs into the current logging context."""

    if not values:
        return

    current = dict(_CONTEXT.get())
    current.update(values)
    _CONTEXT.set(current)


def unbind_contextvars(*keys: str) -> None:
    """Remove keys from the current logging context."""

    if not keys:
        return

    current = dict(_CONTEXT.get())
    for key in keys:
        current.pop(key, None)
    _CONTEXT.set(current)


def get_contextvars() -> MutableMapping[str, object]:
    """Return a shallow copy of the current logging context."""

    return dict(_CONTEXT.get())


def clear_contextvars() -> None:
    """Remove all key/value pairs from the current logging context."""

    _CONTEXT.set({})


def merge_contextvars(
    _logger: object,
    _method_name: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Processor that merges bound context into the event dictionary."""

    context = _CONTEXT.get()
    if not context:
        return event_dict
    merged = dict(context)
    merged.update(event_dict)
    return merged


__all__ = [
    "bind_contextvars",
    "unbind_contextvars",
    "get_contextvars",
    "clear_contextvars",
    "merge_contextvars",
]
