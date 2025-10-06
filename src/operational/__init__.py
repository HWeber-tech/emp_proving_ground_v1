"""Operational modules for IC Markets FIX API integration."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable

from .structured_logging import (
    bind_context,
    configure_structlog,
    get_logger,
    order_logging_context,
    unbind_context,
)

__all__ = [
    "bind_context",
    "configure_structlog",
    "get_logger",
    "order_logging_context",
    "unbind_context",
]


def _register_event_bus_aliases(names: Iterable[str]) -> None:
    """Point legacy event bus import paths at the canonical implementation."""

    try:
        module = importlib.import_module("src.core.event_bus")
    except ModuleNotFoundError:  # pragma: no cover - defensive safeguard
        return

    for name in names:
        sys.modules.setdefault(name, module)


_register_event_bus_aliases(("operational.event_bus", "src.operational.event_bus"))
