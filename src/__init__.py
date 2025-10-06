"""
EMP Proving Ground - Development Trading System

Lightweight package init to avoid heavy imports at module import time.
"""

from __future__ import annotations

import typing as _typing
import datetime as _datetime
import enum as _enum

if not hasattr(_datetime, "UTC"):  # pragma: no cover - runtime compatibility shim
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

if not hasattr(_enum, "StrEnum"):  # pragma: no cover - Python 3.10 compatibility
    class _StrEnum(str, _enum.Enum):
        """Minimal fallback for :class:`enum.StrEnum`."""

    _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

try:  # Python 3.10 compatibility for typing.Optional[...] helpers
    _typing.NotRequired  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - ensure TypedDict extras exist
    try:
        from typing_extensions import NotRequired as _NotRequired  # type: ignore
        from typing_extensions import Required as _Required  # type: ignore
    except ImportError:  # pragma: no cover - minimal shim if typing_extensions missing
        class _Stub:  # pylint: disable=too-few-public-methods
            def __getitem__(self, item):  # type: ignore[override]
                return item

        _NotRequired = _Stub()  # type: ignore[assignment]
        _Required = _Stub()  # type: ignore[assignment]

    _typing.NotRequired = _NotRequired  # type: ignore[attr-defined]
    _typing.Required = _Required  # type: ignore[attr-defined]

__version__ = "2.0.0"
__author__ = "EMP Team"

__all__: list[str] = []
