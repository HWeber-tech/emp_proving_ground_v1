"""
EMP Proving Ground - Development Trading System

Lightweight package init to avoid heavy imports at module import time.
"""

from __future__ import annotations

import datetime as _dt
import enum as _enum
import typing as _typing
from datetime import timezone as _timezone

try:  # pragma: no cover - Python 3.11+ already exposes datetime.UTC
    _dt.UTC  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Python 3.10 compatibility shim
    setattr(_dt, "UTC", _timezone.utc)

if not hasattr(_enum, "StrEnum"):  # pragma: no cover - Python 3.10 fallback
    class _StrEnum(str, _enum.Enum):
        pass

    setattr(_enum, "StrEnum", _StrEnum)

if not hasattr(_typing, "NotRequired"):  # pragma: no cover - typing shim
    try:
        from typing_extensions import NotRequired as _NotRequired  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - minimal environments
        class _NotRequired:  # type: ignore[too-many-ancestors]
            def __class_getitem__(cls, item):  # noqa: D401 - mimic typing form
                return item

    setattr(_typing, "NotRequired", _NotRequired)

if not hasattr(_typing, "Unpack"):  # pragma: no cover - typing shim
    try:
        from typing_extensions import Unpack as _Unpack  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - minimal environments
        class _Unpack:  # type: ignore[too-many-ancestors]
            def __class_getitem__(cls, item):
                return item

    setattr(_typing, "Unpack", _Unpack)

__version__ = "2.0.0"
__author__ = "EMP Team"

__all__: list[str] = []
