import datetime
import importlib
import logging
import sys
import types
import typing
from statistics import mean, median

import pytest


def _install_structlog_stub() -> None:
    if "structlog" in sys.modules:
        return

    module = types.ModuleType("structlog")

    class _ContextVars:
        @staticmethod
        def merge_contextvars(logger, method_name, event_dict):
            return event_dict

        @staticmethod
        def bind_contextvars(**values):
            return None

        @staticmethod
        def unbind_contextvars(*keys):
            return None

    class _Processors:
        @staticmethod
        def TimeStamper(fmt: str, utc: bool):
            def _processor(logger, method_name, event_dict):
                return event_dict

            return _processor

        @staticmethod
        def JSONRenderer():
            def _renderer(logger, method_name, event_dict):
                return event_dict

            return _renderer

    class _BoundLogger:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, **kwargs):  # pragma: no cover - compatibility stub
            return self

        def new(self, **kwargs):  # pragma: no cover - compatibility stub
            return self

        def info(self, *args, **kwargs):
            return None

    class _ProcessorFormatter:
        def __init__(self, processor):
            self.processor = processor

    class _LoggerFactory:
        def __call__(self):
            return logging.getLogger

    def _wrap_for_formatter(handler):  # pragma: no cover - compatibility stub
        return handler

    def _make_filtering_bound_logger(level):  # pragma: no cover - compatibility stub
        return _BoundLogger

    def _configure(**kwargs):  # pragma: no cover - compatibility stub
        return None

    def _get_logger(name=None):  # pragma: no cover - compatibility stub
        return _BoundLogger()

    module.contextvars = _ContextVars()
    module.processors = _Processors()
    module.stdlib = types.SimpleNamespace(
        BoundLogger=_BoundLogger,
        add_logger_name=lambda logger, name, event_dict: event_dict,
        add_log_level=lambda logger, method_name, event_dict: event_dict,
        ProcessorFormatter=_ProcessorFormatter,
        LoggerFactory=_LoggerFactory,
        wrap_for_formatter=_wrap_for_formatter,
        make_filtering_bound_logger=_make_filtering_bound_logger,
    )
    module.configure = _configure
    module.get_logger = _get_logger
    module.make_filtering_bound_logger = _make_filtering_bound_logger

    sys.modules["structlog"] = module


def _install_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return

    module = types.ModuleType("numpy")

    def _coerce(seq):
        return list(seq)

    def _mean(seq):
        data = _coerce(seq)
        return mean(data) if data else 0.0

    def _std(seq):
        data = _coerce(seq)
        if len(data) < 2:
            return 0.0
        avg = _mean(data)
        return (sum((x - avg) ** 2 for x in data) / len(data)) ** 0.5

    def _min(seq):
        data = _coerce(seq)
        return min(data) if data else 0.0

    def _max(seq):
        data = _coerce(seq)
        return max(data) if data else 0.0

    def _median(seq):
        data = _coerce(seq)
        return median(data) if data else 0.0

    def _percentile(seq, percentile):
        data = sorted(_coerce(seq))
        if not data:
            return 0.0
        k = (len(data) - 1) * (percentile / 100.0)
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return float(data[int(k)])
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return float(d0 + d1)

    module.mean = _mean  # type: ignore[attr-defined]
    module.std = _std  # type: ignore[attr-defined]
    module.min = _min  # type: ignore[attr-defined]
    module.max = _max  # type: ignore[attr-defined]
    module.median = _median  # type: ignore[attr-defined]
    module.percentile = _percentile  # type: ignore[attr-defined]

    sys.modules["numpy"] = module


_install_structlog_stub()
_install_numpy_stub()

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]

if not hasattr(typing, "Unpack"):
    from typing import Any

    typing.Unpack = Any  # type: ignore[attr-defined,assignment]


def test_operational_event_bus_module_removed() -> None:
    sys.modules.pop("src.operational.event_bus", None)
    sys.modules.pop("src.operational", None)
    sys.modules.pop("src.core.event_bus", None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("src.operational.event_bus")

    message = str(excinfo.value)
    assert "src.operational.event_bus" in message
