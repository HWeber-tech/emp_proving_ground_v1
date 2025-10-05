"""Helpers for persisting ingest events to columnar Parquet storage."""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence


logger = logging.getLogger(__name__)


class _DataFrameFactory(Protocol):
    def __call__(self, events: Sequence[dict[str, Any]]) -> "_SupportsParquet":
        """Create a DataFrame-like object that supports ``to_parquet``."""


class _SupportsParquet(Protocol):
    def to_parquet(self, path: str, *, index: bool) -> None:
        """Persist a DataFrame to the parquet ``path``."""


_pd: object | None = None
try:  # pragma: no cover - optional dependency
    _pd = importlib.import_module("pandas")
except ImportError:  # pragma: no cover - optional dependency
    _pd = None


def _resolve_dataframe_factory() -> _DataFrameFactory | None:
    if _pd is None:
        return None
    factory: Callable[..., Any] | None = getattr(_pd, "DataFrame", None)
    if factory is None or not callable(factory):
        logger.error("pandas is available but DataFrame constructor is missing")
        return None
    return factory  # type: ignore[return-value]


def write_events_parquet(
    events: Iterable[dict[str, Any]], out_dir: str, partition: str
) -> str:
    """Write ``events`` to a Parquet file under ``out_dir`` and return the file path.

    Returns an empty string when pandas is unavailable or when the payload cannot
    be serialised. Error conditions are logged so ingest operators can diagnose
    failures instead of silently losing telemetry.
    """

    events_list = list(events)

    if not events_list:
        logger.debug("Skipping Parquet write because no events were supplied")
        return ""

    dataframe_factory = _resolve_dataframe_factory()
    if dataframe_factory is None:
        logger.debug("Skipping Parquet write because pandas DataFrame is unavailable")
        return ""

    try:
        dataframe = dataframe_factory(events_list)
    except (TypeError, ValueError):
        logger.exception("Failed to convert ingest events into a pandas DataFrame")
        return ""

    out_path = Path(out_dir)
    partition_path = out_path / f"partition={partition}"
    partition_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = partition_path / f"events_{timestamp}.parquet"

    try:
        dataframe.to_parquet(str(file_path), index=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.exception("Failed to persist ingest events to Parquet")
        return ""

    return str(file_path)
