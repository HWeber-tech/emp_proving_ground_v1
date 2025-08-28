from __future__ import annotations

import importlib
import os
from datetime import datetime
from typing import Any, Dict, List

_pd: object | None = None
try:  # pragma: no cover
    _pd = importlib.import_module("pandas")
except Exception:  # pragma: no cover
    _pd = None


def write_events_parquet(events: list[dict[str, Any]], out_dir: str, partition: str) -> str:
    """Write a list of dict events to Parquet under out_dir/partition=.../file.parquet.
    Returns the file path (or empty string if pandas not available).
    """
    if not events or _pd is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    part_dir = os.path.join(out_dir, f"partition={partition}")
    os.makedirs(part_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(part_dir, f"events_{ts}.parquet")
    try:
        df = getattr(_pd, "DataFrame")(events)
        df.to_parquet(path, index=False)
        return path
    except Exception:
        return ""
