"""Persistence helpers for the data foundation layer."""

from __future__ import annotations

from typing import NoReturn

from . import timescale, timescale_reader

_REMOVED_HELPERS = {
    "jsonl_writer": (
        "src.data_foundation.persist.jsonl_writer was removed. Use the Timescale "
        "persistence surfaces under src.data_foundation.persist.timescale."
    ),
    "parquet_writer": (
        "src.data_foundation.persist.parquet_writer was removed. Use the canonical "
        "Timescale exporters instead."
    ),
}

__all__ = ["timescale", "timescale_reader"]


def __getattr__(name: str) -> NoReturn:
    message = _REMOVED_HELPERS.get(name)
    if message is not None:
        raise ModuleNotFoundError(message)
    raise AttributeError(name)
