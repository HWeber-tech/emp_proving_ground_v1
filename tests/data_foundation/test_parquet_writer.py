from __future__ import annotations

import importlib

import pytest


def test_parquet_writer_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="(?i)timescale"):
        importlib.import_module("src.data_foundation.persist.parquet_writer")


def test_parquet_writer_attribute_guard() -> None:
    from src.data_foundation import persist

    with pytest.raises(ModuleNotFoundError, match="(?i)timescale"):
        _ = persist.parquet_writer
