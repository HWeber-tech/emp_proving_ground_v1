from __future__ import annotations

import importlib

import pytest


def test_jsonl_writer_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="(?i)timescale"):
        importlib.import_module("src.data_foundation.persist.jsonl_writer")


def test_jsonl_writer_attribute_guard() -> None:
    from src.data_foundation import persist

    with pytest.raises(ModuleNotFoundError, match="(?i)timescale"):
        _ = persist.jsonl_writer  # attribute access should raise
