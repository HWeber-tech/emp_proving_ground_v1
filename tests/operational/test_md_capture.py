import importlib

import pytest


def test_md_capture_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="ingest"):
        importlib.import_module("src.operational.md_capture")
