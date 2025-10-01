"""Tests for the defensive imports in :mod:`src.core`."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Optional


def test_sensory_fallback_triggers_logging(monkeypatch, caplog):
    """The module should log and provide stubs when the sensory organ import fails."""

    import src.core as core  # Ensure the module is importable before tampering

    original_module: Optional[types.ModuleType] = sys.modules.get("src.core.sensory_organ")

    stub = types.ModuleType("src.core.sensory_organ")

    def _missing_attr(_name: str):
        raise AttributeError("sensory organ module missing")

    stub.__getattr__ = _missing_attr  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "src.core.sensory_organ", stub)

    caplog.clear()
    caplog.set_level("WARNING", logger="src.core")

    reloaded = importlib.reload(core)

    try:
        assert any(
            "Falling back to sensory organ stubs" in record.message for record in caplog.records
        ), "Expected fallback warning log to be emitted."
        assert reloaded.create_sensory_organ() is None
        assert reloaded.WHAT_ORGAN is None
    finally:
        if original_module is None:
            sys.modules.pop("src.core.sensory_organ", None)
        else:
            sys.modules["src.core.sensory_organ"] = original_module
        importlib.reload(core)

