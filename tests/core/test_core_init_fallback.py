"""Tests covering the sensory organ import behaviour for :mod:`src.core`."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_sensory_import_failure_is_not_silenced(monkeypatch):
    """Regress that missing sensory modules raise import errors instead of shimming."""

    import src.core as core  # Ensure module loads successfully before tampering

    original = sys.modules.get("src.core.sensory_organ")

    stub = types.ModuleType("src.core.sensory_organ")

    def _missing_attr(_name: str):
        raise AttributeError("sensory organ module missing")

    stub.__getattr__ = _missing_attr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.core.sensory_organ", stub)

    with pytest.raises(ImportError):
        importlib.reload(core)

    if original is None:
        sys.modules.pop("src.core.sensory_organ", None)
    else:
        sys.modules["src.core.sensory_organ"] = original
    importlib.reload(core)
