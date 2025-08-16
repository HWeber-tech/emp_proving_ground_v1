from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType

import pytest


def test_sensory_what_canonical_surface_all():
    mod = importlib.import_module("src.sensory.enhanced.what_dimension")
    assert hasattr(mod, "__all__")
    assert "TechnicalRealityEngine" in getattr(mod, "__all__")


def test_sensory_why_canonical_surface_all():
    mod = importlib.import_module("src.sensory.enhanced.why_dimension")
    assert hasattr(mod, "__all__")
    assert "EnhancedFundamentalIntelligenceEngine" in getattr(mod, "__all__")

def test_sensory_when_canonical_surface_all():
    mod = importlib.import_module("src.sensory.enhanced.when_dimension")
    assert hasattr(mod, "__all__")
    assert "ChronalIntelligenceEngine" in getattr(mod, "__all__")


def test_sensory_how_canonical_surface_all():
    mod = importlib.import_module("src.sensory.enhanced.how_dimension")
    assert hasattr(mod, "__all__")
    assert "InstitutionalIntelligenceEngine" in getattr(mod, "__all__")


def test_sensory_anomaly_canonical_surface_all():
    mod = importlib.import_module("src.sensory.enhanced.anomaly_dimension")
    assert hasattr(mod, "__all__")
    assert "AnomalyIntelligenceEngine" in getattr(mod, "__all__")

def test_no_heavy_imports_on_sensory_import(monkeypatch: pytest.MonkeyPatch):
    # Guard builtins.__import__ to forbid heavy deps during import
    forbidden = ("sklearn", "torch", "pandas")

    real_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if any(name == f or name.startswith(f + ".") for f in forbidden):
            raise ImportError(f"Forbidden heavy import during module import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard_import)

    # Ensure fresh import (remove if already imported)
    for mod_name in [
        "src.sensory.enhanced.what_dimension",
        "src.sensory.enhanced.why_dimension",
        "src.sensory.enhanced.when_dimension",
        "src.sensory.enhanced.how_dimension",
        "src.sensory.enhanced.anomaly_dimension",
    ]:
        sys.modules.pop(mod_name, None)

    # Attempt imports should not raise
    importlib.import_module("src.sensory.enhanced.what_dimension")
    importlib.import_module("src.sensory.enhanced.why_dimension")
    importlib.import_module("src.sensory.enhanced.when_dimension")
    importlib.import_module("src.sensory.enhanced.how_dimension")
    importlib.import_module("src.sensory.enhanced.anomaly_dimension")