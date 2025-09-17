import importlib
import sys
import types


def test_package_symbols(monkeypatch) -> None:
    # Stub psutil used by src.validation to avoid optional dependency import
    if "psutil" not in sys.modules:
        psutil_stub = types.ModuleType("psutil")
        monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    di = importlib.import_module("src.data_integration")
    assert hasattr(di, "ValidationResult")
    assert hasattr(di, "ADVANCED_PROVIDERS_AVAILABLE")
    assert di.ADVANCED_PROVIDERS_AVAILABLE is False

    VR = di.ValidationResult  # type: ignore[attr-defined]
    vr = VR("smoke", True, 1.0, 0.5, "u")
    d = vr.to_dict()
    assert d["test_name"] == "smoke"
