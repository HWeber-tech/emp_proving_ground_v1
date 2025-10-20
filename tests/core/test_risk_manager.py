import importlib.util
import sys
from decimal import Decimal
from pathlib import Path


def _load_risk_module():
    module_name = "_core_risk_port"
    module_path = Path(__file__).resolve().parents[2] / "src" / "core" / "risk.py"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load risk module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_risk_module = _load_risk_module()


def test_noop_risk_manager_accepts_non_negative_quantity() -> None:
    manager = _risk_module.NoOpRiskManager()

    assert manager.validate_position({"quantity": Decimal("5")}, {}, Decimal("1000")) is True


def test_noop_risk_manager_rejects_non_numeric_quantity() -> None:
    manager = _risk_module.NoOpRiskManager()

    assert manager.validate_position({"quantity": "not-a-number"}, {}, 1000.0) is False
