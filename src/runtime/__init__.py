"""Runtime assembly helpers for the EMP Professional Predator.

This package re-exports the primary runtime entrypoints while deferring heavy
imports until the attributes are accessed.  That keeps unit tests lightweight
and avoids importing optional dependencies (for example ``structlog`` or
``aiohttp``) when they are not installed.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

__all__ = [
    "BootstrapRuntime",
    "runtime_cli_main",
    "run_cli",
    "ProfessionalPredatorApp",
    "RuntimeApplication",
    "RuntimeWorkload",
    "build_professional_predator_app",
    "build_professional_runtime_application",
    "FixIntegrationPilot",
    "FixPilotState",
    "build_fix_pilot_runtime_application",
    "run_runtime_application",
    "TaskSupervisor",
]

if TYPE_CHECKING:  # pragma: no cover - used for static analysis only
    from .bootstrap_runtime import BootstrapRuntime
    from .cli import main as runtime_cli_main, run_cli
    from .predator_app import ProfessionalPredatorApp, build_professional_predator_app
    from .runtime_builder import (
        RuntimeApplication,
        RuntimeWorkload,
        build_professional_runtime_application,
    )
    from .runtime_runner import run_runtime_application
    from .task_supervisor import TaskSupervisor


def __getattr__(name: str) -> Any:
    if name == "BootstrapRuntime":
        from .bootstrap_runtime import BootstrapRuntime as _BootstrapRuntime

        return _BootstrapRuntime
    if name in {"runtime_cli_main", "run_cli"}:
        from .cli import main as runtime_cli_main, run_cli

        return {"runtime_cli_main": runtime_cli_main, "run_cli": run_cli}[name]
    if name in {"ProfessionalPredatorApp", "build_professional_predator_app"}:
        from .predator_app import (
            ProfessionalPredatorApp as _ProfessionalPredatorApp,
            build_professional_predator_app as _build_professional_predator_app,
        )

        mapping = {
            "ProfessionalPredatorApp": _ProfessionalPredatorApp,
            "build_professional_predator_app": _build_professional_predator_app,
        }
        return mapping[name]
    if name in {
        "FixIntegrationPilot",
        "FixPilotState",
        "build_fix_pilot_runtime_application",
    }:
        from .fix_pilot import (
            FixIntegrationPilot as _FixIntegrationPilot,
            FixPilotState as _FixPilotState,
            build_fix_pilot_runtime_application as _build_fix_pilot_runtime_application,
        )

        mapping = {
            "FixIntegrationPilot": _FixIntegrationPilot,
            "FixPilotState": _FixPilotState,
            "build_fix_pilot_runtime_application": _build_fix_pilot_runtime_application,
        }

        return mapping[name]
    if name in {
        "RuntimeApplication",
        "RuntimeWorkload",
        "build_professional_runtime_application",
    }:
        from .runtime_builder import (
            RuntimeApplication as _RuntimeApplication,
            RuntimeWorkload as _RuntimeWorkload,
            build_professional_runtime_application as _build_professional_runtime_application,
        )

        mapping = {
            "RuntimeApplication": _RuntimeApplication,
            "RuntimeWorkload": _RuntimeWorkload,
            "build_professional_runtime_application": _build_professional_runtime_application,
        }
        return mapping[name]
    if name == "run_runtime_application":
        from .runtime_runner import run_runtime_application as _run_runtime_application

        return _run_runtime_application
    if name == "TaskSupervisor":
        from .task_supervisor import TaskSupervisor as _TaskSupervisor

        return _TaskSupervisor
    raise AttributeError(name)


def _ensure_real_module(module_name: str) -> None:
    placeholder = sys.modules.get(module_name)
    is_stub = placeholder is not None and getattr(placeholder, "__file__", None) in {None, ""}
    if is_stub:
        sys.modules.pop(module_name, None)
    try:  # pragma: no cover - best-effort import to keep lightweight stubs from shadowing
        importlib.import_module(module_name)
    except Exception:  # pragma: no cover - optional dependencies may be absent
        if is_stub and placeholder is not None:
            sys.modules[module_name] = placeholder


for _module in ("src.runtime.fix_pilot", "src.runtime.runtime_builder"):
    _ensure_real_module(_module)
