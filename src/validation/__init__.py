"""Phase 2 Validation Package"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Phase2ValidationSuite"]

# Avoid importing optional heavy dependencies (e.g., psutil) at package import time.
# Provide a lazy attribute to load the suite only when actually accessed.
if TYPE_CHECKING:
    from .phase2_validation_suite import Phase2ValidationSuite  # for type checkers only


def __getattr__(name: str) -> object:
    if name == "Phase2ValidationSuite":
        # Lazy import to prevent ModuleNotFoundError for optional deps during unrelated imports
        from .phase2_validation_suite import Phase2ValidationSuite

        return Phase2ValidationSuite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
