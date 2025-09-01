"""
Runtime check for scientific stack integrity.

This module enforces hard requirements for numpy, pandas, and scipy to avoid
silently running in a degraded mode. Import and call assert_scientific_stack()
from entry points that rely on these libraries (e.g., batch jobs, services).
"""

from __future__ import annotations



def _parse(ver: str) -> tuple[int, int, int]:
    parts = (ver.split("+", 1)[0]).split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except Exception:
        # Conservative default: fail the check if version cannot be parsed
        major, minor, patch = (0, 0, 0)
    return major, minor, patch


def _ge(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return a >= b


def assert_scientific_stack() -> None:
    """
    Assert that required scientific libraries are present and at/above minimum versions.

    Policy (Python-dependent lower bounds are expressed in requirements.txt; these are absolute minima):
      - numpy >= 1.26
      - pandas >= 1.5
      - scipy >= 1.11
    """
    try:
        import numpy
        import pandas
        import scipy
    except Exception as ex:  # pragma: no cover - immediate hard error
        raise ImportError(
            "Scientific stack is missing required libraries (numpy/pandas/scipy). "
            "Install using: pip install -r requirements.txt"
        ) from ex

    np_ok = _ge(_parse(getattr(numpy, "__version__", "0.0.0")), (1, 26, 0))
    pd_ok = _ge(_parse(getattr(pandas, "__version__", "0.0.0")), (1, 5, 0))
    sp_ok = _ge(_parse(getattr(scipy, "__version__", "0.0.0")), (1, 11, 0))

    if not (np_ok and pd_ok and sp_ok):  # pragma: no cover - fail-fast
        raise RuntimeError(
            f"Scientific stack version mismatch: "
            f"numpy={getattr(numpy, '__version__', '?')}, "
            f"pandas={getattr(pandas, '__version__', '?')}, "
            f"scipy={getattr(scipy, '__version__', '?')}. "
            f"Required: numpy>=1.26, pandas>=1.5, scipy>=1.11."
        )
