"""
Runtime check for scientific stack integrity.

This module enforces hard requirements for numpy, pandas, and scipy to avoid
silently running in a degraded mode. Import and call assert_scientific_stack()
from entry points that rely on these libraries (e.g., batch jobs, services).
"""

from __future__ import annotations

import importlib
import re
import sys
from typing import Dict, Tuple


MINIMUM_VERSIONS: Dict[str, Tuple[int, int, int]] = {
    "numpy": (1, 26, 0),
    "pandas": (1, 5, 0),
    "scipy": (1, 11, 0),
}


def _parse(ver: str) -> tuple[int, int, int]:
    parts = (ver.split("+", 1)[0]).split(".")

    def _coerce(part: str) -> int:
        match = re.match(r"\d+", part)
        if not match:
            return 0
        try:
            return int(match.group(0))
        except ValueError:
            return 0

    parsed = [_coerce(part) for part in parts[:3]]
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed)


def _format_version(parts: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in parts)


def _ge(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return a >= b


def check_scientific_stack() -> dict[str, str]:
    """Validate required scientific libraries and return their versions."""

    versions: dict[str, str] = {}
    missing: list[str] = []
    outdated: list[tuple[str, str, tuple[int, int, int]]] = []

    for package, minimum in MINIMUM_VERSIONS.items():
        try:
            module = importlib.import_module(package)
        except Exception as ex:  # pragma: no cover - immediate hard error
            missing.append(f"{package} ({ex})")
            continue

        version = getattr(module, "__version__", "0.0.0")
        versions[package] = version
        if not _ge(_parse(version), minimum):
            outdated.append((package, version, minimum))

    if missing:
        message = [
            "Scientific stack is missing required libraries:",
            *(f"- {item}" for item in missing),
            "Install using: pip install -r requirements/base.txt",
        ]
        raise ImportError("\n".join(message))

    if outdated:
        message = ["Scientific stack version mismatch detected:"]
        for package, version, minimum in outdated:
            message.append(f"- {package}: found {version}, requires >= {_format_version(minimum)}")
        raise RuntimeError("\n".join(message))

    return versions


def assert_scientific_stack() -> None:
    """
    Assert that required scientific libraries are present and at/above minimum versions.

    Policy (Python-dependent lower bounds are expressed in requirements/base.txt; these are absolute minima):
      - numpy >= 1.26
      - pandas >= 1.5
      - scipy >= 1.11
    """

    check_scientific_stack()


def main() -> int:
    try:
        versions = check_scientific_stack()
    except ImportError as exc:  # pragma: no cover - CLI convenience
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:  # pragma: no cover - CLI convenience
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Scientific stack OK:")
    for package in sorted(versions):
        print(f"  - {package} {versions[package]}")
    print("Ensure these versions stay in sync with requirements/base.txt when upgrading the stack.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
