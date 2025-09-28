"""Shared helpers for roadmap evaluation CLIs."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass(frozen=True)
class Requirement:
    """Executable requirement used to score roadmap milestones."""

    label: str
    check: Callable[[Path], bool]

    def evaluate(self, repo_root: Path) -> tuple[bool, str]:
        """Return a tuple describing whether the requirement passes."""

        try:
            ok = bool(self.check(repo_root))
        except Exception as exc:  # pragma: no cover - defensive guard
            return False, f"{self.label} ({exc.__class__.__name__}: {exc})"
        if not ok:
            return False, self.label
        return True, self.label


def repo_root() -> Path:
    """Return the repository root used for roadmap evaluations."""

    return REPO_ROOT


def require_module_attr(module_name: str, attribute: str | None = None) -> Requirement:
    """Create a requirement that verifies a module (and optional attribute) exists."""

    label = module_name if attribute is None else f"{module_name}.{attribute}"

    def check(_: Path) -> bool:
        module = importlib.import_module(module_name)
        if attribute is not None:
            getattr(module, attribute)
        return True

    return Requirement(label=label, check=check)


def require_path(relative_path: str) -> Requirement:
    """Create a requirement that ensures a repository path exists."""

    def check(repo_root: Path) -> bool:
        return (repo_root / relative_path).exists()

    return Requirement(label=relative_path, check=check)


def evaluate_requirements(
    requirements: Sequence[Requirement],
    repo_root: Path,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return evidence and missing requirement labels for a definition."""

    evidence: list[str] = []
    missing: list[str] = []
    for requirement in requirements:
        ok, label = requirement.evaluate(repo_root)
        if ok:
            evidence.append(label)
        else:
            missing.append(label)

    return tuple(evidence), tuple(missing)


__all__ = [
    "Requirement",
    "repo_root",
    "require_module_attr",
    "require_path",
    "evaluate_requirements",
]
