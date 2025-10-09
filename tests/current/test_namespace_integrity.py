"""Regression tests guarding legacy namespace drift."""

from __future__ import annotations

import ast
import pathlib
from typing import Iterable, Iterator, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src", "tools", "scripts")
BANNED_PREFIXES = (
    "src.intelligence",
    "src.market_intelligence",
    "src.thinking.sentient_adaptation_engine",
    "src.thinking.memory.faiss_memory",
    "src.thinking.learning.real_time_learner",
    "src.sensory.organs.yahoo_finance_organ",
)


def _iter_python_files() -> Iterator[pathlib.Path]:
    for dirname in SOURCE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        yield from base.rglob("*.py")


def _module_name_for_path(path: pathlib.Path) -> Tuple[str, str]:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    is_package = parts and parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    module_name = ".".join(parts)
    if is_package:
        package = module_name
    else:
        package = module_name.rpartition(".")[0]
    return module_name, package


def _resolve_relative_import(module: str, level: int, package: str) -> str:
    package_parts = package.split(".") if package else []
    drop = level - 1
    if drop > len(package_parts):
        base_parts: list[str] = []
    else:
        base_parts = package_parts[: len(package_parts) - drop]
    module_parts = module.split(".") if module else []
    parts = [*base_parts, *module_parts]
    return ".".join(part for part in parts if part)


def _extract_imports(path: pathlib.Path) -> Iterable[Tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT).as_posix()
    _, package = _module_name_for_path(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield rel, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = _resolve_relative_import(module, node.level, package)
            yield rel, module


@pytest.mark.guardrail
def test_no_banned_namespace_imports() -> None:
    offenders: list[tuple[str, str, str]] = []
    for path in _iter_python_files():
        for rel, module in _extract_imports(path):
            for prefix in BANNED_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    offenders.append((rel, module, prefix))
                    break

    assert not offenders, "Deprecated namespace imports detected: " + ", ".join(
        f"{rel} -> {module}" for rel, module, _ in sorted(offenders)
    )
