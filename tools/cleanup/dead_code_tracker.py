"""Utilities for triaging the roadmap dead-code backlog.

This module parses the cleanup report, classifies the declared candidates, and
surfaces actionable categories so roadmap owners can distinguish between
removed modules, retired shims, and truly orphaned implementations.  The
classification helps shrink the backlog without deleting files that still
document canonical import paths.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

__all__ = [
    "DeadCodeStatus",
    "DeadCodeSummary",
    "ShimResolution",
    "parse_cleanup_report",
    "summarise_candidates",
    "load_import_map",
    "main",
]


_REPORT_PATTERN = re.compile(r"src\\[\\\\\w./]+")


class DeadCodeStatus(str):
    """Categorisation for candidates listed in the cleanup report."""

    MISSING = "missing"
    MODULE_NOT_FOUND_STUB = "module_not_found_stub"
    SHIM = "shim"
    ACTIVE = "active"


_KNOWN_STATUSES: tuple[str, ...] = (
    DeadCodeStatus.MISSING,
    DeadCodeStatus.MODULE_NOT_FOUND_STUB,
    DeadCodeStatus.SHIM,
    DeadCodeStatus.ACTIVE,
)


@dataclass(frozen=True)
class DeadCodeSummary:
    """Structured view of the dead-code candidates declared in the cleanup report."""

    total_candidates: int
    present: tuple[str, ...]
    missing: tuple[str, ...]
    shim_exports: tuple[str, ...]
    shim_redirects: tuple["ShimResolution", ...]
    _status_by_path: Mapping[str, DeadCodeStatus]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "total_candidates": self.total_candidates,
            "present": list(self.present),
            "missing": list(self.missing),
            "shim_exports": list(self.shim_exports),
            "shim_redirects": [resolution.as_dict() for resolution in self.shim_redirects],
            "status_by_path": {path: status for path, status in self.status_by_path.items()},
            "status_breakdown": self.status_breakdown(),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    @property
    def status_by_path(self) -> Mapping[str, DeadCodeStatus]:
        return self._status_by_path

    def status_breakdown(self) -> Mapping[str, int]:
        counts = Counter(self._status_by_path.values())
        return {status: counts.get(status, 0) for status in _KNOWN_STATUSES}


@dataclass(frozen=True)
class ShimResolution:
    """Represents the canonical landing zone for a detected shim module."""

    path: str
    module: str
    target_module: str | None
    target_exists: bool

    def as_dict(self) -> Mapping[str, object]:
        return {
            "path": self.path,
            "module": self.module,
            "target_module": self.target_module,
            "target_exists": self.target_exists,
        }


def parse_cleanup_report(report_path: Path) -> list[str]:
    """Return normalised candidate paths from ``CLEANUP_REPORT.md``."""

    text = report_path.read_text(encoding="utf-8")
    matches = _REPORT_PATTERN.findall(text)
    candidates: list[str] = []
    for match in matches:
        normalised = match.replace("\\", "/")
        normalised = re.sub(r"/{2,}", "/", normalised)
        normalised = normalised.strip("~ ")
        if normalised.endswith("."):
            normalised = normalised[:-1]
        if normalised not in candidates:
            candidates.append(normalised)
    return candidates


def summarise_candidates(
    candidates: Sequence[str],
    *,
    repo_root: Path,
    import_map: Mapping[str, str] | None = None,
) -> DeadCodeSummary:
    """Classify candidate paths and highlight shim exports for triage."""

    present: list[str] = []
    missing: list[str] = []
    shim_exports: list[str] = []
    shim_redirects: list[ShimResolution] = []
    status_by_path: dict[str, DeadCodeStatus] = {}

    for candidate in sorted(dict.fromkeys(candidates)):
        path = repo_root / candidate
        if path.exists():
            if path.is_file() and path.suffix == ".py" and _is_removed_stub(path):
                missing.append(candidate)
                continue

            present.append(candidate)
            status = _classify_present_candidate(path)
            if status == DeadCodeStatus.SHIM:
                shim_exports.append(candidate)
                shim_redirects.append(
                    _summarise_redirect(candidate, repo_root=repo_root, import_map=import_map)
                )
        else:
            status = DeadCodeStatus.MISSING
            missing.append(candidate)

        status_by_path[candidate] = status

    return DeadCodeSummary(
        total_candidates=len(candidates),
        present=tuple(present),
        missing=tuple(missing),
        shim_exports=tuple(shim_exports),
        shim_redirects=tuple(shim_redirects),
        _status_by_path=MappingProxyType(status_by_path),
    )


def _classify_present_candidate(path: Path) -> DeadCodeStatus:
    if not path.is_file() or path.suffix != ".py":
        return DeadCodeStatus.ACTIVE

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return DeadCodeStatus.ACTIVE

    try:
        tree = ast.parse(text)
    except SyntaxError:
        if "ModuleNotFoundError" in text or "ImportError" in text:
            return DeadCodeStatus.MODULE_NOT_FOUND_STUB
        return DeadCodeStatus.ACTIVE

    if _is_module_not_found_stub(tree):
        return DeadCodeStatus.MODULE_NOT_FOUND_STUB

    if _looks_like_shim(path):
        return DeadCodeStatus.SHIM

    return DeadCodeStatus.ACTIVE


def _looks_like_shim(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fallback to textual heuristic when the file cannot be parsed.
        return "shim" in text.lower()

    docstring = ast.get_docstring(tree)
    if docstring and "shim" in docstring.lower():
        return True

    if _has_module_level_getattr(tree):
        return True

    if _is_reexport_module(tree):
        return True

    if "shim" in text.lower():
        return True

    return False


def _is_module_not_found_stub(tree: ast.AST) -> bool:
    body = getattr(tree, "body", [])

    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import):
            continue
        if isinstance(node, ast.Assign):
            if all(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "__all__":
                continue
        if isinstance(node, ast.Pass):
            continue

        if isinstance(node, ast.Raise):
            return _is_module_not_found_raise(node)

        return False

    return False


def _is_module_not_found_raise(node: ast.Raise) -> bool:
    exc = node.exc
    if exc is None:
        return False
    return _exception_is_module_not_found(exc)


def _exception_is_module_not_found(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        return _exception_is_module_not_found(node.func)
    if isinstance(node, ast.Name):
        return node.id in {"ModuleNotFoundError", "ImportError"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"ModuleNotFoundError", "ImportError"}
    return False


def _has_module_level_getattr(tree: ast.AST) -> bool:
    body = getattr(tree, "body", [])
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            return True
    return False


def _is_reexport_module(tree: ast.AST) -> bool:
    """Detect modules that only re-export symbols from other modules."""

    body = getattr(tree, "body", [])
    has_import = False

    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Docstring or string literal – ignore.
            continue
        if isinstance(node, ast.ImportFrom):
            has_import = True
            continue
        if isinstance(node, ast.Assign):
            if not all(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                return False
            continue
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if not isinstance(target, ast.Name) or target.id != "__all__":
                return False
            continue
        if isinstance(node, ast.Pass):
            continue
        # Any other top-level statement means this is not a pure re-export shim.
        return False

    return has_import


def _is_removed_stub(path: Path) -> bool:
    """Return ``True`` when a module only raises ``ModuleNotFoundError``."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False

    relevant: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            # Module docstring – ignore.
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            # ``from __future__ import annotations`` is common in stubs.
            continue
        if isinstance(node, ast.Import) and all(alias.name == "__future__" for alias in node.names):
            continue
        relevant.append(node)

    if len(relevant) != 1:
        return False

    raise_stmt = relevant[0]
    if not isinstance(raise_stmt, ast.Raise) or raise_stmt.exc is None:
        return False

    exc = raise_stmt.exc
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "ModuleNotFoundError"
    if isinstance(exc, ast.Name):
        return exc.id == "ModuleNotFoundError"
    return False


def _format_summary(summary: DeadCodeSummary) -> str:
    lines = [
        f"Total candidates: {summary.total_candidates}",
        f"Present: {len(summary.present)}",
        f"Missing: {len(summary.missing)}",
        f"Shim exports: {len(summary.shim_exports)}",
    ]
    breakdown = summary.status_breakdown()
    if any(breakdown.values()):
        lines.append("Status breakdown:")
        for status, count in breakdown.items():
            lines.append(f"  - {status}: {count}")
    if summary.shim_exports:
        lines.append("Shim modules:")
        lines.extend(f"  - {path}" for path in summary.shim_exports)
        if summary.shim_redirects:
            lines.append("Shim redirects:")
            for resolution in summary.shim_redirects:
                target = resolution.target_module or "(no canonical mapping)"
                status = "exists" if resolution.target_exists else "missing"
                lines.append(
                    f"  - {resolution.path} → {target} [{status}]"
                )
    if summary.missing:
        lines.append("Missing modules:")
        lines.extend(f"  - {path}" for path in summary.missing)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices={"text", "json"},
        default="text",
        help="Output format (text or json).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/reports/CLEANUP_REPORT.md"),
        help="Path to the cleanup report markdown file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the candidate modules.",
    )
    parser.add_argument(
        "--import-map",
        type=Path,
        default=None,
        help=(
            "Optional path to the import rewrite map; when provided, shim modules "
            "are linked to their canonical targets."
        ),
    )
    args = parser.parse_args(argv)

    candidates = parse_cleanup_report(args.report)
    import_map = None
    if args.import_map:
        import_map = load_import_map(args.import_map)

    summary = summarise_candidates(candidates, repo_root=args.root, import_map=import_map)

    if args.format == "json":
        print(summary.to_json())
    else:
        print(_format_summary(summary))
    return 0


def load_import_map(map_path: Path) -> Mapping[str, str] | None:
    """Load legacy→canonical module mapping if the file is present."""

    if not map_path.exists():
        return None

    text = map_path.read_text(encoding="utf-8")

    parsed: Mapping[str, object] | None = None
    if json is not None:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    if parsed is None and yaml is not None:
        try:
            parsed = yaml.safe_load(text)  # type: ignore[assignment]
        except Exception:
            parsed = None

    if parsed is None:
        return None

    mappings = parsed.get("mappings")  # type: ignore[call-arg]
    if not isinstance(mappings, list):
        return {}

    lookup: dict[str, str] = {}
    for entry in mappings:
        if not isinstance(entry, MappingABC):
            continue
        target = entry.get("target_module")
        sources = entry.get("sources")
        if not isinstance(target, str) or not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, str) and source:
                lookup[source] = target
    return lookup


def _summarise_redirect(
    candidate: str,
    *,
    repo_root: Path,
    import_map: Mapping[str, str] | None,
) -> "ShimResolution":
    module = _path_to_module(candidate)
    target_module: str | None = None

    search_keys = {module}
    if module.startswith("src."):
        search_keys.add(module.removeprefix("src."))

    if import_map:
        for key in search_keys:
            if key in import_map:
                target_module = import_map[key]
                break

    target_exists = _module_path_exists(repo_root, target_module) if target_module else False

    return ShimResolution(
        path=candidate,
        module=module,
        target_module=target_module,
        target_exists=target_exists,
    )


def _path_to_module(candidate: str) -> str:
    module = candidate
    if module.endswith("/__init__.py"):
        module = module[: -len("/__init__.py")]
    elif module.endswith(".py"):
        module = module[: -len(".py")]
    return module.replace("/", ".")


def _module_path_exists(repo_root: Path, module: str | None) -> bool:
    if not module:
        return False
    path = repo_root / module.replace(".", "/")
    if path.with_suffix(".py").exists():
        return True
    return (path / "__init__.py").exists()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
