"""Utilities for triaging the roadmap dead-code backlog."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

__all__ = [
    "DeadCodeSummary",
    "ShimResolution",
    "parse_cleanup_report",
    "summarise_candidates",
    "load_import_map",
    "main",
]


_REPORT_PATTERN = re.compile(r"src\\[\\\\\w./]+")


@dataclass(frozen=True)
class DeadCodeSummary:
    """Structured view of the dead-code candidates declared in the cleanup report."""

    total_candidates: int
    present: tuple[str, ...]
    missing: tuple[str, ...]
    shim_exports: tuple[str, ...]
    shim_redirects: tuple["ShimResolution", ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "total_candidates": self.total_candidates,
            "present": list(self.present),
            "missing": list(self.missing),
            "shim_exports": list(self.shim_exports),
            "shim_redirects": [resolution.as_dict() for resolution in self.shim_redirects],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


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

    for candidate in sorted(dict.fromkeys(candidates)):
        path = repo_root / candidate
        if path.exists():
            present.append(candidate)
            if path.is_file() and path.suffix == ".py" and _looks_like_shim(path):
                shim_exports.append(candidate)
                shim_redirects.append(
                    _summarise_redirect(candidate, repo_root=repo_root, import_map=import_map)
                )
        else:
            missing.append(candidate)

    return DeadCodeSummary(
        total_candidates=len(candidates),
        present=tuple(present),
        missing=tuple(missing),
        shim_exports=tuple(shim_exports),
        shim_redirects=tuple(shim_redirects),
    )


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


def _format_summary(summary: DeadCodeSummary) -> str:
    lines = [
        f"Total candidates: {summary.total_candidates}",
        f"Present: {len(summary.present)}",
        f"Missing: {len(summary.missing)}",
        f"Shim exports: {len(summary.shim_exports)}",
    ]
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
