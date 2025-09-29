"""Extract checklist tasks from the High-impact roadmap documents."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ._shared import repo_root


_CHECKBOX_PATTERN = re.compile(
    r"^(?P<indent>\s*)[-*]\s*\[(?P<mark>[ xX])\]\s+(?P<body>.+?)\s*$"
)


@dataclass(frozen=True)
class RoadmapTask:
    """Representation of a single checklist entry in the roadmap document."""

    description: str
    checked: bool
    line_number: int
    indent: int
    section: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable payload for the task."""

        return {
            "description": self.description,
            "checked": self.checked,
            "line": self.line_number,
            "indent": self.indent,
            "section": list(self.section),
        }


@dataclass(frozen=True)
class RoadmapSummary:
    """Aggregated view across all roadmap tasks in a document."""

    document: Path
    tasks: tuple[RoadmapTask, ...]
    total: int
    completed: int
    remaining: int

    @classmethod
    def from_tasks(cls, document: Path, tasks: Sequence[RoadmapTask]) -> "RoadmapSummary":
        """Create a summary from the parsed tasks."""

        task_tuple = tuple(tasks)
        completed = sum(1 for task in task_tuple if task.checked)
        total = len(task_tuple)
        remaining = total - completed
        return cls(
            document=document,
            tasks=task_tuple,
            total=total,
            completed=completed,
            remaining=remaining,
        )

    def remaining_tasks(self) -> tuple[RoadmapTask, ...]:
        """Return the outstanding roadmap tasks."""

        return tuple(task for task in self.tasks if not task.checked)

    def completed_tasks(self) -> tuple[RoadmapTask, ...]:
        """Return the completed roadmap tasks."""

        return tuple(task for task in self.tasks if task.checked)


def _read_document(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def extract_document_tasks(path: Path) -> tuple[RoadmapTask, ...]:
    """Parse the roadmap document and return all checklist entries."""

    headings: list[str] = []
    tasks: list[RoadmapTask] = []
    in_code_block = False

    for line_number, line in enumerate(_read_document(path), start=1):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped[level:].strip()
            if heading_text:
                while len(headings) >= level:
                    headings.pop()
                headings.append(heading_text)
            continue

        match = _CHECKBOX_PATTERN.match(line)
        if not match:
            continue

        mark = match.group("mark").lower()
        description = match.group("body").strip()
        indent = len(match.group("indent"))

        tasks.append(
            RoadmapTask(
                description=description,
                checked=mark == "x",
                line_number=line_number,
                indent=indent,
                section=tuple(headings),
            )
        )

    return tuple(tasks)


def summarise_document(path: Path) -> RoadmapSummary:
    """Return a :class:`RoadmapSummary` for the given roadmap document."""

    tasks = extract_document_tasks(path)
    return RoadmapSummary.from_tasks(path, tasks)


def _format_task_entry(task: RoadmapTask) -> str:
    context = " › ".join(task.section) if task.section else "Document"
    status = "[x]" if task.checked else "[ ]"
    return f"- {status} {context} — {task.description} (line {task.line_number})"


def format_markdown(summary: RoadmapSummary, *, include_completed: bool = False) -> str:
    """Render the roadmap task summary as Markdown."""

    lines = [
        f"# Roadmap tasks for {summary.document.name}",
        "",
        f"- Total tasks: {summary.total}",
        f"- Completed: {summary.completed}",
        f"- Remaining: {summary.remaining}",
        "",
        "## Remaining tasks",
    ]

    remaining_tasks = summary.remaining_tasks()
    if remaining_tasks:
        lines.extend(_format_task_entry(task) for task in remaining_tasks)
    else:
        lines.append("- [x] None (all tasks complete)")

    if include_completed:
        lines.append("")
        lines.append("## Completed tasks")
        completed_tasks = summary.completed_tasks()
        if completed_tasks:
            lines.extend(_format_task_entry(task) for task in completed_tasks)
        else:
            lines.append("- [ ] None yet")

    return "\n".join(lines)


def format_json(summary: RoadmapSummary, *, include_completed: bool = False) -> str:
    """Render the roadmap task summary as JSON."""

    payload: dict[str, object] = {
        "document": str(summary.document),
        "total": summary.total,
        "completed": summary.completed,
        "remaining": summary.remaining,
        "remaining_tasks": [task.as_dict() for task in summary.remaining_tasks()],
    }

    if include_completed:
        payload["completed_tasks"] = [
            task.as_dict() for task in summary.completed_tasks()
        ]

    return json.dumps(payload, indent=2)


def _default_document() -> Path:
    return repo_root() / "docs/High-Impact Development Roadmap.md"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for roadmap document task extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document",
        type=Path,
        default=_default_document(),
        help="Path to the roadmap markdown document",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include completed tasks in the report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the report to",
    )
    args = parser.parse_args(argv)

    summary = summarise_document(args.document)

    if args.format == "json":
        output = format_json(summary, include_completed=args.include_completed)
    else:
        output = format_markdown(summary, include_completed=args.include_completed)

    if args.output:
        if not output.endswith("\n"):
            output = f"{output}\n"
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


__all__ = [
    "RoadmapTask",
    "RoadmapSummary",
    "extract_document_tasks",
    "summarise_document",
    "format_markdown",
    "format_json",
    "main",
]

