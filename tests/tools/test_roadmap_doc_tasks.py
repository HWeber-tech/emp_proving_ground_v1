"""Tests for the roadmap document task extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.roadmap.doc_tasks as doc_tasks


@pytest.fixture()
def sample_doc(tmp_path: Path) -> Path:
    content = """# High-impact roadmap

Intro text that should be ignored.

```bash
- [ ] not-a-real-task
```

## Phase 1 — Trading Core

- [x] Finish execution lifecycle polish
- [ ] Harden nightly reconciliation pipeline

### Workstream 1B

    - [ ] Add remaining VaR guardrails

## Phase 2 — Strategy Expansion

- [ ] Extend adaptive intelligence backlog
"""
    path = tmp_path / "sample.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_document_tasks_parses_sections(sample_doc: Path) -> None:
    tasks = doc_tasks.extract_document_tasks(sample_doc)

    assert len(tasks) == 4

    assert tasks[0].description == "Finish execution lifecycle polish"
    assert tasks[0].checked is True
    assert tasks[0].section == ("High-impact roadmap", "Phase 1 — Trading Core")

    assert tasks[1].description == "Harden nightly reconciliation pipeline"
    assert tasks[1].checked is False
    assert tasks[1].section == ("High-impact roadmap", "Phase 1 — Trading Core")

    assert tasks[2].description == "Add remaining VaR guardrails"
    assert tasks[2].indent > tasks[1].indent
    assert tasks[2].section == (
        "High-impact roadmap",
        "Phase 1 — Trading Core",
        "Workstream 1B",
    )

    assert tasks[3].section == (
        "High-impact roadmap",
        "Phase 2 — Strategy Expansion",
    )


def test_summarise_document_counts_completed_and_remaining(sample_doc: Path) -> None:
    summary = doc_tasks.summarise_document(sample_doc)

    assert summary.total == 4
    assert summary.completed == 1
    assert summary.remaining == 3
    assert len(summary.remaining_tasks()) == 3
    assert len(summary.completed_tasks()) == 1


def test_format_markdown_includes_remaining_and_completed(sample_doc: Path) -> None:
    summary = doc_tasks.summarise_document(sample_doc)
    report = doc_tasks.format_markdown(summary, include_completed=True)

    assert report.startswith("# Roadmap tasks for sample.md")
    assert "- Total tasks: 4" in report
    assert "- [ ] High-impact roadmap › Phase 2 — Strategy Expansion — Extend adaptive intelligence backlog" in report
    assert "## Completed tasks" in report
    assert "- [x] High-impact roadmap › Phase 1 — Trading Core — Finish execution lifecycle polish" in report


def test_format_json_includes_remaining_tasks(sample_doc: Path) -> None:
    summary = doc_tasks.summarise_document(sample_doc)
    payload = doc_tasks.format_json(summary, include_completed=True)

    data = json.loads(payload)
    assert data["total"] == 4
    assert len(data["remaining_tasks"]) == 3
    assert len(data["completed_tasks"]) == 1

