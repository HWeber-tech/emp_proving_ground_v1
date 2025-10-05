from pathlib import Path

import pytest

from tools.cleanup.dead_code_tracker import (
    DeadCodeSummary,
    parse_cleanup_report,
    summarise_candidates,
)


@pytest.fixture(scope="module")
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_parse_cleanup_report_extracts_paths(_repo_root: Path) -> None:
    report_path = _repo_root / "docs" / "reports" / "CLEANUP_REPORT.md"
    candidates = parse_cleanup_report(report_path)

    assert "src/core/sensory_organ.py" in candidates
    assert all(candidate.startswith("src/") for candidate in candidates)


def test_summarise_candidates_identifies_shims(_repo_root: Path) -> None:
    report_path = _repo_root / "docs" / "reports" / "CLEANUP_REPORT.md"
    candidates = parse_cleanup_report(report_path)

    summary = summarise_candidates(candidates, repo_root=_repo_root)

    assert isinstance(summary, DeadCodeSummary)
    assert "src/core/sensory_organ.py" in summary.present
    assert "src/core/sensory_organ.py" in summary.shim_exports
