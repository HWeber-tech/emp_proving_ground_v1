from pathlib import Path

import pytest

from tools.cleanup.dead_code_tracker import (
    DeadCodeStatus,
    DeadCodeSummary,
    ShimResolution,
    load_import_map,
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
    assert summary.shim_redirects
    first_redirect = summary.shim_redirects[0]
    assert isinstance(first_redirect, ShimResolution)
    assert first_redirect.target_module is None
    assert summary.status_by_path["src/core/sensory_organ.py"] == DeadCodeStatus.SHIM


def test_shim_detection_skips_protocol_getattr_false_positive(_repo_root: Path) -> None:
    report_path = _repo_root / "docs" / "reports" / "CLEANUP_REPORT.md"
    candidates = parse_cleanup_report(report_path)

    summary = summarise_candidates(candidates, repo_root=_repo_root)

    assert "src/operational/fix_connection_manager.py" in summary.present
    assert "src/operational/fix_connection_manager.py" not in summary.shim_exports
    assert summary.status_by_path["src/operational/fix_connection_manager.py"] == DeadCodeStatus.ACTIVE


def test_summarise_candidates_uses_import_map(_repo_root: Path) -> None:
    report_path = _repo_root / "docs" / "reports" / "CLEANUP_REPORT.md"
    candidates = parse_cleanup_report(report_path)

    import_map = load_import_map(
        _repo_root / "docs" / "development" / "import_rewrite_map.yaml"
    )

    summary = summarise_candidates(
        candidates,
        repo_root=_repo_root,
        import_map=import_map,
    )

    redirect = next(
        (entry for entry in summary.shim_redirects if entry.path == "src/core/sensory_organ.py"),
        None,
    )
    assert redirect is not None
    # Sensory organ shim should map to the canonical organ module.
    assert redirect.target_module == "src.sensory.organs.dimensions.base_organ"
    assert redirect.target_exists is True


def test_module_not_found_stubs_flagged_for_triage(_repo_root: Path) -> None:
    report_path = _repo_root / "docs" / "reports" / "CLEANUP_REPORT.md"
    candidates = parse_cleanup_report(report_path)

    summary = summarise_candidates(candidates, repo_root=_repo_root)

    retired_stubs = {
        "src/core/configuration.py",
        "src/core/risk/manager.py",
        "src/core/risk/position_sizing.py",
        "src/thinking/memory/faiss_memory.py",
        "src/thinking/learning/real_time_learner.py",
        "src/thinking/sentient_adaptation_engine.py",
    }

    available = retired_stubs & summary.status_by_path.keys()
    assert available, "expected retired stubs to be present in the cleanup report"

    for path in available:
        assert summary.status_by_path[path] == DeadCodeStatus.MODULE_NOT_FOUND_STUB

    # Ensure the summary breakdown counts the retired stubs.
    breakdown = summary.status_breakdown()
    assert breakdown[DeadCodeStatus.MODULE_NOT_FOUND_STUB] >= len(retired_stubs)
