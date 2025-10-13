from __future__ import annotations

from pathlib import Path

from src.operations.trm_exit_drill import TRMDrillStatus, run_trm_exit_drill


def test_run_trm_exit_drill_generates_markdown(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    diaries = repo_root / "docs/examples/trm_exit_drill_diaries.jsonl"
    schema = repo_root / "interfaces/rim_types.json"

    publish_dir = tmp_path / "suggestions"
    log_dir = tmp_path / "logs"

    report = run_trm_exit_drill(
        diaries_path=diaries,
        schema_path=schema,
        publish_dir=publish_dir,
        log_dir=log_dir,
    )

    assert report.status is TRMDrillStatus.PASS
    assert report.metrics.suggestion_count > 0
    assert report.suggestion_artifact is not None
    assert report.suggestion_artifact.exists()
    assert report.telemetry_log is not None
    assert report.telemetry_log.exists()

    markdown = report.to_markdown()
    assert "TRM milestone exit drill" in markdown
    assert diaries.as_posix() in markdown
    assert str(report.metrics.suggestion_count) in markdown

