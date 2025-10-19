import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.operations.graph_diagnostics import GraphHealthStatus, GraphThresholds
from tools.operations import nightly_graph_diagnostics as job


def test_run_graph_diagnostics_job_emits_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("ALPHATRADE_ARTIFACT_ROOT", archive_root.as_posix())

    run_root = tmp_path / "runs"
    ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    thresholds = GraphThresholds()

    result = job.run_graph_diagnostics_job(run_root=run_root, thresholds=thresholds, timestamp=ts)

    assert result.context.metrics_path.exists()
    assert result.context.snapshot_path.exists()
    assert result.context.dot_path.exists()
    assert result.context.markdown_path.exists()

    summary = json.loads(result.context.metrics_path.read_text(encoding="utf-8"))
    assert summary["evaluation"]["status"] == GraphHealthStatus.ok.value
    assert "degree_histogram" in summary["metrics"]
    assert "tail_index" in summary["metrics"]

    archived = list(archive_root.rglob("graph_metrics.json"))
    assert archived, "expected archived metrics artifact"
    assert all(path.read_text(encoding="utf-8").strip() for path in archived)


def test_main_propagates_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHATRADE_ARTIFACT_ROOT", (tmp_path / "archive").as_posix())

    args = [
        "--run-root",
        (tmp_path / "runs").as_posix(),
        "--min-average-degree",
        "5.0",
    ]

    exit_code = job.main(args)
    assert exit_code == 1
