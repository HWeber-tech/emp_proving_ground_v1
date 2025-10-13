from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_smoke import SmokeRunOptions, run_smoke_workflow


def test_final_dry_run_smoke_workflow(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    review_md = tmp_path / "review.md"
    review_json = tmp_path / "review.json"
    packet_dir = tmp_path / "packet"
    packet_archive = tmp_path / "packet.tar.gz"

    options = SmokeRunOptions(
        output_dir=output_dir,
        duration=timedelta(seconds=4),
        tick_interval=timedelta(seconds=0.5),
        progress_interval=timedelta(seconds=0.5),
        minimum_uptime_ratio=0.9,
        allow_warnings=False,
        metadata={"suite": "unit"},
        packet_dir=packet_dir,
        packet_archive=packet_archive,
        review_markdown=review_md,
        review_json=review_json,
        review_notes=("Validated via unit test",),
    )

    result = run_smoke_workflow(options)

    assert result.run_result.summary.status is DryRunStatus.pass_
    assert result.run_result.sign_off is not None
    assert result.run_result.sign_off.status is DryRunStatus.pass_
    assert (output_dir / "decision_diary.jsonl").exists()
    assert (output_dir / "performance_metrics.json").exists()
    assert review_md.read_text(encoding="utf-8").startswith("# Final Dry Run Review")
    assert review_json.read_text(encoding="utf-8").strip().startswith("{")
    assert DryRunStatus.pass_ is result.review.status
    assert packet_dir.exists()
    assert packet_archive.exists()
