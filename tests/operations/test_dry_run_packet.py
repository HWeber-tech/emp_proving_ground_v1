from __future__ import annotations

import json
import tarfile
from datetime import timedelta
from pathlib import Path
import importlib.util
import sys


_MODULE_ROOT = Path(__file__).resolve().parents[2] / "src" / "operations"

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "src.operations.dry_run_audit", _MODULE_ROOT / "dry_run_audit.py"
)
assert _AUDIT_SPEC and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules["src.operations.dry_run_audit"] = _AUDIT_MODULE
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)

DryRunStatus = _AUDIT_MODULE.DryRunStatus
assess_sign_off_readiness = _AUDIT_MODULE.assess_sign_off_readiness
evaluate_dry_run = _AUDIT_MODULE.evaluate_dry_run

_PACKET_SPEC = importlib.util.spec_from_file_location(
    "src.operations.dry_run_packet", _MODULE_ROOT / "dry_run_packet.py"
)
assert _PACKET_SPEC and _PACKET_SPEC.loader is not None
_PACKET_MODULE = importlib.util.module_from_spec(_PACKET_SPEC)
sys.modules["src.operations.dry_run_packet"] = _PACKET_MODULE
_PACKET_SPEC.loader.exec_module(_PACKET_MODULE)

write_dry_run_packet = _PACKET_MODULE.write_dry_run_packet


def _write_log(path: Path, timestamps: list[str]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": ts,
                    "level": "INFO",
                    "event": "loop.tick",
                    "message": "tick",
                }
            )
            for ts in timestamps
        )
        + "\n",
        encoding="utf-8",
    )


def test_write_dry_run_packet_materialises_artifacts(tmp_path: Path) -> None:
    log_path = tmp_path / "dry_run.log.jsonl"
    timestamps = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T12:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-02T12:00:00Z",
    ]
    _write_log(log_path, timestamps)

    summary = evaluate_dry_run(
        log_paths=[log_path],
        log_gap_warn=timedelta(days=2),
        log_gap_fail=timedelta(days=3),
    )
    assert summary.log_summary is not None
    assert summary.log_summary.status is DryRunStatus.pass_

    sign_off = assess_sign_off_readiness(
        summary,
        minimum_duration=timedelta(hours=24),
        allow_warnings=True,
    )
    assert sign_off.status is DryRunStatus.pass_

    diary_path = tmp_path / "diary.json"
    diary_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
    performance_path = tmp_path / "performance.json"
    performance_path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")

    output_dir = tmp_path / "packet"
    archive_path = tmp_path / "packet.tar.gz"

    paths = write_dry_run_packet(
        summary=summary,
        output_dir=output_dir,
        sign_off_report=sign_off,
        log_paths=[log_path],
        diary_path=diary_path,
        performance_path=performance_path,
        archive_path=archive_path,
    )

    assert paths.summary_json.exists()
    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["status"] == summary.status.value

    assert paths.summary_markdown.exists()
    text = paths.summary_markdown.read_text(encoding="utf-8")
    assert "Final dry run summary" in text
    assert "Dry run sign-off" in text

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["summary_json"] == paths.summary_json.name
    assert any(artifact.startswith("raw/") for artifact in manifest["raw_artifacts"])

    assert paths.sign_off_json is not None and paths.sign_off_json.exists()
    sign_off_payload = json.loads(paths.sign_off_json.read_text(encoding="utf-8"))
    assert sign_off_payload["status"] == sign_off.status.value

    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
    assert "dry_run_summary.json" in members
    assert any(name.startswith("raw/") for name in members)

    raw_relative = [path.relative_to(output_dir) for path in paths.raw_artifacts]
    assert all(str(rel).startswith("raw/") for rel in raw_relative)


def test_write_dry_run_packet_without_raw(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])

    summary = evaluate_dry_run(
        log_paths=[log_path],
        log_gap_warn=timedelta(days=2),
        log_gap_fail=timedelta(days=3),
    )

    output_dir = tmp_path / "packet"
    paths = write_dry_run_packet(
        summary=summary,
        output_dir=output_dir,
        include_raw_artifacts=False,
    )

    assert not paths.raw_artifacts
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["raw_artifacts"] == []
