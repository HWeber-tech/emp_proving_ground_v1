"""Assemble reproducible evidence packets for final dry run reviews."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.operations.dry_run_audit import DryRunSignOffReport, DryRunSummary


@dataclass(frozen=True)
class DryRunPacketPaths:
    """Paths of artifacts included in a dry run evidence packet."""

    output_dir: Path
    summary_json: Path
    summary_markdown: Path
    manifest_json: Path
    sign_off_json: Path | None
    archive_path: Path | None
    raw_artifacts: tuple[Path, ...]

    def as_dict(self) -> Mapping[str, str | None | list[str]]:
        return {
            "output_dir": str(self.output_dir),
            "summary_json": str(self.summary_json),
            "summary_markdown": str(self.summary_markdown),
            "manifest_json": str(self.manifest_json),
            "sign_off_json": str(self.sign_off_json) if self.sign_off_json else None,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "raw_artifacts": [str(path) for path in self.raw_artifacts],
        }


def write_dry_run_packet(
    *,
    summary: DryRunSummary,
    output_dir: Path,
    sign_off_report: DryRunSignOffReport | None = None,
    log_paths: Sequence[Path] = (),
    diary_path: Path | None = None,
    performance_path: Path | None = None,
    include_raw_artifacts: bool = True,
    archive_path: Path | None = None,
) -> DryRunPacketPaths:
    """Materialise a dry run evidence packet on disk.

    The packet bundles JSON and Markdown summaries alongside optional raw
    telemetry so review boards can inspect the exact artefacts referenced
    during sign-off. When ``archive_path`` is provided, the directory is
    additionally compressed into a ``.tar.gz`` archive for distribution.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    copied_artifacts: list[Path] = []

    if include_raw_artifacts:
        raw_dir.mkdir(exist_ok=True)
        copied_artifacts.extend(
            _copy_artifacts(raw_dir / "logs", log_paths)
        )
        if diary_path is not None:
            copied_artifacts.extend(
                _copy_artifacts(raw_dir, [diary_path], target_name="decision_diary.json")
            )
        if performance_path is not None:
            copied_artifacts.extend(
                _copy_artifacts(
                    raw_dir,
                    [performance_path],
                    target_name="performance_metrics.json",
                )
            )

    summary_payload = summary.as_dict()
    if sign_off_report is not None:
        summary_payload["sign_off"] = sign_off_report.as_dict()

    summary_json = output_dir / "dry_run_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    markdown_sections = [summary.to_markdown().rstrip("\n")]
    if sign_off_report is not None:
        markdown_sections.append(sign_off_report.to_markdown().rstrip("\n"))
    summary_markdown = output_dir / "dry_run_summary.md"
    summary_markdown.write_text("\n\n".join(markdown_sections) + "\n", encoding="utf-8")

    sign_off_json: Path | None = None
    if sign_off_report is not None:
        sign_off_json = output_dir / "dry_run_sign_off.json"
        sign_off_json.write_text(
            json.dumps(sign_off_report.as_dict(), indent=2), encoding="utf-8"
        )

    manifest_payload = {
        "generated_at": summary.generated_at.astimezone(UTC).isoformat(),
        "summary_json": summary_json.name,
        "summary_markdown": summary_markdown.name,
        "sign_off_json": sign_off_json.name if sign_off_json else None,
        "raw_artifacts": [path.relative_to(output_dir).as_posix() for path in copied_artifacts],
    }
    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    archive_result: Path | None = None
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in [summary_json, summary_markdown, manifest_json]:
                archive.add(path, arcname=path.relative_to(output_dir))
            if sign_off_json is not None:
                archive.add(sign_off_json, arcname=sign_off_json.relative_to(output_dir))
            for artifact in copied_artifacts:
                archive.add(artifact, arcname=artifact.relative_to(output_dir))
        archive_result = archive_path

    return DryRunPacketPaths(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_markdown=summary_markdown,
        manifest_json=manifest_json,
        sign_off_json=sign_off_json,
        archive_path=archive_result,
        raw_artifacts=tuple(copied_artifacts),
    )


def _copy_artifacts(
    destination: Path,
    sources: Iterable[Path],
    *,
    target_name: str | None = None,
) -> list[Path]:
    """Copy artefacts into ``destination`` and return the copied paths."""

    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, source in enumerate(sources):
        if target_name is None:
            target = destination / source.name
        elif index == 0:
            target = destination / target_name
        else:
            stem = Path(target_name).stem
            suffix = Path(target_name).suffix
            target = destination / f"{stem}_{index}{suffix}"
        data = source.read_bytes()
        target.write_bytes(data)
        copied.append(target)
    return copied


__all__ = ["DryRunPacketPaths", "write_dry_run_packet"]
