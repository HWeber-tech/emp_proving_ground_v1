"""Build distributable compliance evidence packs with audit log exports."""

from __future__ import annotations

import json
import logging
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from src.governance.audit_logger import AuditLogger
from src.operations.compliance_readiness import ComplianceReadinessSnapshot
from src.operations.regulatory_telemetry import RegulatoryTelemetrySnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplianceArtifactPackPaths:
    """File layout for a generated compliance evidence pack."""

    output_dir: Path
    manifest_json: Path
    audit_log: Path | None
    compliance_readiness_json: Path | None
    compliance_readiness_markdown: Path | None
    regulatory_snapshot_json: Path | None
    archive_path: Path | None

    def as_dict(self) -> Mapping[str, str | None]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_json": str(self.manifest_json),
            "audit_log": str(self.audit_log) if self.audit_log else None,
            "compliance_readiness_json": (
                str(self.compliance_readiness_json)
                if self.compliance_readiness_json
                else None
            ),
            "compliance_readiness_markdown": (
                str(self.compliance_readiness_markdown)
                if self.compliance_readiness_markdown
                else None
            ),
            "regulatory_snapshot_json": (
                str(self.regulatory_snapshot_json)
                if self.regulatory_snapshot_json
                else None
            ),
            "archive_path": str(self.archive_path) if self.archive_path else None,
        }


def build_compliance_artifact_pack(
    *,
    audit_log_path: Path,
    output_dir: Path,
    compliance_snapshot: ComplianceReadinessSnapshot
    | Mapping[str, object]
    | None = None,
    regulatory_snapshot: RegulatoryTelemetrySnapshot
    | Mapping[str, object]
    | None = None,
    metadata: Mapping[str, object] | None = None,
    archive_path: Path | None = None,
) -> ComplianceArtifactPackPaths:
    """Create a compliance evidence bundle with the audit log export."""

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source_audit_log": str(audit_log_path),
        "metadata": dict(metadata or {}),
    }
    generated_files: list[Path] = []

    audit_copy: Path | None = None
    audit_statistics: Mapping[str, object] | None = None
    audit_digest: str | None = None
    audit_bytes = 0
    audit_lines = 0

    if audit_log_path.exists():
        audit_copy = output_dir / "audit_log.jsonl"
        audit_digest_hasher = sha256()
        with audit_log_path.open("rb") as source, audit_copy.open("wb") as target:
            for chunk in iter(lambda: source.readline(), b""):
                target.write(chunk)
                audit_digest_hasher.update(chunk)
                audit_bytes += len(chunk)
                if chunk.strip():
                    audit_lines += 1
        audit_digest = audit_digest_hasher.hexdigest()
        audit_statistics = AuditLogger(log_file=str(audit_log_path)).get_audit_statistics()
        generated_files.append(audit_copy)
    else:
        logger.warning("Audit log not found at %s; pack will omit raw log", audit_log_path)

    manifest_payload["audit_log"] = {
        "exported": audit_copy.name if audit_copy else None,
        "sha256": audit_digest,
        "size_bytes": audit_bytes,
        "line_count": audit_lines if audit_copy else 0,
        "statistics": audit_statistics or {},
    }

    compliance_json_path: Path | None = None
    compliance_md_path: Path | None = None

    if compliance_snapshot is not None:
        payload = _serialise_compliance_snapshot(compliance_snapshot)
        compliance_json_path = output_dir / "compliance_readiness.json"
        compliance_json_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        markdown = _render_compliance_markdown(payload)
        compliance_md_path = output_dir / "compliance_readiness.md"
        compliance_md_path.write_text(markdown + "\n", encoding="utf-8")
        generated_files.extend([compliance_json_path, compliance_md_path])
        manifest_payload["compliance_readiness"] = {
            "summary_json": compliance_json_path.name,
            "summary_markdown": compliance_md_path.name,
            "status": payload.get("status"),
        }
    else:
        manifest_payload["compliance_readiness"] = None

    regulatory_json_path: Path | None = None
    if regulatory_snapshot is not None:
        regulatory_payload = _serialise_regulatory_snapshot(regulatory_snapshot)
        regulatory_json_path = output_dir / "regulatory_telemetry.json"
        regulatory_json_path.write_text(
            json.dumps(regulatory_payload, indent=2),
            encoding="utf-8",
        )
        generated_files.append(regulatory_json_path)
        manifest_payload["regulatory_telemetry"] = {
            "summary_json": regulatory_json_path.name,
            "status": regulatory_payload.get("status"),
            "coverage_ratio": regulatory_payload.get("coverage_ratio"),
        }
    else:
        manifest_payload["regulatory_telemetry"] = None

    manifest_path = output_dir / "manifest.json"
    files_section = sorted(
        path.relative_to(output_dir).as_posix()
        for path in generated_files
        if path.is_file()
    )
    files_section.append(manifest_path.name)
    manifest_payload["files"] = files_section

    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    archive_result: Path | None = None
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    archive.add(file_path, arcname=file_path.relative_to(output_dir))
        archive_result = archive_path
        manifest_payload["archive"] = archive_path.name
        manifest_payload["files"].append(archive_path.name)

        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )

    return ComplianceArtifactPackPaths(
        output_dir=output_dir,
        manifest_json=manifest_path,
        audit_log=audit_copy,
        compliance_readiness_json=compliance_json_path,
        compliance_readiness_markdown=compliance_md_path,
        regulatory_snapshot_json=regulatory_json_path,
        archive_path=archive_result,
    )


def _serialise_compliance_snapshot(
    snapshot: ComplianceReadinessSnapshot | Mapping[str, object]
) -> MutableMapping[str, object]:
    if isinstance(snapshot, ComplianceReadinessSnapshot):
        return snapshot.as_dict()
    if isinstance(snapshot, MutableMapping):
        return dict(snapshot)
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise TypeError("Unsupported compliance snapshot payload")


def _render_compliance_markdown(payload: Mapping[str, object]) -> str:
    components = payload.get("components")
    rows = ["| Component | Status | Summary |", "| --- | --- | --- |"]
    if isinstance(components, Iterable):
        for component in components:
            if not isinstance(component, Mapping):
                continue
            name = str(component.get("name", "unknown"))
            status = str(component.get("status", "unknown")).upper()
            summary = str(component.get("summary", "")).replace("\n", " ").strip()
            rows.append(f"| {name} | {status} | {summary or 'n/a'} |")
    status_label = str(payload.get("status", "UNKNOWN")).upper()
    header = f"**Compliance Readiness:** {status_label}"
    return "\n".join([header, "", *rows])


def _serialise_regulatory_snapshot(
    snapshot: RegulatoryTelemetrySnapshot | Mapping[str, object]
) -> MutableMapping[str, object]:
    if isinstance(snapshot, RegulatoryTelemetrySnapshot):
        return snapshot.as_dict()
    if isinstance(snapshot, MutableMapping):
        return dict(snapshot)
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise TypeError("Unsupported regulatory snapshot payload")


__all__ = [
    "ComplianceArtifactPackPaths",
    "build_compliance_artifact_pack",
]
