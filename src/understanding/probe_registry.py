"""Probe registry describing ownership and escalation for understanding loop telemetry.

The roadmap calls for a probe registry that documents drift sentries, router
checks, and governance hooks so reviewers inherit clear ownership metadata.
This module provides an append-friendly registry with Markdown/JSON exports used
by CLI tooling and governance checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for runtime compatibility
    UTC = timezone.utc  # type: ignore[assignment]

__all__ = [
    "ProbeDefinition",
    "ProbeRegistry",
]


@dataclass(slots=True, frozen=True)
class ProbeDefinition:
    """Metadata describing a single telemetry probe/alert surface."""

    probe_id: str
    name: str
    description: str
    owner: str
    contact: str
    severity: str = "info"
    runbook: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "probe_id": self.probe_id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "contact": self.contact,
            "severity": self.severity,
            "tags": list(self.tags),
        }
        if self.runbook:
            payload["runbook"] = self.runbook
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProbeDefinition":
        probe_id = str(payload.get("probe_id"))
        name = str(payload.get("name", probe_id))
        description = str(payload.get("description", ""))
        owner = str(payload.get("owner", "unknown"))
        contact = str(payload.get("contact", "unknown"))
        severity = str(payload.get("severity", "info"))
        runbook = payload.get("runbook")
        tags_payload = payload.get("tags", ())
        metadata_payload = payload.get("metadata")

        tags: tuple[str, ...]
        if isinstance(tags_payload, (list, tuple, set)):
            tags = tuple(str(tag) for tag in tags_payload if str(tag).strip())
        else:
            tags = ()

        metadata: Mapping[str, object]
        if isinstance(metadata_payload, Mapping):
            metadata = {str(key): value for key, value in metadata_payload.items()}
        else:
            metadata = {}

        runbook_value = str(runbook).strip() if isinstance(runbook, str) else None
        if runbook_value == "":
            runbook_value = None

        return cls(
            probe_id=probe_id,
            name=name,
            description=description,
            owner=owner,
            contact=contact,
            severity=severity,
            runbook=runbook_value,
            tags=tags,
            metadata=metadata,
        )


class ProbeRegistry:
    """Append-friendly registry backing governance exports and CLI tooling."""

    def __init__(self, *, generated_at: datetime | None = None) -> None:
        self._definitions: MutableMapping[str, ProbeDefinition] = {}
        self._generated_at = (generated_at or datetime.now(tz=UTC))

    @property
    def generated_at(self) -> datetime:
        return self._generated_at

    def register(self, definition: ProbeDefinition) -> None:
        key = definition.probe_id
        if key in self._definitions:
            raise ValueError(f"probe '{key}' already registered")
        self._definitions[key] = definition

    def upsert(self, definition: ProbeDefinition) -> None:
        self._definitions[definition.probe_id] = definition

    def remove(self, probe_id: str) -> None:
        self._definitions.pop(probe_id, None)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "generated_at": self._generated_at.isoformat(),
            "probes": {probe_id: definition.as_dict() for probe_id, definition in sorted(self._definitions.items())},
        }

    def to_markdown(self) -> str:
        header = [
            "# Understanding loop probe registry",
            f"- Generated at: {self._generated_at.isoformat()}",
            "",
            "| Probe ID | Name | Owner | Contact | Severity | Tags |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        rows: list[str] = []
        for probe_id, definition in sorted(self._definitions.items()):
            tags = ", ".join(definition.tags) if definition.tags else "—"
            rows.append(
                f"| {probe_id} | {definition.name} | {definition.owner} | {definition.contact} | "
                f"{definition.severity} | {tags} |"
            )
        if not rows:
            rows.append("| — | — | — | — | — | — |")
        footer: list[str] = []
        for definition in self._definitions.values():
            details = [f"## {definition.probe_id} — {definition.name}"]
            details.append(definition.description or "No description provided.")
            details.append("")
            details.append(f"- Owner: {definition.owner}")
            details.append(f"- Contact: {definition.contact}")
            details.append(f"- Severity: {definition.severity}")
            if definition.runbook:
                details.append(f"- Runbook: {definition.runbook}")
            if definition.tags:
                details.append(f"- Tags: {', '.join(definition.tags)}")
            if definition.metadata:
                details.append("- Metadata:")
                for key, value in sorted(definition.metadata.items()):
                    details.append(f"  - {key}: {value}")
            footer.extend(details + [""])
        return "\n".join(header + rows + [""] + footer).strip() + "\n"

    def __contains__(self, probe_id: str) -> bool:
        return probe_id in self._definitions

    def get(self, probe_id: str) -> ProbeDefinition | None:
        return self._definitions.get(probe_id)

    def definitions(self) -> tuple[ProbeDefinition, ...]:
        return tuple(self._definitions.values())

    @classmethod
    def from_file(cls, path: str | Path) -> "ProbeRegistry":
        data = Path(path).read_text(encoding="utf-8")
        payload = json.loads(data) if data else {}
        if not isinstance(payload, Mapping):
            raise ValueError("Probe registry payload must be a mapping")
        generated_raw = payload.get("generated_at")
        generated_at: datetime | None = None
        if isinstance(generated_raw, str):
            try:
                generated_at = datetime.fromisoformat(generated_raw)
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=UTC)
            except ValueError:
                generated_at = None
        registry = cls(generated_at=generated_at)
        probes_payload = payload.get("probes", {})
        if not isinstance(probes_payload, Mapping):
            raise ValueError("Probe registry 'probes' field must be a mapping")
        for probe_id, definition_payload in probes_payload.items():
            if not isinstance(definition_payload, Mapping):
                continue
            definition = ProbeDefinition.from_dict({"probe_id": probe_id, **definition_payload})
            registry.upsert(definition)
        return registry

    def write(self, path: str | Path) -> None:
        payload = self.as_dict()
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_definitions(
        cls,
        definitions: Iterable[ProbeDefinition],
        *,
        generated_at: datetime | None = None,
    ) -> "ProbeRegistry":
        registry = cls(generated_at=generated_at)
        for definition in definitions:
            registry.upsert(definition)
        return registry
