"""Daily self-audit introspection logs.

This module persists daily self-audit entries that capture what the system
"knows", what it "doubts", and what has "changed".  Each entry is chained with a
cryptographic signature so tampering becomes evident.  Logs are written into a
hierarchical ``<root>/<YYYY>/<MM>/<DD>/self_audit.json`` structure to mirror the
rest of the artifact layout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.observability.immutable_audit import compute_audit_signature

logger = logging.getLogger(__name__)

try:  # Python < 3.11 compatibility
    from datetime import UTC as _UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older runtimes
    _UTC = timezone.utc  # type: ignore[assignment]


@dataclass(slots=True, frozen=True)
class SelfAuditEntry:
    """Snapshot of a self-audit observation."""

    recorded_at: datetime
    knows: tuple[str, ...]
    doubts: tuple[str, ...]
    changes: tuple[str, ...]
    signature: str
    previous_signature: str | None = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the entry."""

        payload: dict[str, Any] = {
            "recorded_at": self.recorded_at.astimezone(_UTC).isoformat(),
            "knows": list(self.knows),
            "doubts": list(self.doubts),
            "changes": list(self.changes),
            "signature": self.signature,
        }
        if self.previous_signature:
            payload["previous_signature"] = self.previous_signature
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class SelfAuditLog:
    """Manage daily self-audit report persistence."""

    __slots__ = ("_root", "_filename")

    def __init__(
        self,
        *,
        root: str | Path = Path("artifacts/self_audit"),
        filename: str = "self_audit.json",
    ) -> None:
        self._root = Path(root)
        self._filename = filename

    def record(
        self,
        *,
        knows: Sequence[str] | str,
        doubts: Sequence[str] | str = (),
        changes: Sequence[str] | str = (),
        metadata: Mapping[str, Any] | None = None,
        when: datetime | None = None,
    ) -> SelfAuditEntry:
        """Persist a new self-audit entry and return the recorded snapshot."""

        recorded_at = _normalise_timestamp(when)
        day = recorded_at.date()
        path = self._resolve_path(day)

        log_payload = self._load_day(day, path)
        entries_raw = log_payload.get("entries")
        if isinstance(entries_raw, list):
            entries: list[dict[str, Any]] = entries_raw
        else:
            entries = []
            log_payload["entries"] = entries
        previous_signature = _latest_signature(entries)

        entry_metadata = _normalise_metadata(metadata)
        knows_bucket = _normalise_bucket(knows)
        doubts_bucket = _normalise_bucket(doubts)
        changes_bucket = _normalise_bucket(changes)

        signature_payload: dict[str, Any] = {
            "recorded_at": recorded_at.astimezone(_UTC).isoformat(),
            "knows": list(knows_bucket),
            "doubts": list(doubts_bucket),
            "changes": list(changes_bucket),
        }
        if entry_metadata:
            signature_payload["metadata"] = entry_metadata

        signature = compute_audit_signature(
            kind="self_audit_entry",
            payload=signature_payload,
            previous_signature=previous_signature,
        )

        entry = SelfAuditEntry(
            recorded_at=recorded_at,
            knows=knows_bucket,
            doubts=doubts_bucket,
            changes=changes_bucket,
            signature=signature,
            previous_signature=previous_signature,
            metadata=entry_metadata,
        )

        entries.append(entry.as_dict())
        log_payload["date"] = day.isoformat()
        log_payload.setdefault("version", 1)

        self._write_day(path, log_payload)
        logger.debug(
            "Recorded self-audit entry",
            extra={
                "self_audit.path": str(path),
                "self_audit.signature": signature,
                "self_audit.knows": entry.knows,
                "self_audit.doubts": entry.doubts,
                "self_audit.changes": entry.changes,
            },
        )
        return entry

    def _resolve_path(self, day: date) -> Path:
        return self._root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}" / self._filename

    def _load_day(self, day: date, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"version": 1, "date": day.isoformat(), "entries": []}

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load self-audit log; starting fresh",
                exc_info=exc,
                extra={"self_audit.path": str(path)},
            )
            return {"version": 1, "date": day.isoformat(), "entries": []}

        entries_raw = raw.get("entries") if isinstance(raw, Mapping) else None
        entries: list[dict[str, Any]] = []
        if isinstance(entries_raw, list):
            for item in entries_raw:
                if isinstance(item, Mapping):
                    entries.append(dict(item))

        version = raw.get("version") if isinstance(raw, Mapping) else 1
        date_text = raw.get("date") if isinstance(raw, Mapping) else day.isoformat()

        return {
            "version": int(version) if isinstance(version, int) else 1,
            "date": str(date_text) if isinstance(date_text, str) else day.isoformat(),
            "entries": entries,
        }

    def _write_day(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - filesystem edge cases
            logger.exception("Failed to create self-audit directory", extra={"self_audit.path": str(path)})
            raise

        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")


def _normalise_timestamp(candidate: datetime | None) -> datetime:
    if candidate is None:
        return datetime.now(_UTC)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=_UTC)
    return candidate.astimezone(_UTC)


def _normalise_bucket(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = [values]
    else:
        candidates = list(values)

    normalised: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        normalised.append(text)
        seen.add(text)
    return tuple(normalised)


def _normalise_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    return {str(key): value for key, value in metadata.items()}


def _latest_signature(entries: Sequence[Mapping[str, Any]]) -> str | None:
    if not entries:
        return None
    last = entries[-1]
    if not isinstance(last, Mapping):
        return None
    signature = last.get("signature")
    return signature if isinstance(signature, str) and signature.strip() else None


__all__ = ["SelfAuditEntry", "SelfAuditLog"]
