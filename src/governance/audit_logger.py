"""
EMP Audit Logger v1.1

Audit logging for the governance layer.
Tracks all governance decisions and actions for compliance and transparency.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _safe_parse_timestamp(value: object, *, context: str = "audit entry") -> datetime | None:
    """Best-effort parsing that preserves log access despite corrupt payloads."""

    if value in (None, ""):
        return None

    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        logger.warning("%s has invalid timestamp %r: %s", context, value, exc)
        return None


@dataclass(frozen=True)
class AuditIntegrityViolation:
    """Structured representation of audit integrity issues."""

    line_number: int
    reason: str


_MISSING = object()


class AuditLogger:
    """Audit logger for governance layer actions with tamper evidence."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        *,
        enable_integrity: bool = True,
        _integrity_serializer: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.log_file = log_file or "data/audit_log.jsonl"
        self.log_path = Path(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._enable_integrity = enable_integrity
        self._integrity_serializer = (
            _integrity_serializer or _canonicalize_for_integrity
        )
        self._last_entry_hash: str | None = None
        if self._enable_integrity:
            self._last_entry_hash = self._load_last_entry_hash()

        logger.info(
            "Audit Logger initialized with log file: %s (integrity=%s)",
            self.log_file,
            "enabled" if self._enable_integrity else "disabled",
        )

    def log_decision(
        self,
        decision_type: str,
        strategy_id: str,
        genome_id: str,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a governance decision."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "governance_decision",
            "decision_type": decision_type,
            "strategy_id": strategy_id,
            "genome_id": genome_id,
            "approver": approver,
            "reason": reason,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Governance decision logged: {decision_type} for {strategy_id}")

    def log_fitness_evaluation(
        self,
        genome_id: str,
        strategy_id: str,
        fitness_score: float,
        generation: int,
        performance_metrics: dict[str, Any],
        risk_metrics: dict[str, Any],
    ) -> None:
        """Log a fitness evaluation."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "fitness_evaluation",
            "genome_id": genome_id,
            "strategy_id": strategy_id,
            "fitness_score": fitness_score,
            "generation": generation,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
        }

        self._write_log_entry(audit_entry)
        logger.debug(f"Fitness evaluation logged: {fitness_score:.4f} for {genome_id}")

    def log_strategy_registration(
        self,
        strategy_id: str,
        genome_id: str,
        registration_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a strategy registration event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "strategy_registration",
            "strategy_id": strategy_id,
            "genome_id": genome_id,
            "registration_type": registration_type,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Strategy registration logged: {registration_type} for {strategy_id}")

    def log_approval_workflow(
        self,
        workflow_id: str,
        strategy_id: str,
        workflow_type: str,
        status: str,
        approver: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an approval workflow event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "approval_workflow",
            "workflow_id": workflow_id,
            "strategy_id": strategy_id,
            "workflow_type": workflow_type,
            "status": status,
            "approver": approver,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Approval workflow logged: {workflow_type} - {status} for {strategy_id}")

    def log_system_event(
        self,
        event_type: str,
        component: str,
        severity: str,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a system event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "system_event",
            "system_event_type": event_type,
            "component": component,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)

        # Log to console based on severity
        if severity == "critical":
            logger.critical(f"System event: {message}")
        elif severity == "error":
            logger.error(f"System event: {message}")
        elif severity == "warning":
            logger.warning(f"System event: {message}")
        else:
            logger.info(f"System event: {message}")

    def log_compliance_check(
        self,
        check_type: str,
        strategy_id: str,
        passed: bool,
        violations: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a compliance check."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "compliance_check",
            "check_type": check_type,
            "strategy_id": strategy_id,
            "passed": passed,
            "violations": violations,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)

        if passed:
            logger.info(f"Compliance check passed: {check_type} for {strategy_id}")
        else:
            logger.warning(
                f"Compliance check failed: {check_type} for {strategy_id} - violations: {violations}"
            )

    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the audit log file."""
        prepared_entry = dict(entry)
        if self._enable_integrity:
            prepared_entry["integrity"] = self._build_integrity_record(prepared_entry)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(_serialize_entry(prepared_entry))
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")
            return

        if self._enable_integrity:
            self._last_entry_hash = prepared_entry["integrity"]["hash"]

    def get_audit_history(
        self,
        strategy_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get audit history with optional filtering."""
        try:
            if not self.log_path.exists():
                return []

            entries: list[tuple[dict[str, Any], datetime]] = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    payload = line.strip()
                    if not payload:
                        continue

                    try:
                        entry = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in audit log: %s", payload)
                        continue

                    if not isinstance(entry, dict):
                        logger.warning("Ignoring non-object audit entry: %s", payload)
                        continue

                    if strategy_id and entry.get("strategy_id") != strategy_id:
                        continue
                    if event_type and entry.get("event_type") != event_type:
                        continue

                    timestamp_raw = entry.get("timestamp")
                    entry_time: datetime | None
                    if start_time or end_time:
                        entry_time = _safe_parse_timestamp(timestamp_raw)
                        if entry_time is None:
                            # Unable to prove ordering; skip to keep filters reliable.
                            continue
                    else:
                        entry_time = None

                    if start_time and entry_time and entry_time < start_time:
                        continue
                    if end_time and entry_time and entry_time > end_time:
                        continue

                    entries.append(entry)

            if limit:
                entries = entries[-limit:]

            return entries

        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
            return []

    def get_audit_statistics(self) -> dict[str, Any]:
        """Get audit log statistics."""
        try:
            if not self.log_path.exists():
                return {"total_entries": 0, "event_types": {}, "strategies": {}, "date_range": None}

            entries = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    payload = line.strip()
                    if not payload:
                        continue

                    try:
                        entry = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in audit log: %s", payload)
                        continue

                    if not isinstance(entry, dict):
                        logger.warning("Ignoring non-object audit entry: %s", payload)
                        continue

                    parsed_ts = _safe_parse_timestamp(entry.get("timestamp"))
                    if parsed_ts is None:
                        continue

                    entries.append((entry, parsed_ts))

            if not entries:
                return {"total_entries": 0, "event_types": {}, "strategies": {}, "date_range": None}

            # Calculate statistics
            event_types: dict[str, int] = {}
            strategies: dict[str, int] = {}
            timestamps = []

            for entry, parsed_ts in entries:
                event_type = entry.get("event_type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

                strategy_id = entry.get("strategy_id")
                if strategy_id:
                    strategies[strategy_id] = strategies.get(strategy_id, 0) + 1

                timestamps.append(parsed_ts)

            # Calculate date range
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                date_range = {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_days": (end_time - start_time).days,
                }
            else:
                date_range = None

            return {
                "total_entries": len(entries),
                "event_types": event_types,
                "strategies": strategies,
                "date_range": date_range,
            }

        except Exception as e:
            logger.error(f"Error calculating audit statistics: {e}")
            return {
                "total_entries": 0,
                "event_types": {},
                "strategies": {},
                "date_range": None,
                "error": str(e),
            }

    def clear_audit_log(self) -> None:
        """Clear the audit log file."""
        try:
            if self.log_path.exists():
                self.log_path.unlink()
            logger.info("Audit log cleared")
        except Exception as e:
            logger.error(f"Error clearing audit log: {e}")

    def export_audit_log(self, export_file: str) -> None:
        """Export audit log to a different file."""
        try:
            if not self.log_path.exists():
                logger.warning("No audit log to export")
                return

            import shutil

            shutil.copy2(self.log_path, export_file)
            logger.info(f"Audit log exported to: {export_file}")
        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")

    def search_entries(
        self,
        *,
        text: str | Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        strategy_ids: Sequence[str] | None = None,
        metadata_filters: Mapping[str, object] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        match_all_terms: bool = True,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the audit log using structured and free-text criteria.

        Parameters
        ----------
        text:
            A single term or collection of terms that should appear anywhere in a
            serialised representation of the entry. By default all terms must be
            present (case insensitive unless ``case_sensitive`` is true). When
            ``match_all_terms`` is ``False`` at least one term must match.
        event_types:
            Optional collection of event types to match (exact string match).
        strategy_ids:
            Optional collection of strategy identifiers to match.
        metadata_filters:
            Mapping of ``"dot.separated"`` paths to either literal values or
            callables. Literal string comparisons respect ``case_sensitive`` and
            list-valued fields are treated as containing a match when the value is
            present. Callables receive the resolved value and should return a
            truthy value to keep the entry.
        start_time, end_time:
            Optional inclusive bounds applied to the ``timestamp`` field. Entries
            with unparsable timestamps are skipped when date filters are active to
            preserve deterministic filtering.
        limit:
            Maximum number of matching entries to return (the most recent matches
            are kept when the limit is exceeded).
        match_all_terms:
            Whether all text terms must match (``True``) or any single term is
            sufficient (``False``).
        case_sensitive:
            Whether text and string metadata comparisons should honour case.
        """

        if not self.log_path.exists():
            return []

        terms: list[str]
        if text is None:
            terms = []
        elif isinstance(text, str):
            terms = [text]
        else:
            terms = [term for term in text if isinstance(term, str)]

        if not case_sensitive:
            terms = [term.lower() for term in terms]

        event_filter = {evt for evt in event_types} if event_types else None
        strategy_filter = {sid for sid in strategy_ids} if strategy_ids else None
        metadata_filter = dict(metadata_filters or {})

        matches: list[dict[str, Any]] = []

        try:
            with open(self.log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    payload = line.strip()
                    if not payload:
                        continue

                    try:
                        entry = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in audit log: %s", payload)
                        continue

                    if not isinstance(entry, dict):
                        logger.warning("Ignoring non-object audit entry: %s", payload)
                        continue

                    timestamp_raw = entry.get("timestamp")
                    entry_time: datetime | None
                    if start_time or end_time:
                        entry_time = _safe_parse_timestamp(timestamp_raw)
                        if entry_time is None:
                            # Unable to guarantee ordering when date filters are
                            # provided, therefore skip unparsable entries.
                            continue
                    else:
                        entry_time = None

                    if start_time and entry_time and entry_time < start_time:
                        continue
                    if end_time and entry_time and entry_time > end_time:
                        continue

                    if event_filter and entry.get("event_type") not in event_filter:
                        continue
                    if strategy_filter and entry.get("strategy_id") not in strategy_filter:
                        continue

                    if metadata_filter and not _metadata_matches(
                        entry,
                        metadata_filter,
                        case_sensitive=case_sensitive,
                    ):
                        continue

                    if terms:
                        haystack = _build_search_haystack(
                            entry,
                            case_sensitive=case_sensitive,
                        )
                        if match_all_terms:
                            if not all(term in haystack for term in terms):
                                continue
                        elif not any(term in haystack for term in terms):
                            continue

                    matches.append(entry)

            if limit is not None and limit >= 0 and len(matches) > limit:
                matches = matches[-limit:]

            return matches

        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Error searching audit log: %s", exc)
            return []

    def verify_integrity(self) -> dict[str, Any]:
        """Verify the integrity chain of the audit log."""

        if not self._enable_integrity:
            logger.info("Integrity verification skipped because integrity is disabled")
            return {
                "valid": True,
                "checked_entries": 0,
                "violations": [],
                "mode": "disabled",
            }

        if not self.log_path.exists():
            return {"valid": True, "checked_entries": 0, "violations": [], "mode": "empty"}

        violations: list[AuditIntegrityViolation] = []
        previous_hash: str | None = None
        checked_entries = 0

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                payload = line.strip()
                if not payload:
                    continue

                try:
                    entry = json.loads(payload)
                except json.JSONDecodeError as exc:
                    violations.append(
                        AuditIntegrityViolation(line_number=line_number, reason=str(exc))
                    )
                    previous_hash = None
                    continue

                if not isinstance(entry, dict):
                    violations.append(
                        AuditIntegrityViolation(
                            line_number=line_number,
                            reason="entry is not a JSON object",
                        )
                    )
                    previous_hash = None
                    continue

                integrity_meta = entry.get("integrity")
                if not isinstance(integrity_meta, dict):
                    violations.append(
                        AuditIntegrityViolation(
                            line_number=line_number,
                            reason="missing integrity metadata",
                        )
                    )
                    previous_hash = None
                    continue

                recorded_hash = integrity_meta.get("hash")
                recorded_prev = integrity_meta.get("previous_hash")

                expected_prev = previous_hash
                if recorded_prev != expected_prev:
                    violations.append(
                        AuditIntegrityViolation(
                            line_number=line_number,
                            reason=(
                                "previous hash mismatch: expected %r but found %r"
                                % (expected_prev, recorded_prev)
                            ),
                        )
                    )

                computed = self._integrity_serializer(
                    _canonical_integrity_payload(entry, recorded_prev)
                )
                computed_hash = sha256(computed.encode("utf-8")).hexdigest()

                if recorded_hash != computed_hash:
                    violations.append(
                        AuditIntegrityViolation(
                            line_number=line_number,
                            reason="hash mismatch",
                        )
                    )
                    previous_hash = recorded_hash if isinstance(recorded_hash, str) else None
                else:
                    previous_hash = recorded_hash
                    checked_entries += 1

        return {
            "valid": not violations,
            "checked_entries": checked_entries,
            "violations": violations,
            "mode": "verified",
        }

    def _build_integrity_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        payload = _canonical_integrity_payload(entry, self._last_entry_hash)
        serialized = self._integrity_serializer(payload)
        digest = sha256(serialized.encode("utf-8")).hexdigest()
        return {
            "hash": digest,
            "previous_hash": self._last_entry_hash,
            "version": 1,
        }

    def _load_last_entry_hash(self) -> str | None:
        if not self.log_path.exists():
            return None

        try:
            with open(self.log_path, "rb") as f:
                try:
                    f.seek(-4096, 2)
                except OSError:
                    f.seek(0)
                tail = f.read().decode("utf-8", errors="ignore")
        except OSError:
            return None

        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            integrity_meta = entry.get("integrity")
            if isinstance(integrity_meta, dict):
                hash_value = integrity_meta.get("hash")
                if isinstance(hash_value, str):
                    return hash_value
        return None


def _canonical_integrity_payload(
    entry: dict[str, Any], previous_hash: str | None
) -> dict[str, Any]:
    payload = {key: value for key, value in entry.items() if key != "integrity"}
    payload["_integrity_previous"] = previous_hash
    return payload


def _canonicalize_for_integrity(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_stringify)


def _serialize_entry(entry: dict[str, Any]) -> str:
    return json.dumps(entry, separators=(",", ":"), default=_stringify) + "\n"


def _stringify(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _metadata_matches(
    entry: Mapping[str, Any],
    requirements: Mapping[str, object],
    *,
    case_sensitive: bool,
) -> bool:
    for path, expected in requirements.items():
        actual = _resolve_path(entry, path)
        if actual is _MISSING:
            return False

        if callable(expected):
            try:
                if not expected(actual):
                    return False
            except Exception:  # pragma: no cover - defensive, user provided callable
                logger.debug(
                    "Audit metadata filter callable raised; rejecting entry", exc_info=True
                )
                return False
            continue

        if isinstance(actual, str) and isinstance(expected, str) and not case_sensitive:
            if actual.lower() != expected.lower():
                return False
            continue

        if _sequence_contains(actual, expected, case_sensitive=case_sensitive):
            continue

        if actual != expected:
            return False

    return True


def _sequence_contains(value: object, expected: object, *, case_sensitive: bool) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False

    if isinstance(expected, str):
        return any(
            isinstance(item, str) and _compare_strings(item, expected, case_sensitive)
            for item in value
        )

    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        expected_items = list(expected)
        if not expected_items:
            return True
        for candidate in expected_items:
            if isinstance(candidate, str):
                if not _sequence_contains(value, candidate, case_sensitive=case_sensitive):
                    return False
            elif candidate not in value:
                return False
        return True

    return expected in value


def _compare_strings(a: str, b: str, case_sensitive: bool) -> bool:
    return a == b if case_sensitive else a.lower() == b.lower()


def _resolve_path(entry: Mapping[str, Any], path: str) -> object:
    if not path:
        return entry

    current: object = entry
    for component in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(component, _MISSING)
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if not component.isdigit():
                return _MISSING
            index = int(component)
            try:
                current = current[index]
            except (IndexError, TypeError):
                return _MISSING
        else:
            return _MISSING

        if current is _MISSING:
            return _MISSING

    return current


def _build_search_haystack(entry: Mapping[str, Any], *, case_sensitive: bool) -> str:
    parts: list[str] = []

    def _collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                parts.append(str(key))
                _collect(value[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                _collect(item)
        else:
            parts.append(_stringify(value))

    for key in sorted(entry):
        if key == "integrity":
            continue
        parts.append(str(key))
        _collect(entry[key])

    haystack = " ".join(parts)
    return haystack if case_sensitive else haystack.lower()
