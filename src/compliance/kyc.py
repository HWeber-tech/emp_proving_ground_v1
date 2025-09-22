"""KYC/AML monitor that evaluates onboarding cases and emits telemetry."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Deque, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from src.core.event_bus import Event, EventBus, get_global_bus
from src.governance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class KycSnapshotJournal(Protocol):
    """Persistence layer contract for storing KYC/AML snapshots."""

    def record_case(
        self, snapshot: Mapping[str, Any], *, strategy_id: str
    ) -> Mapping[str, Any]: ...

    def fetch_recent(
        self,
        *,
        limit: int = 5,
        strategy_id: str | None = None,
        entity_id: str | None = None,
    ) -> Sequence[Mapping[str, Any]]: ...

    def close(self) -> None: ...


def _coerce_datetime(value: Any, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return default
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [token.strip() for token in str(value).split(",") if token.strip()]


@dataclass(frozen=True)
class KycChecklistItem:
    """Single checklist requirement captured during a KYC review."""

    item_id: str
    name: str
    status: str
    severity: str = "info"
    notes: str | None = None
    due: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": self.item_id,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
        }
        if self.notes:
            payload["notes"] = self.notes
        if self.due is not None:
            payload["due"] = self.due.astimezone(UTC).isoformat()
        return payload


@dataclass(frozen=True)
class KycCaseSnapshot:
    """Immutable representation of a KYC/AML case evaluation."""

    case_id: str
    entity_id: str
    entity_type: str
    status: str
    risk_score: float
    risk_rating: str
    watchlist_hits: tuple[str, ...]
    outstanding_items: tuple[str, ...]
    checklist: tuple[KycChecklistItem, ...]
    alerts: tuple[str, ...]
    metadata: Mapping[str, Any]
    assigned_to: str | None
    last_reviewed_at: datetime
    next_review_due: datetime
    evaluated_at: datetime
    report_channel: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "status": self.status,
            "risk_score": self.risk_score,
            "risk_rating": self.risk_rating,
            "watchlist_hits": list(self.watchlist_hits),
            "outstanding_items": list(self.outstanding_items),
            "checklist": [item.as_dict() for item in self.checklist],
            "alerts": list(self.alerts),
            "metadata": dict(self.metadata),
            "assigned_to": self.assigned_to,
            "last_reviewed_at": self.last_reviewed_at.astimezone(UTC).isoformat(),
            "next_review_due": self.next_review_due.astimezone(UTC).isoformat(),
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
            "report_channel": self.report_channel,
            "markdown": self.markdown_summary(),
        }

    def markdown_summary(self) -> str:
        headline = f"**{self.status}** – risk {self.risk_rating} ({self.risk_score:.1f})"
        lines = [
            f"### KYC case `{self.case_id}` for `{self.entity_id}`",
            headline,
            f"- Assigned to: `{self.assigned_to}`" if self.assigned_to else "- Unassigned",
            f"- Last review: {self.last_reviewed_at.astimezone(UTC).isoformat()}",
            f"- Next review due: {self.next_review_due.astimezone(UTC).isoformat()}",
        ]
        if self.watchlist_hits:
            lines.append(f"- Watchlist hits: {', '.join(sorted(self.watchlist_hits))}")
        if self.outstanding_items:
            lines.append(f"- Outstanding requirements: {', '.join(sorted(self.outstanding_items))}")
        if self.alerts:
            lines.append(f"- Alerts: {', '.join(self.alerts)}")
        return "\n".join(lines)


class KycAmlMonitor:
    """Evaluate KYC/AML cases, emit telemetry, and persist audit snapshots."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        report_channel: str = "telemetry.compliance.kyc",
        audit_logger: AuditLogger | None = None,
        strategy_id: str = "default",
        snapshot_journal: KycSnapshotJournal | None = None,
        history_limit: int = 20,
    ) -> None:
        self.event_bus = event_bus
        self.report_channel = report_channel
        self.audit_logger = audit_logger
        self.strategy_id = strategy_id
        self.snapshot_journal = snapshot_journal
        self._history: Deque[KycCaseSnapshot] = deque(maxlen=max(1, history_limit))
        self._last_snapshot: KycCaseSnapshot | None = None
        self._last_journal_entry: Mapping[str, Any] | None = None

    def evaluate_case(
        self,
        payload: Mapping[str, Any],
        *,
        observed_at: datetime | None = None,
    ) -> KycCaseSnapshot:
        snapshot = self._build_snapshot(payload, observed_at=observed_at)
        self._history.appendleft(snapshot)
        self._last_snapshot = snapshot
        self._emit_snapshot(snapshot)
        self._record_audit(snapshot)
        self._persist_snapshot(snapshot)
        return snapshot

    def _build_snapshot(
        self, payload: Mapping[str, Any], *, observed_at: datetime | None
    ) -> KycCaseSnapshot:
        observed = observed_at or datetime.now(tz=UTC)

        case_id = str(payload.get("case_id") or uuid4())
        entity_id = str(payload.get("entity_id") or "UNKNOWN")
        entity_type = str(payload.get("entity_type") or "client")
        assigned_to = (
            str(payload.get("assigned_to")).strip() if payload.get("assigned_to") else None
        )

        checklist_items = self._parse_checklist(payload.get("checklist"))
        outstanding = tuple(
            item.name for item in checklist_items if item.status.lower() != "complete"
        )
        watchlist_hits = tuple(_as_list(payload.get("watchlist_hits")))
        alerts = tuple(_as_list(payload.get("alerts")))

        risk_score = _coerce_float(payload.get("risk_score"), 0.0)
        risk_rating = str(payload.get("risk_rating") or "").upper()
        if not risk_rating:
            risk_rating = self._grade_risk(risk_score, outstanding, watchlist_hits)

        escalated = bool(watchlist_hits) or any(
            item.severity.lower() in {"critical", "high"}
            for item in checklist_items
            if item.status.lower() != "complete"
        )
        status = str(payload.get("status") or "").upper()
        if not status:
            status = (
                "ESCALATED" if escalated else ("REVIEW_REQUIRED" if outstanding else "APPROVED")
            )

        last_review = _coerce_datetime(payload.get("last_reviewed_at"), observed)
        review_frequency_days = payload.get("review_frequency_days")
        try:
            frequency = int(review_frequency_days)
        except Exception:
            frequency = 365
        next_due = _coerce_datetime(payload.get("next_review_due"))
        if next_due is None and last_review is not None:
            next_due = last_review + timedelta(days=max(frequency, 1))
        if next_due is None:
            next_due = observed + timedelta(days=max(frequency, 1))

        metadata = (
            dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        )

        snapshot = KycCaseSnapshot(
            case_id=case_id,
            entity_id=entity_id,
            entity_type=entity_type,
            status=status,
            risk_score=risk_score,
            risk_rating=risk_rating,
            watchlist_hits=watchlist_hits,
            outstanding_items=outstanding,
            checklist=tuple(checklist_items),
            alerts=alerts,
            metadata=metadata,
            assigned_to=assigned_to,
            last_reviewed_at=last_review,
            next_review_due=next_due,
            evaluated_at=observed,
            report_channel=self.report_channel,
        )
        return snapshot

    def _parse_checklist(self, raw: Any) -> list[KycChecklistItem]:
        items: list[KycChecklistItem] = []
        if not raw:
            return items
        iterable: Iterable[Mapping[str, Any]]
        if isinstance(raw, Mapping):
            iterable = [raw]
        elif isinstance(raw, Iterable):
            iterable = [item for item in raw if isinstance(item, Mapping)]
        else:
            return items

        for entry in iterable:
            item_id = str(entry.get("item_id") or uuid4())
            name = str(entry.get("name") or item_id)
            status = str(entry.get("status") or "UNKNOWN").upper()
            severity = str(entry.get("severity") or "info").lower()
            notes_raw = entry.get("notes")
            notes = str(notes_raw) if notes_raw is not None else None
            due = _coerce_datetime(entry.get("due"))
            items.append(
                KycChecklistItem(
                    item_id=item_id,
                    name=name,
                    status=status,
                    severity=severity,
                    notes=notes,
                    due=due,
                )
            )
        return items

    def _grade_risk(
        self,
        risk_score: float,
        outstanding: Sequence[str],
        watchlist_hits: Sequence[str],
    ) -> str:
        if watchlist_hits:
            return "CRITICAL"
        if risk_score >= 80:
            return "CRITICAL"
        if risk_score >= 60:
            return "HIGH"
        if risk_score >= 40:
            return "MODERATE"
        if outstanding:
            return "MODERATE"
        return "LOW"

    def _emit_snapshot(self, snapshot: KycCaseSnapshot) -> None:
        payload = snapshot.as_dict()
        event = Event(
            type=self.report_channel,
            payload=payload,
            source="kyc_aml_monitor",
        )

        published = False
        publish_from_sync = getattr(self.event_bus, "publish_from_sync", None)
        if callable(publish_from_sync):
            try:
                published = publish_from_sync(event) is not None
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("KYC monitor publish_from_sync failed", exc_info=True)

        if not published:
            try:
                topic_bus = get_global_bus()
                topic_bus.publish_sync(
                    self.report_channel,
                    payload,
                    source="kyc_aml_monitor",
                )
            except Exception:  # pragma: no cover - optional background bus
                logger.debug("KYC telemetry publish failed", exc_info=True)

    def _record_audit(self, snapshot: KycCaseSnapshot) -> None:
        if self.audit_logger is None:
            return
        try:
            self.audit_logger.log_compliance_check(
                check_type="kyc_case",
                strategy_id=self.strategy_id,
                passed=snapshot.status == "APPROVED",
                violations=list(snapshot.outstanding_items) + list(snapshot.watchlist_hits),
                metadata={
                    "case_id": snapshot.case_id,
                    "entity_id": snapshot.entity_id,
                    "risk_rating": snapshot.risk_rating,
                },
            )
        except Exception:  # pragma: no cover - audit logging best effort
            logger.debug("Failed to write KYC audit log", exc_info=True)

    def _persist_snapshot(self, snapshot: KycCaseSnapshot) -> None:
        if self.snapshot_journal is None:
            return
        try:
            entry = self.snapshot_journal.record_case(
                snapshot.as_dict(), strategy_id=self.strategy_id
            )
        except Exception:  # pragma: no cover - persistence optional
            logger.debug("Failed to persist KYC snapshot", exc_info=True)
        else:
            self._last_journal_entry = entry

    def summary(self) -> dict[str, Any]:
        history = list(self._history)
        payload: dict[str, Any] = {
            "report_channel": self.report_channel,
            "recent": [snapshot.as_dict() for snapshot in history],
            "open_cases": sum(1 for snapshot in history if snapshot.outstanding_items),
            "escalations": sum(1 for snapshot in history if snapshot.status == "ESCALATED"),
        }
        if self._last_snapshot is not None:
            payload["last_snapshot"] = self._last_snapshot.as_dict()
        if self._last_journal_entry is not None:
            payload["journal"] = {"last_entry": dict(self._last_journal_entry)}
        elif self.snapshot_journal is not None:
            try:
                recent = [dict(entry) for entry in self.snapshot_journal.fetch_recent(limit=5)]
                if recent:
                    payload["journal"] = {"recent_entries": recent}
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to fetch KYC journal summary", exc_info=True)
        return payload

    def close(self) -> None:
        if self.snapshot_journal is not None:
            try:
                self.snapshot_journal.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to close KYC snapshot journal", exc_info=True)


__all__ = [
    "KycAmlMonitor",
    "KycCaseSnapshot",
    "KycChecklistItem",
]
