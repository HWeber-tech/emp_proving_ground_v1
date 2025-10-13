"""Decision diary capture pipeline for the AlphaTrade understanding loop.

The roadmap requires automated decision diaries that preserve pre/post routing
context, probe ownership, and reviewer-facing narratives.  The store provided
here persists entries to JSON while validating probe references against the
registry shipped in :mod:`src.understanding.probe_registry`.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from src.thinking.adaptation.policy_router import PolicyDecision, RegimeState
from src.understanding.probe_registry import ProbeDefinition, ProbeRegistry
from src.core.event_bus import Event, EventBus
from src.operations.event_bus_failover import EventPublishError, publish_event_with_failover

if TYPE_CHECKING:
    from src.understanding.belief import BeliefState

logger = logging.getLogger(__name__)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for runtime compatibility
    UTC = timezone.utc  # type: ignore[assignment]

__all__ = [
    "ProbeActivation",
    "DecisionDiaryEntry",
    "DecisionDiaryStore",
]


def _normalise_timestamp(value: datetime | str | None, *, default: Callable[[], datetime]) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            logger.debug("Failed to parse diary timestamp %r; falling back to default", value)
            return default()
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return default()


def _serialise_policy_decision(decision: PolicyDecision | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(decision, Mapping):
        payload = dict(decision)
        payload.setdefault("rationale", payload.get("rationale", ""))
        timestamp_value = payload.get("decision_timestamp")
        if isinstance(timestamp_value, datetime):
            payload["decision_timestamp"] = timestamp_value.astimezone(UTC).isoformat()
        elif isinstance(timestamp_value, str):
            payload["decision_timestamp"] = _normalise_timestamp(
                timestamp_value,
                default=lambda: datetime.now(tz=UTC),
            ).isoformat()
        else:
            payload["decision_timestamp"] = None
        return payload
    return {
        "tactic_id": decision.tactic_id,
        "parameters": dict(decision.parameters),
        "selected_weight": float(decision.selected_weight),
        "guardrails": dict(decision.guardrails),
        "rationale": decision.rationale,
        "experiments_applied": list(decision.experiments_applied),
        "reflection_summary": dict(decision.reflection_summary),
        "weight_breakdown": dict(decision.weight_breakdown),
        "decision_timestamp": (
            decision.decision_timestamp.astimezone(UTC).isoformat()
            if decision.decision_timestamp
            else None
        ),
    }


def _serialise_regime_state(regime: RegimeState | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(regime, Mapping):
        payload = dict(regime)
        confidence = payload.get("confidence")
        try:
            payload["confidence"] = float(confidence)
        except (TypeError, ValueError):
            payload["confidence"] = confidence
        return payload
    return {
        "regime": regime.regime,
        "confidence": float(regime.confidence),
        "features": dict(regime.features),
        "timestamp": regime.timestamp.astimezone(UTC).isoformat(),
    }


def _serialise_belief_state(state: BeliefState | Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return dict(state)
    return dict(state.as_dict())


@dataclass(slots=True, frozen=True)
class ProbeActivation:
    """Runtime metadata tying a diary entry to a probe invocation."""

    probe_id: str
    status: str
    severity: str | None = None
    owner: str | None = None
    contact: str | None = None
    runbook: str | None = None
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "probe_id": self.probe_id,
            "status": self.status,
            "notes": list(self.notes),
        }
        if self.severity:
            payload["severity"] = self.severity
        if self.owner:
            payload["owner"] = self.owner
        if self.contact:
            payload["contact"] = self.contact
        if self.runbook:
            payload["runbook"] = self.runbook
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: ProbeRegistry | None = None,
    ) -> "ProbeActivation":
        probe_id = str(payload.get("probe_id"))
        if not probe_id:
            raise ValueError("probe_id is required for probe activation")
        status = str(payload.get("status", "unknown"))
        severity = payload.get("severity")
        owner = payload.get("owner")
        contact = payload.get("contact")
        runbook = payload.get("runbook")
        notes_payload = payload.get("notes", ())
        metadata_payload = payload.get("metadata")

        notes: tuple[str, ...]
        if isinstance(notes_payload, (list, tuple, set)):
            notes = tuple(str(note).strip() for note in notes_payload if str(note).strip())
        else:
            notes = ()

        if isinstance(metadata_payload, Mapping):
            metadata_dict: MutableMapping[str, Any] = {
                str(key): value for key, value in metadata_payload.items()
            }
        else:
            metadata_dict = {}

        definition: ProbeDefinition | None = registry.get(probe_id) if registry else None
        if definition is not None:
            owner = owner or definition.owner
            contact = contact or definition.contact
            severity = severity or definition.severity
            runbook = runbook or definition.runbook
            if definition.tags:
                metadata_dict = {"probe_tags": list(definition.tags)} | metadata_dict
            if definition.metadata:
                metadata_dict = dict(definition.metadata) | metadata_dict

        owner_value = str(owner).strip() if owner else None
        contact_value = str(contact).strip() if contact else None
        severity_value = str(severity).strip() if severity else None
        runbook_value = str(runbook).strip() if runbook else None

        return cls(
            probe_id=probe_id,
            status=status,
            severity=severity_value or None,
            owner=owner_value or None,
            contact=contact_value or None,
            runbook=runbook_value or None,
            notes=notes,
            metadata=dict(metadata_dict),
        )


@dataclass(slots=True, frozen=True)
class DecisionDiaryEntry:
    """Persisted snapshot describing a single routing decision."""

    entry_id: str
    recorded_at: datetime
    policy_id: str
    decision: Mapping[str, Any]
    regime_state: Mapping[str, Any]
    outcomes: Mapping[str, Any]
    belief_state: Mapping[str, Any] | None = None
    probes: tuple[ProbeActivation, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "entry_id": self.entry_id,
            "recorded_at": self.recorded_at.astimezone(UTC).isoformat(),
            "policy_id": self.policy_id,
            "decision": dict(self.decision),
            "regime_state": dict(self.regime_state),
            "outcomes": dict(self.outcomes),
            "probes": [activation.as_dict() for activation in self.probes],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.belief_state is not None:
            payload["belief_state"] = dict(self.belief_state)
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"# Decision diary entry — {self.entry_id}",
            f"- Recorded at: {self.recorded_at.astimezone(UTC).isoformat()}",
            f"- Policy ID: {self.policy_id}",
            f"- Tactic ID: {self.decision.get('tactic_id', 'unknown')}",
        ]
        rationale = self.decision.get("rationale")
        if rationale:
            lines.append(f"- Rationale: {rationale}")
        lines.append("")
        lines.append("## Regime context")
        lines.append(f"- Regime: {self.regime_state.get('regime', 'unknown')}")
        confidence = self.regime_state.get("confidence")
        if isinstance(confidence, (int, float)):
            lines.append(f"- Confidence: {confidence:.3f}")
        features = self.regime_state.get("features")
        if isinstance(features, Mapping) and features:
            lines.append("- Features:")
            for key, value in sorted(features.items()):
                lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("## Decision parameters")
        parameters = self.decision.get("parameters")
        if isinstance(parameters, Mapping) and parameters:
            for key, value in sorted(parameters.items()):
                lines.append(f"- {key}: {value}")
        guardrails = self.decision.get("guardrails")
        if isinstance(guardrails, Mapping) and guardrails:
            lines.append("- Guardrails:")
            for key, value in sorted(guardrails.items()):
                lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("## Outcomes")
        if self.outcomes:
            for key, value in sorted(self.outcomes.items()):
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- No outcomes recorded.")
        if self.belief_state:
            lines.append("")
            lines.append("## Belief snapshot")
            belief_meta = self.belief_state.get("metadata") if isinstance(self.belief_state, Mapping) else {}
            lines.append(f"- Version: {self.belief_state.get('version', 'unknown')}")
            symbol = self.belief_state.get("symbol")
            if symbol:
                lines.append(f"- Symbol: {symbol}")
            if isinstance(belief_meta, Mapping) and belief_meta:
                lines.append("- Metadata:")
                for key, value in sorted(belief_meta.items()):
                    lines.append(f"  - {key}: {value}")
        if self.probes:
            lines.append("")
            lines.append("## Probe ownership")
            lines.append("| Probe ID | Status | Owner | Severity | Contact | Notes |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for activation in self.probes:
                notes = "; ".join(activation.notes) if activation.notes else "—"
                owner = activation.owner or "—"
                severity = activation.severity or "—"
                contact = activation.contact or "—"
                lines.append(
                    f"| {activation.probe_id} | {activation.status} | {owner} | {severity} | {contact} | {notes} |"
                )
        if self.metadata:
            lines.append("")
            lines.append("## Metadata")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- {key}: {value}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: ProbeRegistry | None = None,
    ) -> "DecisionDiaryEntry":
        entry_id = str(payload.get("entry_id")) or str(uuid.uuid4())
        recorded_at = _normalise_timestamp(payload.get("recorded_at"), default=lambda: datetime.now(tz=UTC))
        policy_id = str(payload.get("policy_id", "unknown"))
        decision_payload = payload.get("decision", {})
        if not isinstance(decision_payload, Mapping):
            decision_payload = {}
        regime_payload = payload.get("regime_state", {})
        if not isinstance(regime_payload, Mapping):
            regime_payload = {}
        outcomes_payload = payload.get("outcomes", {})
        if not isinstance(outcomes_payload, Mapping):
            outcomes_payload = {}
        belief_payload = payload.get("belief_state")
        if belief_payload is not None and not isinstance(belief_payload, Mapping):
            belief_payload = None
        notes_payload = payload.get("notes", ())
        metadata_payload = payload.get("metadata", {})
        probes_payload = payload.get("probes", ())

        notes: tuple[str, ...]
        if isinstance(notes_payload, (list, tuple, set)):
            notes = tuple(str(note).strip() for note in notes_payload if str(note).strip())
        else:
            notes = ()

        metadata: Mapping[str, Any]
        if isinstance(metadata_payload, Mapping):
            metadata = {str(key): value for key, value in metadata_payload.items()}
        else:
            metadata = {}

        probes: list[ProbeActivation] = []
        if isinstance(probes_payload, Sequence):
            for raw in probes_payload:
                if not isinstance(raw, Mapping):
                    continue
                try:
                    probes.append(ProbeActivation.from_mapping(raw, registry=registry))
                except Exception as exc:
                    logger.warning("Skipping invalid probe activation for %s: %s", entry_id, exc)

        return cls(
            entry_id=entry_id,
            recorded_at=recorded_at,
            policy_id=policy_id,
            decision=dict(decision_payload),
            regime_state=dict(regime_payload),
            outcomes=dict(outcomes_payload),
            belief_state=dict(belief_payload) if isinstance(belief_payload, Mapping) else None,
            probes=tuple(probes),
            notes=notes,
            metadata=metadata,
        )


class DecisionDiaryStore:
    """JSON-backed decision diary store with probe registry integration."""

    def __init__(
        self,
        path: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
        probe_registry: ProbeRegistry | None = None,
        event_bus: EventBus | None = None,
        event_type: str = "governance.decision_diary.recorded",
        event_source: str = "understanding.decision_diary",
        publish_on_record: bool = True,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._entries: MutableMapping[str, DecisionDiaryEntry] = {}
        self._probe_registry = probe_registry or ProbeRegistry()
        self._event_bus = event_bus
        self._event_type = event_type
        self._event_source = event_source
        self._publish_on_record = publish_on_record
        self._load()

    @property
    def probe_registry(self) -> ProbeRegistry:
        return self._probe_registry

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = self._path.read_text(encoding="utf-8")
        if not data.strip():
            return
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise ValueError("Decision diary payload must be a mapping")
        registry_payload = payload.get("probe_registry")
        if isinstance(registry_payload, Mapping):
            try:
                probes_payload = registry_payload.get("probes", {})
                if not isinstance(probes_payload, Mapping):
                    probes_payload = {}
                loaded_registry = ProbeRegistry.from_definitions(
                    (
                        ProbeDefinition.from_dict({"probe_id": probe_id, **definition})
                        for probe_id, definition in probes_payload.items()
                        if isinstance(definition, Mapping)
                    ),
                    generated_at=_normalise_timestamp(
                        registry_payload.get("generated_at"),
                        default=lambda: datetime.now(tz=UTC),
                    ),
                )
            except Exception as exc:
                logger.warning("Failed to load probe registry from diary store: %s", exc)
            else:
                for definition in loaded_registry.definitions():
                    self._probe_registry.upsert(definition)
        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, Sequence):
            raise ValueError("Decision diary 'entries' field must be a list")
        for entry_payload in entries_payload:
            if not isinstance(entry_payload, Mapping):
                continue
            try:
                entry = DecisionDiaryEntry.from_dict(entry_payload, registry=self._probe_registry)
            except Exception as exc:
                logger.warning("Skipping invalid diary entry: %s", exc)
                continue
            self._entries[entry.entry_id] = entry

    def _dump(self) -> None:
        payload = {
            "generated_at": self._now().isoformat(),
            "entries": [entry.as_dict() for entry in sorted(self._entries.values(), key=lambda e: e.recorded_at)],
            "probe_registry": self._probe_registry.as_dict(),
        }
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def exists(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def get(self, entry_id: str) -> DecisionDiaryEntry | None:
        return self._entries.get(entry_id)

    def entries(self) -> tuple[DecisionDiaryEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.recorded_at))

    def record(
        self,
        *,
        policy_id: str,
        decision: PolicyDecision | Mapping[str, Any],
        regime_state: RegimeState | Mapping[str, Any],
        outcomes: Mapping[str, Any] | None = None,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
        probes: Iterable[Mapping[str, Any]] | None = None,
        notes: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        entry_id: str | None = None,
        recorded_at: datetime | None = None,
    ) -> DecisionDiaryEntry:
        entry_identifier = entry_id or f"dd-{self._now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        timestamp = _normalise_timestamp(recorded_at, default=self._now)
        decision_payload = _serialise_policy_decision(decision)
        regime_payload = _serialise_regime_state(regime_state)
        outcomes_payload = dict(outcomes or {})
        belief_payload = _serialise_belief_state(belief_state)
        notes_tuple: tuple[str, ...] = tuple(str(note).strip() for note in (notes or ()) if str(note).strip())
        metadata_payload: Mapping[str, Any] = {str(k): v for k, v in (metadata or {}).items()}

        probe_activations: list[ProbeActivation] = []
        for probe in probes or ():
            try:
                probe_activations.append(ProbeActivation.from_mapping(probe, registry=self._probe_registry))
            except Exception as exc:
                logger.warning("Skipping probe activation for %s: %s", entry_identifier, exc)

        entry = DecisionDiaryEntry(
            entry_id=entry_identifier,
            recorded_at=timestamp,
            policy_id=policy_id,
            decision=decision_payload,
            regime_state=regime_payload,
            outcomes=outcomes_payload,
            belief_state=belief_payload,
            probes=tuple(probe_activations),
            notes=notes_tuple,
            metadata=metadata_payload,
        )
        self._entries[entry.entry_id] = entry
        self._dump()
        logger.info(
            "Decision diary entry recorded",
            extra={"entry_id": entry.entry_id, "policy_id": policy_id},
        )
        self._publish_entry(entry)
        return entry

    def merge_metadata(
        self,
        entry_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> DecisionDiaryEntry:
        """Merge additional metadata into an existing diary entry."""

        if not metadata:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Decision diary entry {entry_id} not found")
            return entry

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

        entry = self._entries.get(entry_id)
        if entry is None:
            raise KeyError(f"Decision diary entry {entry_id} not found")

        merged = dict(entry.metadata)
        merged.update({str(key): value for key, value in metadata.items()})
        new_entry = replace(entry, metadata=merged)
        self._entries[entry.entry_id] = new_entry
        self._dump()
        logger.info(
            "Decision diary entry metadata merged",
            extra={"entry_id": entry.entry_id},
        )
        self._publish_entry(new_entry)
        return new_entry

    def export_markdown(self, *, since: datetime | None = None) -> str:
        since_ts = _normalise_timestamp(since, default=lambda: datetime.fromtimestamp(0, tz=UTC)) if since else None
        entries = [entry for entry in self.entries() if since_ts is None or entry.recorded_at >= since_ts]
        if not entries:
            return "# Decision diary export\n\n_No entries available for the selected window._\n"
        chunks = [entry.to_markdown() for entry in entries]
        return "\n---\n\n".join(chunks)

    def export_json(self, *, since: datetime | None = None, indent: int = 2) -> str:
        since_ts = _normalise_timestamp(since, default=lambda: datetime.fromtimestamp(0, tz=UTC)) if since else None
        payload = {
            "generated_at": self._now().isoformat(),
            "entries": [entry.as_dict() for entry in self.entries() if since_ts is None or entry.recorded_at >= since_ts],
        }
        return json.dumps(payload, indent=indent, sort_keys=True)

    def _publish_entry(self, entry: DecisionDiaryEntry) -> None:
        if not self._publish_on_record or self._event_bus is None:
            return

        event = Event(
            type=self._event_type,
            payload={
                "generated_at": self._now().isoformat(),
                "entry": entry.as_dict(),
                "markdown": entry.to_markdown(),
                "probe_registry": self._probe_registry.as_dict(),
            },
            source=self._event_source,
        )

        try:
            publish_event_with_failover(
                event_bus=self._event_bus,
                event=event,
                logger=logger,
                runtime_fallback_message="Decision diary event publish failed via runtime bus; attempting global bus",
                runtime_unexpected_message="Unexpected failure publishing decision diary event via runtime bus",
                runtime_none_message="Runtime bus returned None for decision diary event; attempting global bus",
                global_not_running_message="Global event bus not running for decision diary event",
                global_unexpected_message="Unexpected failure publishing decision diary event via global bus",
            )
        except EventPublishError as exc:
            logger.warning(
                "Decision diary event publish failed",
                exc_info=exc,
                extra={"entry_id": entry.entry_id, "policy_id": entry.policy_id},
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "Unexpected error while publishing decision diary event",
                exc_info=exc,
                extra={"entry_id": entry.entry_id, "policy_id": entry.policy_id},
            )
