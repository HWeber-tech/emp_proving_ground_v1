"""Governance promotion guardrails enforcing policy ledger and regime coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from src.governance.policy_ledger import (
    PolicyLedgerRecord,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.understanding.decision_diary import DecisionDiaryEntry, DecisionDiaryStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.governance.strategy_registry import StrategyStatus


_UTC = timezone.utc
_SECONDS_PER_DAY = 86400.0


_STAGE_ORDER: Mapping[PolicyLedgerStage, int] = {
    PolicyLedgerStage.EXPERIMENT: 0,
    PolicyLedgerStage.PAPER: 1,
    PolicyLedgerStage.PILOT: 2,
    PolicyLedgerStage.LIMITED_LIVE: 3,
}


class PromotionIntegrityError(RuntimeError):
    """Raised when promotion prerequisites are not satisfied."""


@dataclass(frozen=True)
class RegimeCoverage:
    """Summary of regime observations used for promotion gating."""

    counts: Mapping[str, int]

    def missing(self, required: Sequence[str], minimum: int) -> tuple[str, ...]:
        missing: list[str] = []
        for regime in required:
            observations = self.counts.get(regime, 0)
            if observations < minimum:
                missing.append(regime)
        return tuple(missing)


class PromotionGuard:
    """Validate ledger posture and regime coverage before strategy promotion."""

    _MIN_PAPER_GREEN_DAYS = 14.0

    def __init__(
        self,
        *,
        ledger_path: str | Path,
        diary_path: str | Path,
        stage_requirements: Mapping[str, PolicyLedgerStage | str] | None = None,
        required_regimes: Iterable[str] | None = None,
        min_decisions_per_regime: int = 3,
        policy_id_resolver: Callable[[str], str] | None = None,
        regime_gate_statuses: Iterable[str] | None = None,
    ) -> None:
        self._ledger_path = Path(ledger_path)
        self._diary_path = Path(diary_path)
        self._stage_requirements = self._normalise_stage_requirements(stage_requirements)
        self._required_regimes = self._normalise_regimes(required_regimes)
        self._min_decisions_per_regime = max(0, int(min_decisions_per_regime))
        self._policy_id_resolver = policy_id_resolver or (lambda strategy_id: strategy_id)
        if regime_gate_statuses is None:
            self._regime_gate_statuses = tuple(self._stage_requirements.keys())
        else:
            self._regime_gate_statuses = tuple(
                sorted({status.strip().lower() for status in regime_gate_statuses if status and status.strip()})
            )

    @staticmethod
    def _normalise_stage_requirements(
        stage_requirements: Mapping[str, PolicyLedgerStage | str] | None,
    ) -> Mapping[str, PolicyLedgerStage]:
        if stage_requirements is None:
            stage_requirements = {
                "approved": PolicyLedgerStage.PAPER,
                "active": PolicyLedgerStage.LIMITED_LIVE,
            }
        normalised: dict[str, PolicyLedgerStage] = {}
        for status, stage in stage_requirements.items():
            key = status.strip().lower()
            if not key:
                continue
            if isinstance(stage, PolicyLedgerStage):
                normalised[key] = stage
            else:
                normalised[key] = PolicyLedgerStage.from_value(stage)
        return normalised

    @staticmethod
    def _normalise_regimes(required_regimes: Iterable[str] | None) -> tuple[str, ...]:
        if not required_regimes:
            return ()
        cleaned = []
        for regime in required_regimes:
            text = str(regime).strip().lower()
            if text:
                cleaned.append(text)
        seen: dict[str, None] = {}
        for value in cleaned:
            seen.setdefault(value, None)
        return tuple(seen.keys())

    def validate(self, strategy_id: str, target_status: "StrategyStatus | str") -> None:
        status_key = self._normalise_status(target_status)
        required_stage = self._stage_requirements.get(status_key)
        if required_stage is None:
            return

        policy_id = self._policy_id_resolver(strategy_id)
        record = self._load_ledger_record(policy_id)
        if record is None:
            raise PromotionIntegrityError(
                f"Policy ledger record for policy '{policy_id}' not found at {self._ledger_path}"
            )

        declared = record.stage
        if _STAGE_ORDER.get(declared, -1) < _STAGE_ORDER.get(required_stage, 0):
            raise PromotionIntegrityError(
                f"Policy '{policy_id}' stage '{declared.value}' below required '{required_stage.value}'"
            )

        gaps = record.audit_gaps(expected_stage=required_stage)
        if gaps:
            gap_text = ", ".join(gaps)
            raise PromotionIntegrityError(
                f"Policy '{policy_id}' audit gaps present for stage '{required_stage.value}': {gap_text}"
            )

        require_regime_gate = bool(self._required_regimes and status_key in self._regime_gate_statuses)
        require_paper_span = required_stage is PolicyLedgerStage.LIMITED_LIVE

        diary_entries: Sequence[DecisionDiaryEntry] | None = None
        if require_regime_gate or require_paper_span:
            diary_entries = self._policy_entries(policy_id)

        if require_regime_gate:
            coverage = self._regime_coverage(policy_id, diary_entries)
            missing = coverage.missing(self._required_regimes, self._min_decisions_per_regime)
            if missing:
                missing_text = ", ".join(missing)
                raise PromotionIntegrityError(
                    f"Decision diary coverage for policy '{policy_id}' missing regimes: {missing_text}"
                )

        if require_paper_span:
            paper_green_span = self._paper_green_span_days(policy_id, diary_entries)
            if paper_green_span < self._MIN_PAPER_GREEN_DAYS:
                raise PromotionIntegrityError(
                    "paper_green_gate_duration_below:" +
                    f"{paper_green_span:.2f}<{self._MIN_PAPER_GREEN_DAYS:.2f}"
                )

    def _load_ledger_record(self, policy_id: str) -> PolicyLedgerRecord | None:
        if not self._ledger_path.exists():
            return None
        store = PolicyLedgerStore(self._ledger_path)
        return store.get(policy_id)

    def _regime_coverage(
        self,
        policy_id: str,
        entries: Sequence[DecisionDiaryEntry] | None = None,
    ) -> RegimeCoverage:
        if entries is None:
            entries = self._policy_entries(policy_id)
        if not entries:
            raise PromotionIntegrityError(
                f"Decision diary entries for policy '{policy_id}' not found in {self._diary_path}"
            )
        counts: MutableMapping[str, int] = {}
        for entry in entries:
            regime = self._extract_regime(entry)
            if regime:
                counts[regime] = counts.get(regime, 0) + 1
        return RegimeCoverage(counts)

    def _policy_entries(self, policy_id: str) -> tuple[DecisionDiaryEntry, ...]:
        if not self._diary_path.exists():
            raise PromotionIntegrityError(
                f"Decision diary artifact not found at {self._diary_path}"
            )
        diary = DecisionDiaryStore(self._diary_path, publish_on_record=False)
        selected = tuple(entry for entry in diary.entries() if entry.policy_id == policy_id)
        if not selected:
            raise PromotionIntegrityError(
                f"Decision diary entries for policy '{policy_id}' not found in {self._diary_path}"
            )
        return selected

    def _paper_green_span_days(
        self,
        policy_id: str,
        entries: Sequence[DecisionDiaryEntry] | None = None,
    ) -> float:
        if entries is None:
            entries = self._policy_entries(policy_id)
        if not entries:
            raise PromotionIntegrityError(
                f"Decision diary entries for policy '{policy_id}' not found in {self._diary_path}"
            )

        current_span = 0.0
        current_start: datetime | None = None
        saw_paper_entry = False

        for entry in entries:
            stage = self._extract_stage(entry)
            if stage is not PolicyLedgerStage.PAPER:
                continue
            saw_paper_entry = True

            if self._is_paper_gate_green(entry):
                timestamp = entry.recorded_at.astimezone(_UTC)
                if current_start is None:
                    current_start = timestamp
                current_span = (timestamp - current_start).total_seconds() / _SECONDS_PER_DAY
            else:
                current_start = None
                current_span = 0.0

        if not saw_paper_entry:
            raise PromotionIntegrityError(
                f"Decision diary paper stage entries for policy '{policy_id}' not found in {self._diary_path}"
            )

        return current_span

    @staticmethod
    def _extract_regime(entry: DecisionDiaryEntry) -> str | None:
        regime_payload = entry.regime_state
        regime_value = None
        if hasattr(regime_payload, "get"):
            try:
                regime_value = regime_payload.get("regime")  # type: ignore[attr-defined]
            except Exception:
                regime_value = None
        if regime_value is None and hasattr(regime_payload, "regime"):
            regime_value = getattr(regime_payload, "regime", None)
        if not isinstance(regime_value, str):
            return None
        text = regime_value.strip().lower()
        return text or None

    @staticmethod
    def _extract_stage(entry: DecisionDiaryEntry) -> PolicyLedgerStage | None:
        metadata = entry.metadata or {}
        candidate = metadata.get("release_stage")
        if not candidate:
            release_meta = metadata.get("release_execution")
            if isinstance(release_meta, Mapping):
                candidate = release_meta.get("stage") or release_meta.get("release_stage")
        if not candidate:
            return None
        try:
            return PolicyLedgerStage.from_value(candidate)
        except Exception:
            return None

    @staticmethod
    def _extract_severity(entry: DecisionDiaryEntry) -> str:
        metadata = entry.metadata or {}
        drift_meta = metadata.get("drift_decision")
        if isinstance(drift_meta, Mapping):
            severity = drift_meta.get("severity")
            if severity:
                text = str(severity).strip().lower()
                if text:
                    return text
        return "unknown"

    @staticmethod
    def _is_forced(entry: DecisionDiaryEntry) -> bool:
        metadata = entry.metadata or {}
        drift_meta = metadata.get("drift_decision")
        if isinstance(drift_meta, Mapping):
            forced = drift_meta.get("force_paper")
            if isinstance(forced, bool):
                return forced
            if isinstance(forced, str):
                lowered = forced.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
        release_meta = metadata.get("release_execution")
        if isinstance(release_meta, Mapping):
            forced = release_meta.get("forced")
            if isinstance(forced, bool):
                return forced
            forced_reason = release_meta.get("forced_reason")
            if forced_reason:
                return True
        forced_summary = metadata.get("release_execution_forced")
        return bool(forced_summary)

    @classmethod
    def _is_paper_gate_green(cls, entry: DecisionDiaryEntry) -> bool:
        return cls._extract_severity(entry) == "normal" and not cls._is_forced(entry)

    @staticmethod
    def _normalise_status(target_status: "StrategyStatus | str") -> str:
        if hasattr(target_status, "value"):
            value = getattr(target_status, "value")
            if isinstance(value, str):
                return value.strip().lower()
        return str(target_status).strip().lower()
